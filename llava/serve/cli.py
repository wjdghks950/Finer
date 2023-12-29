import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init, postprocess_llava_output
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.eval.eval_inaturalist import remove_ans_prefix

from PIL import Image
from tqdm import tqdm

import requests
import os
import json
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from prompts import PROMPT_DICT


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = None

    # iNaturalist dataset inference
    if args.data_dir is not None:
        pred_dict = {}

        task_types = ["high_coarse", "coarse", "fine", "attr_gen"]
        task_type_id = args.task_type_id

        # TODO: Implement the `attr_seek` case (1. Ask for attributes -> 3. Ask for lower-level concept) (i.e., ask the details)
        use_prompt = args.use_prompt
        prompt_types = ["cot_0shot", "cot_fewshot", "attr_seek"]
        prompt_type_id = args.prompt_type_id

        task_name = f"{prompt_types[prompt_type_id]}_{task_types[task_type_id]}" if use_prompt else f"{task_types[task_type_id]}"
        out_path = f"../preds/{args.data_dir.split('/')[-1]}_{task_name}_output.json"
        if task_types[task_type_id] == 3:
            out_path = f"../preds/{args.data_dir.split('/')[-1]}_attr_gen_{args.input_type}_output.json"
        user_prompt = PROMPT_DICT[task_name]

        if prompt_types[prompt_type_id] == "attr_seek":
            args.dialogue_mode = 'multi'

        if task_type_id in [1, 2, 3]:  # Inject higher-level granularity output into finer-level granularity input
            if args.use_gold_coarse and task_type_id == 2:
                # Use the GPT-3.5-turbo generated Coarse-grained labels (treat them as gold)
                coarse_lbl_dir = args.coarse_lbl_dir
                coarse_lbl_file = args.coarse_lbl_file
                with open(os.path.join(coarse_lbl_dir, coarse_lbl_file), "r") as fp:
                    coarse_out_dict = json.load(fp)
                    fp.close()
            elif task_type_id in [1,2]:
                # Use the LLaVA-predicted Coarse-grained outputs
                coarse_out_path = f"../preds/{args.data_dir.split('/')[-1]}_{task_types[task_type_id - 1]}_output.json"
                with open(coarse_out_path, "r") as fp:
                    coarse_out_dict = json.load(fp)
                    fp.close()
            
            placeholder_str = "{concept_placeholder}"

        # Load the original iNaturalist dataset
        path = os.path.join(args.data_dir, args.data_path)
        print(f"Loading from [ {path} ]")
        with open(os.path.join(args.data_dir, args.data_path), "r") as fp:
            dataset = json.load(fp)
            fp.close()

        data_iter = dataset['annotations']
        
        if task_type_id == 3:
            print("(demo) dataset.keys() : ", dataset.keys())
            print("categories (len) : ", len(dataset['categories']))
            print("categories (sample) : ", dataset['categories'][0])
            data_iter = dataset['categories']
        

        for idx, data in enumerate(data_iter):
            # dataset.keys() - dict_keys(['info', 'images', 'categories', 'annotations', 'licenses'])
            # print()
            img_dict = dataset['images'][idx]

            if task_type_id in [1, 2]:
                # For the "coarse" and "fine" cases, inject the previous output into the current prompt (or gold if --use_gold_coarse)
                user_prompt = PROMPT_DICT[task_name]
                user_prompt = user_prompt.replace(placeholder_str, remove_ans_prefix(coarse_out_dict[str(idx)], prefix="Answer:"))
                
                if prompt_types[prompt_type_id] == "attr_seek":
                    aseek_init_prompt = PROMPT_DICT[prompt_types[prompt_type_id]]
                    aseek_init_prompt = aseek_init_prompt.replace(placeholder_str, remove_ans_prefix(coarse_out_dict[str(idx)], prefix="Answer:"))
                
                image = load_image(os.path.join(args.data_dir, img_dict['file_name']))
                # Similar operation in model_worker.py
                image_tensor = process_images([image], image_processor, args)

                # print(f"======idx: {idx} | image_tensor (shape) : {image_tensor.shape}======")
                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            elif task_type_id == 3:
                # For the "attr_gen" case, simply replace the user_prompt with binomial nomenclature (or common name)
                user_prompt = PROMPT_DICT[task_name]
                binomial_name = data['name']
                common_name = data['common_name']
                user_prompt = user_prompt.replace(placeholder_str, binomial_name) if args.input_type == "binomial" \
                                else user_prompt.replace(placeholder_str, common_name)
            
            inp = user_prompt

            # print("PRED OUTPUT :: ", coarse_out_dict[str(idx)])
            # print(f"USER_PROMPT : {user_prompt}")
            # if prompt_types[prompt_type_id] == "attr_seek":
            #     print(f"ATTR_SEEK_PROMPT : {aseek_init_prompt}")
            # # exit()

            if args.dialogue_mode == 'multi' and prompt_types[prompt_type_id] == "attr_seek":

                # Attribute Seek - Initial message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + aseek_init_prompt
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + aseek_init_prompt
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                # Run the model for attribute extraction
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        do_sample=True,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria])

                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                conv.messages[-1][-1] = outputs  # Append the extracted attribute set answer to the input stream
                # Append the attribute-using prompt: [attr_seek_coarse, attr_seek_fine] to the input stream
                user_prompt = PROMPT_DICT[task_name]
                user_prompt = user_prompt.replace(placeholder_str, remove_ans_prefix(coarse_out_dict[str(idx)], prefix="Answer:"))
                conv.append_message(conv.roles[0], user_prompt)
                conv.append_message(conv.roles[1], None)

                # print("****ATTRIBUTE_SEEK_PROMPT (INTERM.) >> ", conv.get_prompt())

            elif args.dialogue_mode == 'single':  # Single-turn case
                if task_type_id == 3 and args.modality == "text":
                    pass  # No image token needed - Text-only attr_gen case
                else:
                    if model.config.mm_use_im_start_end:
                        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                    else:
                        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)

            if (idx + 1) % 1 == 0:
                print(f" >> conv.get_prompt() : {conv.get_prompt()}\n")

            prompt = conv.get_prompt()
            
            # In `single` turn, empty conv after one turn
            if args.dialogue_mode == 'single':
                conv.clear_message()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            # print(f"======idx: {idx} | input_ids: {input_ids.shape}======")

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor if image is not None else None,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            pred_dict[idx] = outputs.strip()
            print("==\n\nAGENT >> ", outputs)
            # print("<< pred_dict >>\n", pred_dict)
            print("==\n"*2)
            conv.clear_message()
            if idx > 2:
                exit()

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        # Save the outputs
        with open(out_path, "w") as outfp:
            json.dump(pred_dict, outfp)
            print(f"Saved `pred_dict` to {out_path}")
            outfp.close()


    else:  # Original, multi-turn dialogue setting

        image = load_image(args.image_file)
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, args)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        while True:
            try:
                inp = input(f"{roles[0]}: ")
            except EOFError:
                inp = ""
            if not inp:
                print("exit...")
                break

            print(f"{roles[1]}: ", end="")

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="val_noplant_64.json")
    parser.add_argument("--coarse_lbl_dir", type=str, default= "/home/jk100/code/ecole/gpt_output")
    parser.add_argument("--coarse_lbl_file", type=str, default="coarse_grained_lbls_gpt4_val_noplant64.json")
    parser.add_argument("--preds_dir", type=str, default="")

    parser.add_argument("--task_type_id", type=int, required=True)
    parser.add_argument("--use_prompt", action="store_true", default=False)
    parser.add_argument("--use_gold_coarse", action="store_true", default=False, help="Using GPT-3.5-turbo generated coarse-grained labels")
    parser.add_argument("--prompt_type_id", type=int, default=0)
    parser.add_argument("--input_type", type=str, help="When task_type_id=`attr_gen`, choose from [binomial or common]")
    parser.add_argument("--modality", type=str, default="text", help="When task_type_id == 3, args.modality is either `image` or `text`")

    parser.add_argument("--dialogue-mode", type=str, default='single')  # ['multi', 'single']
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)
