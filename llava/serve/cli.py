import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init, encode_image, extract_attributes, openai_gpt_call, preprocess_gpt_output
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, load_image
from llava.eval.eval_fgvc import remove_ans_prefix

from PIL import Image
from tqdm import tqdm

import jsonlines
import requests
import os
import json
import time
import random
from dotenv import load_dotenv  

from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from prompts import PROMPT_DICT

# OpenAI query module
from openai import OpenAI


# TODO: Refactor this code to incorporate the following:
# 1) Change all the paths to those received from args...
# 2) unify dataset name and file_path
# 3) use_wiki case only for gpt-4 & contain it within 'attr_gen' case
# 4) Prompt Type ID - should be contained within 'coarse' and 'fine' categories


def main(args):
    # Model
    disable_torch_init()

    data_split = None  # or "train"
    if args.data_path in ["train.json", "val.json"]:
        data_split = args.data_path.split(".")[0]
    elif "_" in args.data_path:  # e.g., "val_noplant_64.json"
        data_split = args.data_path.split("_")[0]
    elif "unified-" in args.data_path: # e.g., "unified-fgvc-aircraft-test-combined.jsonl"
        data_split = args.data_path.split("-")[-2]
    else:
        raise Exception(f"args.data_path format is incorrect : [{args.data_path}]")

    dataset_name = args.data_dir.split('/')[-1]
    if dataset_name == "inaturalist":
        dataset_txt = "inaturalist"
    elif dataset_name ==  "fgvc-aircraft-2013b":
        dataset_txt = "fgvc_aircraft"
    elif dataset_name == "CUB_200_2011":
        dataset_txt = "cub_200_2011"
    elif dataset_name == "stanford_dogs":
        dataset_txt = "stanford_dogs"
    elif dataset_name == "nabirds":
        dataset_txt = "nabirds"
    elif dataset_name == "stanford_cars":
        dataset_txt = "stanford_cars"

    if args.task_type_id <= 3:
        if args.model_base == "null":
            args.model_base = None

        if "gpt-" not in args.model_path:
            model_name = get_model_name_from_path(args.model_path)
            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
        else:  # GPT-4V
            model_name = args.model_path
            load_dotenv()  # Load environment variables from .env file
            api_key = os.getenv("OPENAI_API_KEY")  # Access the API key using the variable name defined in the .env file
            client = OpenAI(api_key=api_key)

        print(f"==========[ Loading {model_name} ] ==========")

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower() or "gpt-4" in model_name.lower():
            conv_mode = "llava_v1"  # GPT-4V uses the same `system_prompt` as llava_v1
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

        if args.data_dir is not None:
            out_preds = []

            task_types = ["high_coarse", "coarse", "fine", "attr_gen"]
            task_type_id = args.task_type_id

            use_prompt = args.use_prompt
            prompt_types = ["cot_0shot", "cot_fewshot", "attr_seek"]
            prompt_type_id = args.prompt_type_id

            task_name = f"{prompt_types[prompt_type_id]}_{task_types[task_type_id]}" if use_prompt else f"{task_types[task_type_id]}"

            out_dir = os.path.join(args.preds_dir, f"{dataset_txt}_outputs")

            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

            if task_type_id < 3:
                out_path = f"{dataset_txt}_{task_name}_{model_name}_output.jsonl"
                
                ''' Load the dataset'''
                cached_path = os.path.join(args.data_dir, args.data_path)
                print(f"[ DATASET: Loading dataset from [ {cached_path} ] ]")
                if dataset_name == 'inaturalist':
                    with open(os.path.join(args.data_dir, args.data_path), "r") as fp:
                        dataset = json.load(fp)  # dataset.keys() - dict_keys(['info', 'images', 'categories', 'annotations', 'licenses'])
                        data_iter = dataset['annotations']
                else:
                    with jsonlines.open(os.path.join(args.data_dir, args.data_path), "r") as fp:
                        data_iter = [d for d in fp]

            elif task_type_id == 3:  # task_type = 'attr_gen'
                if args.modality == "text":
                    if dataset_name == "inaturalist":
                        out_path = f"{dataset_name}_attr_gen_{args.modality}_{args.input_type}_{args.model_path}_output.jsonl"
                    else:
                        out_path = f"{dataset_name}_attr_gen_{args.modality}_{args.model_path}_output.jsonl"
                else:  # args.modality == "image"
                    out_path = f"{dataset_name}_attr_gen_{args.modality}_{args.model_path}_output.jsonl"

                if args.use_wiki:  # Using Wikipedia document as input for attribute generation
                    out_dir = os.path.join(args.preds_dir, f"parsed_{dataset_name}_outputs")
                    out_path = f"{dataset_name}_attr_gen_wiki_{args.modality}_{args.model_path}_output.jsonl"
                    cached_wiki_attr_dicts = []
                    if dataset_name == "inaturalist":
                        cached_wiki_docs_path = os.path.join(out_dir, "parsed-llava-7b-text-queried-wiki-out.json")
                    else:
                        cached_wiki_docs_path = os.path.join(out_dir, f"parsed-{dataset_name}-queried-wiki-out.jsonl")
                    if os.path.exists(cached_wiki_docs_path):
                        with jsonlines.open(cached_wiki_docs_path, "r") as cached_reader:
                            cached_wiki_attr_dicts = [d for d in cached_reader]
                    
                    # TODO: If 'cached_wiki_attr_dicts' is not ready, you need to construct it with 'wiki_linearize_attr.py' - Integrate this part
                    assert len(cached_wiki_attr_dicts) > 0

                ''' Load the dataset'''
                if dataset_name == 'inaturalist':
                    cached_path = os.path.join(args.data_dir, args.ctgr2img_path)
                    data_iter = dataset['categories']
                    ctgr2img_path = os.path.join(args.data_dir, args.ctgr2img_path) # Load the dataset['categories'] to actual image files mapped dict
                    if os.path.exists(ctgr2img_path):
                        with open(ctgr2img_path, "r") as reader_obj:
                            ctgr2img_dict = json.load(reader_obj)
                else:  # For datasets other than 'inaturalist'
                    cached_path = cached_wiki_docs_path
                    data_iter = cached_wiki_attr_dicts
                
                print(f"[ DATASET: Loading dataset from [ {cached_path} ] ]")
            
            if prompt_type_id is not None and prompt_types[prompt_type_id] == "attr_seek":
                args.dialogue_mode = 'multi'

            if task_type_id in [1, 2]:  # Inject higher-level granularity output into finer-level granularity input
                if args.use_gold_coarse:
                    # Use the gold coarse-grained labels
                    if task_type_id == 1: # coarse (gold basic)
                        if dataset_name == 'inaturalist':
                            coarse_out_dict = {str(idx): dataset['categories'][data['category_id']]['supercategory'] for idx, data in enumerate(data_iter)}
                        else:
                            coarse_out_dict = {str(data['idx']): data['basic-level-lbl'] for data in data_iter}
                    elif task_type_id == 2:  # fine (gold coarse)
                        if dataset_name == 'inaturalist':
                            coarse_lbl_file = os.path.join(args.data_dir, args.coarse_lbl_file)
                            coarse_out_dict = {}
                            with open(coarse_lbl_file, "r") as fp:
                                coarse_family_dict = json.load(fp)
                            for idx, data in enumerate(data_iter):
                                cid = data['category_id']
                                family_name = dataset['categories'][cid]['family']
                                coarse_out_dict[str(idx)] = coarse_family_dict[family_name]
                        else:
                            coarse_out_dict = {str(data['idx']): data['coarse-level-lbl'] for data in data_iter}

                elif task_type_id in [1,2]:
                    # Use the LLaVA-predicted Coarse-grained outputs
                    coarse_out_path = f"{dataset_name}_{task_types[task_type_id - 1]}_output.jsonl"
                    coarse_out_path = os.path.join(out_dir, coarse_out_path)
                    with open(coarse_out_path, "r") as fp:
                        coarse_out_dict = json.load(fp)

            # If `out_path` exists, load it up and count the number of idx it should start from!
            out_path = os.path.join(out_dir, out_path)
            stopped_idx = 0
            if os.path.exists(out_path):
                print(f"[ OUT_PATH already exists in [ {out_path} ]. Reading the file ... ]")
                with open(out_path, "r") as reader_obj:
                    lines = reader_obj.readlines()
                    out_preds = [json.loads(l) for l in lines]
                    stopped_idx = len(out_preds)


            placeholder_str = '{concept_placeholder}'

            if task_type_id < 3:
                if prompt_type_id:
                    setting_desc_text = f"[ Model: {model_name} ]_[ Task: {task_types[task_type_id]} | Prompt-Type: {prompt_types[prompt_type_id]} | {dataset_name} ]"
                else:
                    setting_desc_text = f"[ Model: {model_name} ]_[ Task: {task_types[task_type_id]} | {dataset_name} ]"
            elif task_type_id == 3:
                setting_desc_text = f"[ Model: {model_name} ]_[ Task: {task_types[task_type_id]} | {dataset_name} | {args.modality} | {args.input_type} ]"

            if stopped_idx < len(data_iter):  # Terminate if the generated file already exists
                
                with jsonlines.open(out_path, mode='a') as writer_obj:

                    for idx, data in enumerate(tqdm(data_iter, desc=setting_desc_text.strip())):
                        if stopped_idx > 0 and idx < stopped_idx:
                            continue

                        if task_type_id < 3:
                            if dataset_name == 'inaturalist':
                                img_dict = dataset['images'][idx]

                            # For the "coarse" and "fine" cases, 
                            # inject the previous output into the current prompt (or gold if --use_gold_coarse)
                            if prompt_type_id is not None and prompt_types[prompt_type_id] == "attr_seek":
                                user_prompt = PROMPT_DICT[task_name]
                            else:
                                user_prompt = PROMPT_DICT[task_name + f"_{dataset_txt}"]

                            if task_type_id in [1,2]:
                                coarse_lbl = coarse_out_dict[str(idx)]
                                if type(coarse_out_dict[str(idx)]) == list and len(coarse_out_dict) > 0:
                                    coarse_lbl = random.choice(coarse_out_dict[str(idx)])
                                user_prompt = user_prompt.replace(placeholder_str, remove_ans_prefix(coarse_lbl, prefix="Answer:"))

                            if prompt_type_id is not None and prompt_types[prompt_type_id] == "attr_seek":
                                aseek_init_prompt = PROMPT_DICT[prompt_types[prompt_type_id]]
                                coarse_label = coarse_out_dict[str(idx)]
                                if type(coarse_out_dict[str(idx)]) == list:
                                    coarse_label = "/".join(coarse_out_dict[str(idx)])
                                aseek_init_prompt = aseek_init_prompt.replace(placeholder_str, remove_ans_prefix(coarse_label, prefix="Answer:"))

                            # Load the input image 
                            if dataset_name == 'inaturalist':
                                image_path = os.path.join(args.data_dir, img_dict['file_name'])
                            else:
                                image_path = data['img_path']

                        elif task_type_id == 3 and args.modality == "image":
                            
                            user_prompt = PROMPT_DICT[task_name + "_" + args.modality]  # "attr_gen_image"
                            user_prompt = user_prompt.replace(placeholder_str, data['supercategory'])  # e.g., Insects, Birds, Animalia, Mollusks

                            image_path = os.path.join(args.data_dir, data_split, data['image_dir_name'], ctgr2img_dict[data['image_dir_name']][0])

                        elif task_type_id == 3 and args.modality == "text":
                            # For the "attr_gen" & "modality == text" case, 
                            # simply replace the user_prompt with the concept name
                            if args.use_wiki:
                                user_prompt = PROMPT_DICT[task_name + "_wiki"]
                            else:
                                user_prompt = PROMPT_DICT[task_name]
                            binomial_name = data['name'] if dataset_name == 'inaturalist' else ""
                            common_name = data['common_name'] if dataset_name == 'inaturalist' else data['query']
                            user_prompt = user_prompt.replace(placeholder_str, binomial_name) if args.input_type == "binomial" \
                                            else user_prompt.replace(placeholder_str, common_name)
                        
                        inp = user_prompt

                        # load and process image only when the model is not GPT-4 (no need for image processing)
                        if 'gpt-4' not in args.model_path:
                            image = load_image(image_path)
                            image_tensor = process_images([image], image_processor, args)  # Similar operation in model_worker.py
                            # print(f"====== idx: {idx} | image_tensor (shape) : {image_tensor.shape} ======")
                            if type(image_tensor) is list:
                                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

                        # if prompt_type_id is not None and prompt_types[prompt_type_id] == "attr_seek":
                        #     print(f"ATTR_SEEK_PROMPT : {aseek_init_prompt}")

                        # print(f"TASK_NAME : {task_name}")
                        # print(f"USER_PROMPT : {user_prompt}")
                        # print(f"IMG_PATH : {image_path}")
                        # print(f"DATA: {data}")

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
                            user_prompt = user_prompt.replace(placeholder_str, remove_ans_prefix(coarse_label, prefix="Answer:"))
                            conv.append_message(conv.roles[0], user_prompt)
                            conv.append_message(conv.roles[1], None)
                            # print("===== INTERIM CONVERSATION :: ", conv)

                        elif args.dialogue_mode == 'single':  # Single-turn case
                            if task_type_id == 3 and args.modality == "text":
                                pass  # No image token needed - Text-only attr_gen case
                            elif "gpt-4" in args.model_path:
                                pass
                            else:
                                if model.config.mm_use_im_start_end:
                                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                                else:
                                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                            conv.append_message(conv.roles[0], inp)
                            conv.append_message(conv.roles[1], None)

                        if idx == 0:
                            print(f" >> conv.get_prompt() : {conv.get_prompt()}\n")
                            
                        if "gpt-4" not in args.model_path:
                            prompt = conv.get_prompt()
                        else:
                            system_prompt = conv.system
                            user_prompt = inp
                            if args.use_wiki:
                                user_prompt += "Document: " + cached_wiki_attr_dicts[idx]['binomial_wiki_doc'] if dataset_name == "inaturalist" \
                                        else "Document: " + cached_wiki_attr_dicts[idx]['common_wiki_doc']

                        # In `single` turn, empty conv after one turn
                        if args.dialogue_mode == 'single':
                            conv.clear_message()
                        
                        if "llava" in args.model_path:
                            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                            keywords = [stop_str]
                            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if args.text_streamer else None

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
                            out_dict = {"idx": idx, "text": outputs.strip()}
                            conv.clear_message()
                            # print("OUTPUTS : ", out_dict)
                            # print("==\n\nAGENT >> ", outputs)
                            # print("<< pred_dict >>\n", pred_dict)
                            # print("==\n"*2)

                        elif "gpt-4" in args.model_path:
                            max_tokens = args.max_new_tokens
                            temp = args.temperature
                            image_path = image_path if args.modality == "image" or task_type_id < 3 else None
                            raw_response, response = openai_gpt_call(client, system_prompt, user_prompt, model_name, image_path, max_tokens, temp)
                            response_dict = response.dict() if type(response) != dict else response
                            out_dict = {"idx": idx, 
                                        "text": response_dict['message']['content'].strip(), 
                                        "gpt4_raw_response": raw_response.dict() if type(response) != dict else raw_response}
                        else:
                            raise Exception(f"{args.model_path} does not exist among the model choices!")

                        if (idx + 1) % 100 == 0:
                            print(f"======== SAMPLE_IDX {idx+1} ========")
                            print("======== OUT_DICT ========\n\n", out_dict)
                            print("="* 10)
                            print("======== TEXT ========\n\n", out_dict['text'])
                            print("="* 10)

                        out_preds.append(out_dict)
                        writer_obj.write(out_dict)  # Save the outputs

                        if args.debug:
                            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


            if (len(out_preds) == len(data_iter)) and task_types[task_type_id] == "attr_gen":  # Generating attributes for data
                # TODO: Parsing of the GPT-4-generated attributes should happen automatically!
                if not args.use_wiki:    # Model-generated attributes
                    if dataset_name == "inaturalist":
                        if args.modality == "text":
                            binomial_out_path = f"../preds/{dataset_name}_attr_gen_{args.modality}_binomial_{args.model_path}_output.jsonl"
                            common_out_path = f"../preds/{dataset_name}_attr_gen_{args.modality}_common_{args.model_path}_output.jsonl"
                            with jsonlines.open(binomial_out_path, "r") as bin_reader:
                                binomial_out_dicts = [l for l in bin_reader]
                            with jsonlines.open(common_out_path, "r") as com_reader:
                                common_out_dicts = [l for l in com_reader]
                        
                else:  # Wikipedia-extracted attributes
                    args.modality = f"wiki-{args.modality}"
                
                # Attributes-generated are parsed into dictionary and saved in 'parsed_out_path'
                parsed_out_path = os.path.join(args.preds_dir, f"parsed_{dataset_name}_outputs", f"parsed-{dataset_name}-{args.model_path}-{args.modality}-combined.json")
                parsed_outputs = []

                if dataset_name == "inaturalist":
                    data_iter = dataset['categories']

                print("===== data (samples) : ", data_iter[0])
                print("===== out_predictions (samples) : ", out_preds[0])
                print("===== # of fine-grained classes : ", len(data_iter))

                print(f"[ Parsing generated attributes ... <Dataset: {dataset_name}> ] ")
                for idx, data in enumerate(tqdm(out_preds)):
                    if not args.use_wiki:
                        if dataset_name == "inaturalist":
                            if args.modality == "text":
                                bin_required_attr, bin_likely_attr = extract_attributes(binomial_out_dicts[idx]['text'])
                                com_required_attr, com_likely_attr = extract_attributes(common_out_dicts[idx]['text'])
                            else:
                                bin_required_attr, bin_likely_attr = extract_attributes(out_preds[idx]['text'])
                                com_required_attr, com_likely_attr = [], []
                            parsed_attr = {"id": idx,
                                "name": data['name'],
                                "common_name": data['common_name'], # data['common_name'],
                                "attr_common": {"required": com_required_attr, "likely": com_likely_attr},
                                "attr_binomial": {"required": bin_required_attr, "likely": bin_likely_attr}
                            }
                    else:  # args.use_wiki
                        fine_name = data_iter[idx]['query'].strip()
                        required_attr, likely_attr = extract_attributes(data['text'])
                        parsed_attr = {"id": idx,
                                       "name": fine_name,
                                       "attr_binomial": {"required": required_attr, "likely": likely_attr}}
                        
                    parsed_outputs.append(parsed_attr)

                with open(parsed_out_path, "w") as parse_writer:
                    json.dump(parsed_outputs, parse_writer)
                    

    else:  # Original LLaVA, multi-turn dialogue setting

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
    parser.add_argument("--data_path", type=str, default="val.json")
    parser.add_argument("--coarse_lbl_dir", type=str, default= "/shared/nas/data/m1/jk100/data")
    parser.add_argument("--coarse_lbl_file", type=str, default="coarse-classes-family.json")
    parser.add_argument("--preds_dir", type=str, default="../preds")
    parser.add_argument("--ctgr2img-path", type=str, default="ctgr2img_dict.json")

    parser.add_argument("--task_type_id", type=int, required=True)
    parser.add_argument("--use_prompt", action="store_true", default=False)
    parser.add_argument("--use_gold_coarse", action="store_true", default=False, help="Using GPT-3.5-turbo generated coarse-grained labels")
    parser.add_argument("--prompt_type_id", type=int, default=None)
    
    parser.add_argument("--input_type", type=str, help="When task_type_id==`attr_gen`, choose from [binomial or common]")
    parser.add_argument("--modality", type=str, default="text", help="When task_type_id == 3, args.modality is either `image` or `text`")
    parser.add_argument("--parse_attr", action="store_true", default=False, help="Parse the generated attributes into parsed-llama-70b file format")
    parser.add_argument("--use_wiki", action="store_true", default=False, help="Use Wikipedia document as input for attribute extraction")

    parser.add_argument("--text_streamer", action="store_true", default=False)

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
