import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init, encode_image, extract_attributes
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.eval.eval_inaturalist import remove_ans_prefix

from PIL import Image
from tqdm import tqdm

import jsonlines
import requests
import os
import json
import time
from dotenv import load_dotenv  

from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from prompts import PROMPT_DICT

# OpenAI query module
from openai import OpenAI


'''
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])
'''


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def openai_gpt_call(client, system_prompt, user_prompt, model, image_path=None, max_tokens=256, temp=0.0):

    if image_path is not None:
        base64_image = encode_image(image_path)  # base64-encoded image
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            },
        ]
    else:
        user_content = [
            {"type": "text", "text": user_prompt},
        ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    response = client.chat.completions.create(
        model=model, 
        messages=messages,
        max_tokens=max_tokens,
        temperature=temp
    )
    return response, response.choices[0]

def openai_gpt_call(client, system_prompt, user_prompt, 
                    model, image_path=None, max_tokens=256, 
                    temp=0.0, max_retries=5, retry_delay=2):
    retry_count = 0
    while True:
        try:
            if image_path is not None:
                base64_image = encode_image(image_path)  # base64-encoded image
                user_content = [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }}
                ]
            else:
                user_content = [{"type": "text", "text": user_prompt}]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            response = client.chat.completions.create(
                model=model, 
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp
            )

            # If response is successful, break the loop
            if response:
                return response, response.choices[0]

        except Exception as e:
            # Handle specific exceptions if needed
            print(f"An error occurred: {e}")

        retry_count += 1
        if retry_count >= max_retries:
            raise Exception("Maximum number of retries reached. API call failed.")

        time.sleep(retry_delay)


def preprocess_gpt_output(response, start_str="Answer:"):
    answer_start_idx = response.find(start_str) + len(start_str) if response.find(start_str) != -1 else 0
    prepro_response = response[answer_start_idx:]
    return prepro_response 


def main(args):
    # Model
    disable_torch_init()

    data_split = None  # or "train"
    if args.data_path in ["train.json", "val.json"]:
        data_split = args.data_path.split(".")[0]
    elif "_" in args.data_path:
        data_split = args.data_path.split("_")[0]
    else:
        raise Exception(f"args.data_path format is incorrect : [{args.data_path}]")

    if not args.parse_attr:  # No model loading if args.parse_attr == True

        if "gpt-" not in args.model:
            model_name = get_model_name_from_path(args.model_path)
            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
        else:  # GPT-4V
            model_name = args.model_path
            load_dotenv()  # Load environment variables from .env file
            api_key = os.getenv("OPENAI_API_KEY")  # Access the API key using the variable name defined in the .env file
            client = OpenAI(api_key=api_key)

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

    # iNaturalist dataset inference
    if args.data_dir is not None:
        out_preds = []

        task_types = ["high_coarse", "coarse", "fine", "attr_gen"]
        task_type_id = args.task_type_id

        use_prompt = args.use_prompt
        prompt_types = ["cot_0shot", "cot_fewshot", "attr_seek"]
        prompt_type_id = args.prompt_type_id

        task_name = f"{prompt_types[prompt_type_id]}_{task_types[task_type_id]}" if use_prompt else f"{task_types[task_type_id]}"
        out_path = f"../preds/{args.data_dir.split('/')[-1]}_{task_name}_output.jsonl"
        
        if task_types[task_type_id] == "attr_gen":
            if args.modality == "text":
                out_path = f"../preds/{args.data_dir.split('/')[-1]}_attr_gen_{args.modality}_{args.input_type}_{args.model}_output.jsonl"
            else:  # args.modality == "image"
                out_path = f"../preds/{args.data_dir.split('/')[-1]}_attr_gen_{args.modality}_{args.model}_output.jsonl"
            if args.use_wiki:
                out_path = f"../preds/{args.data_dir.split('/')[-1]}_attr_gen_wiki_{args.modality}_{args.model}_output.jsonl"
                cached_wiki_attr_dicts = []
                cached_wiki_docs_path = os.path.join("../preds", "parsed-llava-7b-text-queried-wiki-out.json")
                if os.path.exists(cached_wiki_docs_path):
                    with open(cached_wiki_docs_path, "r") as cached_reader:
                        for d in tqdm(cached_reader, desc=f"Loading cached Wikipedia documents for use [{cached_wiki_docs_path}]"):
                            d = json.loads(d)
                            cached_wiki_attr_dicts.append(d)
        
            # If `out_path` exists, load it up and count the number of idx it should start from!
            stopped_idx = 0
            if os.path.exists(out_path):
                print("Out_path already exists in [ {out_path} ]. Reading the file ...")
                with open(out_path, "r") as reader_obj:
                    lines = reader_obj.readlines()
                    out_preds = [json.loads(l) for l in lines]
                    stopped_idx = len(out_preds)

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
        print(f"Loading dataset from [ {path} ]")
        with open(os.path.join(args.data_dir, args.data_path), "r") as fp:
            dataset = json.load(fp)  # dataset.keys() - dict_keys(['info', 'images', 'categories', 'annotations', 'licenses'])
            fp.close()

        data_iter = dataset['annotations']
        
        if task_type_id == 3:
            print("(demo) dataset.keys() : ", dataset.keys())
            print("categories (len) : ", len(dataset['categories']))
            print("categories (sample) : ", dataset['categories'][0])
            data_iter = dataset['categories']
            # Load the dataset['categories'] to actual image files mapped dict
            ctgr2img_path = os.path.join(args.data_dir, args.ctgr2img_path)
            if os.path.exists(ctgr2img_path):
                with open(ctgr2img_path, "r") as reader_obj:
                    ctgr2img_dict = json.load(reader_obj)
        
        print("dataset (length) : ", len(data_iter))

        setting_desc_text = f"[ Model: {args.model_path} ]_[ Task: {task_types[task_type_id]} | {args.modality} | {args.input_type} ] "

        if stopped_idx < len(data_iter):  # Terminate if the generated file already exists
            
            setting_desc_text = f"[ Model: {args.model_path} ]_[ Task: {task_types[task_type_id]} | {args.modality} | {args.input_type} ] "

            with jsonlines.open(out_path, mode='a') as writer_obj:

                for idx, data in enumerate(tqdm(data_iter, desc=setting_desc_text.strip())):
                    if stopped_idx > 0 and idx < stopped_idx:
                        continue

                    if task_type_id < 3:
                        img_dict = dataset['images'][idx]

                        # For the "coarse" and "fine" cases, 
                        # inject the previous output into the current prompt (or gold if --use_gold_coarse)
                        user_prompt = PROMPT_DICT[task_name]
                        user_prompt = user_prompt.replace(placeholder_str, remove_ans_prefix(coarse_out_dict[str(idx)], prefix="Answer:"))
                        
                        if prompt_types[prompt_type_id] == "attr_seek":
                            aseek_init_prompt = PROMPT_DICT[prompt_types[prompt_type_id]]
                            aseek_init_prompt = aseek_init_prompt.replace(placeholder_str, remove_ans_prefix(coarse_out_dict[str(idx)], prefix="Answer:"))

                        image_path = os.path.join(args.data_dir, img_dict['file_name'])
                        image = load_image(image_path)
                        image_tensor = process_images([image], image_processor, args)  # Similar operation in model_worker.py

                        # print(f"====== idx: {idx} | image_tensor (shape) : {image_tensor.shape} ======")
                        if type(image_tensor) is list:
                            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(model.device, dtype=torch.float16)


                    elif task_type_id == 3 and args.modality == "image":
                        
                        user_prompt = PROMPT_DICT[task_name + "_" + args.modality]  # "attr_gen_image"
                        user_prompt = user_prompt.replace(placeholder_str, data['supercategory'])

                        image_path = os.path.join(args.data_dir, data_split, data['image_dir_name'], ctgr2img_dict[data['image_dir_name']][0])
                        image = load_image(image_path)
                        image_tensor = process_images([image], image_processor, args)  # Similar operation in model_worker.py

                        # print(f"====== idx: {idx} | image_tensor (shape) : {image_tensor.shape} ======")
                        if type(image_tensor) is list:
                            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(model.device, dtype=torch.float16)


                    elif task_type_id == 3 and args.modality == "text":
                        # For the "attr_gen" & "modality == text" case, 
                        # simply replace the user_prompt with binomial nomenclature (or common name)
                        if args.use_wiki:
                            user_prompt = PROMPT_DICT[task_name + "_wiki"]
                        else:
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

                    # if (idx + 1) % 1 == 0:
                    #     print(f" >> conv.get_prompt() : {conv.get_prompt()}\n")
                        
                    if "gpt-4" not in args.model:
                        prompt = conv.get_prompt()
                    else:
                        system_prompt = conv.system
                        user_prompt = inp
                        if args.use_wiki:
                            user_prompt += "Document: " + cached_wiki_attr_dicts[idx]['binomial_wiki_doc']

                    # In `single` turn, empty conv after one turn
                    if args.dialogue_mode == 'single':
                        conv.clear_message()
                    
                    if "llava" in args.model:
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

                    elif "gpt-4" in args.model:
                        max_tokens = args.max_new_tokens
                        temp = args.temperature
                        image_path = image_path if args.modality == "image" else None
                        raw_response, response = openai_gpt_call(client, system_prompt, user_prompt, model_name, image_path, max_tokens, temp)
                        response_dict = response.dict()
                        out_dict = {"idx": idx, "text": response_dict['message']['content'].strip(), "gpt4_raw_response": raw_response.dict()}

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
            
        else:  # If the file generated file already exists
            # Use `extract_attributes(text)` to post-process the generated attributes
            # And save them in the same format Ansel sent me
            print(f"[ {out_path} ] already exists! ")
            if task_types[task_type_id] == "attr_gen":

                if args.modality == "text":
                    binomial_out_path = f"../preds/{args.data_dir.split('/')[-1]}_attr_gen_{args.modality}_binomial_{args.model}_output.jsonl"
                    common_out_path = f"../preds/{args.data_dir.split('/')[-1]}_attr_gen_{args.modality}_common_{args.model}_output.jsonl"
                    
                    with open(binomial_out_path, "r") as bin_reader:
                        lines = bin_reader.readlines()
                        binomial_out_dicts = [json.loads(l) for l in lines]
                        bin_reader.close()

                    with open(common_out_path, "r") as com_reader:
                        lines = com_reader.readlines()
                        common_out_dicts = [json.loads(l) for l in lines]
                        com_reader.close()

                    if len(binomial_out_dicts) != len(common_out_dicts):
                        min_idx = min(len(binomial_out_dicts), len(common_out_dicts))

                else:  # args.modality == "image"
                    img_out_path = f"../preds/{args.data_dir.split('/')[-1]}_attr_gen_{args.modality}_{args.model}_output.jsonl"
                    with open(img_out_path, "r") as img_reader:
                        lines = img_reader.readlines()
                        img_out_dicts = [json.loads(l) for l in lines]
                        img_reader.close()
                    
                parsed_out_path = f"parsed-{args.model}-{args.modality}-combined.json"
                parsed_outputs = []

                print("Parsing generate attributes ...")
                for idx, data in enumerate(tqdm(dataset['categories'])):
                    if args.modality == "text" and len(binomial_out_dicts) != len(common_out_dicts):  # FIXME: Hacky fix - GPT-4 may have stopped in the middle
                        if idx == min_idx:
                            break
                    if args.modality == "text":
                        bin_required_attr, bin_likely_attr = extract_attributes(binomial_out_dicts[idx]['text'])
                        com_required_attr, com_likely_attr = extract_attributes(common_out_dicts[idx]['text'])
                    else:
                        bin_required_attr, bin_likely_attr = extract_attributes(img_out_dicts[idx]['text'])
                        com_required_attr, com_likely_attr = [], []
                    # print(" == binomial original == \n", binomial_out_dicts[idx]['text'])
                    # print("\n\n == common original == \n", common_out_dicts[idx]['text'])
                    # print("== binomial - required_attr: ", bin_required_attr)
                    # print("\n== binomial - likley_attr: ", bin_likely_attr)
                    # print("=" * 20)
                    # print("== common - required_attr: ", com_required_attr)
                    # print("\n== common - likley_attr: ", com_likely_attr)
                    # exit()
                    parsed_attr = {"id": idx,
                                "name": data['name'],
                                "common_name": data['common_name'],
                                "attr_common": {"required": com_required_attr, "likely": com_likely_attr},
                                "attr_binomial": {"required": bin_required_attr, "likely": bin_likely_attr}
                                }
                    parsed_outputs.append(parsed_attr)

                with open(os.path.join("../preds", parsed_out_path), "w") as parse_writer:
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
    parser.add_argument("--data_path", type=str, default="val_noplant_64.json")
    parser.add_argument("--coarse_lbl_dir", type=str, default= "/home/jk100/code/ecole/gpt_output")
    parser.add_argument("--coarse_lbl_file", type=str, default="coarse_grained_lbls_gpt4_val_noplant64.json")
    parser.add_argument("--preds_dir", type=str, default="")
    parser.add_argument("--ctgr2img-path", type=str, default="ctgr2img_dict.json")

    parser.add_argument("--task_type_id", type=int, required=True)
    parser.add_argument("--use_prompt", action="store_true", default=False)
    parser.add_argument("--use_gold_coarse", action="store_true", default=False, help="Using GPT-3.5-turbo generated coarse-grained labels")
    parser.add_argument("--prompt_type_id", type=int, default=0)
    
    parser.add_argument("--input_type", type=str, help="When task_type_id==`attr_gen`, choose from [binomial or common]")
    parser.add_argument("--modality", type=str, default="text", help="When task_type_id == 3, args.modality is either `image` or `text`")
    parser.add_argument("--model", type=str, default="llava-7b", help="Model to use for attribute generation when task_type_id==`attr_gen`")
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
