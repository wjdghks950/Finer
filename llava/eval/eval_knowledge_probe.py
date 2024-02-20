import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init, encode_image, extract_attributes, openai_gpt_call, preprocess_gpt_output
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, load_image
from llava.serve.prompts import PROMPT_DICT
from llava.eval.eval_fgvc import eval, remove_ans_prefix, annotation_loader

from PIL import Image
from tqdm import tqdm

import jsonlines
import os
import json
import random
import numpy as np
from dotenv import load_dotenv

from transformers import TextStreamer

# OpenAI query module
from openai import OpenAI


def main(args):
    # Model
    disable_torch_init()

    if args.data_path == "val.json":
        data_split = args.data_path.split(".")[0]
    elif "_" in args.data_path:  # e.g., "val_noplant_64.json"
        data_split = args.data_path.split("_")[0]
    elif "unified-" in args.data_path: # e.g., "unified-fgvc-aircraft-test-combined.jsonl"
        data_split = args.data_path.split("-")[-2]
    else:
        raise Exception(f"args.data_path format is incorrect : [{args.data_path}]")
    
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

    data_dir = os.path.join(args.data_dir, args.dataset_name)
    dataset_name = args.dataset_name

    # iNaturalist dataset inference
    if os.path.isdir(data_dir) is not None:
        if dataset_name == "inaturalist":
            dataset_txt = "inaturalist"
        elif dataset_name == "fgvc-aircraft-2013b":
            dataset_txt = "fgvc_aircraft"
        elif dataset_name == "CUB_200_2011":
            dataset_txt = "cub_200_2011"
        elif dataset_name == "stanford_dogs":
            dataset_txt = "stanford_dogs"
        elif dataset_name == "nabirds":
            dataset_txt = "nabirds"
        # elif dataset_name == "stanford_cars":  # TODO: Construct 'unified-...' first
        #     dataset_txt = "stanford_cars"
        #     annot_path = "unified-stanford-cars-test-combined.jsonl"

    task_type = f"knowledge_probe_{dataset_txt}"
    out_path = f"knowledge_probe_{dataset_txt}_{model_name}_output.jsonl"
    out_dir = f"../preds/{dataset_txt}_outputs"

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # If `out_path` exists, load it up and count the number of idx it should start from!
    out_path = os.path.join(out_dir, out_path)
    print(f"[ OUT_PATH: {out_path} ")

    stopped_idx = 0
    out_preds = []
    if os.path.exists(out_path):
        print(f"[ OUT_PATH already exists in [ {out_path} ]. Reading the file ... ]")
        with open(out_path, "r") as reader_obj:
            lines = reader_obj.readlines()
            out_preds = [json.loads(l) for l in lines]
            stopped_idx = len(out_preds)

    ''' Load the dataset'''

    # Load the original iNaturalist dataset
    path = os.path.join(data_dir, args.data_path)
    print(f"[ DATASET: Loading dataset from [ {path} ] ]")

    if dataset_txt == 'inaturalist':
        coarse_family_path = os.path.join(data_dir, "coarse-classes-family.json")
        # Load 'inaturalist' dataset
        with open(path, "r") as fp:
            dataset = json.load(fp)  # dataset.keys() - dict_keys(['info', 'images', 'categories', 'annotations', 'licenses'])
            data_iter = dataset['annotations']
        # Load 'family to coarse-lbls' json file
        with open(coarse_family_path, "r") as fp:  
            coarse_family_dict = json.load(fp)
        # Load the attributes per concept
        attr_path = os.path.join(args.preds_dir, "parsed_inaturalist_outputs/parsed-gpt-4-wiki-text-combined.json")
        with open(attr_path, "r") as fp:
            attr_dict = json.load(fp)
    else:
        with jsonlines.open(path, "r") as fp:
            data_iter = [d for d in fp]

        # Load the parsed, gpt-4 extracted attributes from the dataset
        attr_path = os.path.join(args.preds_dir, f"parsed_{dataset_name}_outputs", f"parsed-{dataset_name}-gpt-4-vision-preview-wiki-text-combined.json")
        with open(attr_path, "r") as fp:
            attr_dict = json.load(fp)
        # Construct fine-grained label to class-idx dict
        fine2cidx = {d['name']: d['id']for d in attr_dict}

    setting_desc_text = f"[ Knowledge Probing - Model: {model_name} ]_[ Task: {task_type} | {dataset_name} ] "

    image = None
    image_tensor = None

    if stopped_idx < len(data_iter):  # Terminate if the generated file already exists
        with jsonlines.open(out_path, mode='a') as writer_obj:
            for idx, data in enumerate(tqdm(data_iter, desc=setting_desc_text.strip())):
                if stopped_idx > 0 and idx < stopped_idx:
                    continue

                placeholder_str = '{concept_placeholder}'
                supercategory_placeholder_str = '{supercategory_placeholder}'
                attr_placeholder_str = '{attribute_placeholder}'

                user_prompt = PROMPT_DICT[task_type]
                if dataset_txt == 'inaturalist':
                    kingdom_placeholder_str = '{kingdom_placeholder}'
                    phylum_placeholder_str = '{phylum_placeholder}'
                    class_placeholder_str = '{class_placeholder}'
                    order_placeholder_str = '{order_placeholder}'
                    family_placeholder_str = '{family_placeholder}'
                    genus_placeholder_str = '{genus_placeholder}'
                    concept_dict = dataset['categories'][data['category_id']]
                    
                    family_name = concept_dict['family']
                    gold_attributes = "; ".join(attr_dict[data['category_id']]['attr_binomial']['required'])
                    user_prompt = user_prompt.replace(supercategory_placeholder_str, concept_dict['supercategory'].strip())
                    user_prompt = user_prompt.replace(kingdom_placeholder_str, concept_dict['kingdom'])
                    user_prompt = user_prompt.replace(phylum_placeholder_str, concept_dict['phylum'])
                    user_prompt = user_prompt.replace(class_placeholder_str, concept_dict['class'])
                    user_prompt = user_prompt.replace(order_placeholder_str, concept_dict['order'])
                    user_prompt = user_prompt.replace(family_placeholder_str, concept_dict['family'])
                    user_prompt = user_prompt.replace(genus_placeholder_str, concept_dict['genus'])
                    user_prompt = user_prompt.replace(attr_placeholder_str, gold_attributes)

                elif dataset_txt == "fgvc_aircraft":
                    coarse_placeholder_str = '{coarse_placeholder}'
                    user_prompt = user_prompt.replace(supercategory_placeholder_str, data['basic-level-lbl'])
                    user_prompt = user_prompt.replace(coarse_placeholder_str, random.choice(data['coarse-level-lbl']))
                    attributes = []
                    for fine_lbl in data['fine-level-lbl']:
                        cidx = fine2cidx[fine_lbl]
                        attributes.extend(attr_dict[cidx]['attr_binomial']['required'])
                    filtered_attributes = [a for a in attributes if a != "none"]
                    user_prompt = user_prompt.replace(attr_placeholder_str, "; ".join(filtered_attributes))

                elif dataset_txt == "stanford_dogs":
                    coarse_placeholder_str = '{coarse_placeholder}'
                    user_prompt = user_prompt.replace(supercategory_placeholder_str, data['basic-level-lbl'])
                    user_prompt = user_prompt.replace(coarse_placeholder_str, random.choice(data['coarse-level-lbl']))
                    attributes = []
                    for fine_lbl in data['fine-level-lbl']:
                        cidx = fine2cidx[fine_lbl]
                        attributes.extend(attr_dict[cidx]['attr_binomial']['required'])
                    filtered_attributes = [a for a in attributes if a != "none"]
                    user_prompt = user_prompt.replace(attr_placeholder_str, "; ".join(filtered_attributes))

                inp = user_prompt
 
                if "gpt-4" in args.model_path:
                    pass
                # else:  # TODO: Revive if you want to experiment with "Image + attribute" input for concept classification
                #     if model.config.mm_use_im_start_end:
                #         inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                #     else:
                #         inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)

                if idx == 0:
                    print(f" >> conv.get_prompt() : {conv.get_prompt()}\n")
                    
                if "gpt-4" not in args.model_path:
                    prompt = conv.get_prompt()
                else:
                    system_prompt = conv.system
                    user_prompt = inp

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
                    image_path = image_path if args.modality == "image" else None
                    raw_response, response = openai_gpt_call(client, system_prompt, user_prompt, model_name, image_path, max_tokens, temp)
                    response_dict = response.dict() if type(response) != dict else response
                    out_dict = {"idx": idx, 
                                "text": response_dict['message']['content'].strip(), 
                                "gpt4_raw_response": raw_response.dict() if type(response) != dict else raw_response}

                if (idx + 1) % 100 == 0:
                    print(f"[ ======== SAMPLE_IDX {idx+1} ======== ]")
                    print("======== OUT_DICT ========\n\n", out_dict)
                    print("="* 10)
                    print("======== TEXT ========\n\n", out_dict['text'])
                    print("="* 10)

                out_preds.append(out_dict)
                writer_obj.write(out_dict)  # Save the outputs

                if args.debug:
                    print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    
    else:  # Calculate the generated answers in 'out_preds'

        print(f"[ OUT_PATH: {out_path} already exists ]")
        print(f"({os.path.join(args.preds_dir, out_path)}) => LENGTH: ", len(out_preds))
        out_dir = os.path.join(args.preds_dir, dataset_txt + "_outputs")
        if not os.path.exists(out_dir):
            raise FileNotFoundError(f"{out_dir} does not exist!")

        if dataset_txt == "inaturalist":
            annot_path = "unified-inaturalist-test-combined.jsonl"
            ans_prefix = "Specific Epithet:"
        elif dataset_txt == "fgvc_aircraft":
            annot_path = "unified-fgvc-aircraft-test-combined.jsonl"
            ans_prefix = "Specific Airplane:"
        elif dataset_txt == "stanford_dogs":
            annot_path = "unified-stanford-dogs-test-combined.jsonl"
            ans_prefix = "Specific Dog:"
        elif dataset_txt == "cub_200_2011":
            annot_path = "unified-cub-200-test-combined.jsonl"
        elif dataset_txt == "nabirds":
            annot_path = "unified-nabirds-test-combined.jsonl"
        elif dataset_txt == "stanford_cars":
            annot_path = "unified-stanford-cars-test-combined.jsonl"

        # Loading the ground-truth labels for each dataset from 'annot_path
        ground_truths = annotation_loader(os.path.join(args.data_dir, args.dataset_name, annot_path))

        print(f"\n\n==TASK: {task_type}==")
        lbl_key = "fine-level-lbl"

        total_em, total_f1 = [], []
        for pidx, pdict in enumerate(tqdm(out_preds, desc=f'Calculating EM for Knowledge Probe Outputs ({task_type})')):
            idx = int(pdict["idx"])
            output_text = remove_ans_prefix(pdict["text"], prefix=ans_prefix)  # Model-generated text

            ems, f1s = [], []  # Considers maximum EM and F1 scores for fine-grained cases
            ground_truth_lbls = []
            ground_truth_lbls += ground_truths[idx]["fine-level-lbl"]

            print(f"[ ({idx}) GROUND_TRUTH vs. OUTPUT_TEXT: {ground_truth_lbls} || {output_text}")

            for lbl in ground_truth_lbls:
                temp_f1, temp_em = eval(output_text, lbl)
                f1s.append(temp_f1)
                ems.append(temp_em)
            em = max(ems)
            f1 = max(f1s)
            total_em.append(em)
            total_f1.append(f1)

        print(f"\n[ ====== {task_type} ====== ]")
        print(f"F1 Score : {np.mean(total_f1)}")
        print(f"EM Score : {np.mean(total_em)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="llava-7b", help="Model name")
    parser.add_argument("--image-file", type=str, required=False)

    parser.add_argument("--data_dir", type=str, default="/shared/nas/data/m1/jk100/data")
    parser.add_argument("--dataset_name", type=str, default="inaturalist")
    parser.add_argument("--data_path", type=str, default="val.json")
    parser.add_argument("--coarse_lbl_file", type=str, default="coarse-classes-family.json")
    parser.add_argument("--preds_dir", type=str, default="/shared/nas/data/m1/jk100/code/ecole/LLaVA/llava/preds")
    
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
