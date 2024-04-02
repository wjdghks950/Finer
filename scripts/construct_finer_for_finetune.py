import os
import pandas as pd
import numpy as np
import argparse
import json
import jsonlines
import time
import random

from dotenv import load_dotenv
from tqdm import tqdm
from data_loader import *
from finetune_prompts import *

from llava.serve.wiki_linearize_attr import linearize_attr


def convert_to_llava_mix_format(mixture_type, held_out_dataset_name=None, root_dir=None, out_data_dir="./playground/data"):
   # Unify their format into the same format as iNaturalist
    '''
    Convert each 'dataset' to the same format as 'llava_v1_5_mix665k.json' format

    'held_out_dataset_name' - The one held-out dataset out of six available datasets

    The following script does the following
    (i) The 'held_out_dataset_name' refers to the held-out dataset name
    (ii) There are 5 held-in datasets and 1 held-out dataset (for zero-shot transferability evaluation)
    (iii) Follows the AttrSeek pipeline

    '''
    root_dir = root_dir if root_dir is not None else ""
    out_data_dir = out_data_dir if out_data_dir is not None else ""

    if not os.path.isdir(out_data_dir):
        os.mkdir(out_data_dir)
        print(f"[ {out_data_dir} does not exist! ]")
        return 0
    
    variant_type = 'fine_answer'  # The last response of the model constitutes the fine-grained label

    # max_k refers to the number of turns
    max_k_per_category = 1  # TODO: Increase the 'max_k_per_category' to see the fine-tuning performance {1, 3, 5, 10}
    max_per_cls = args.max_per_cls

    if mixture_type == 'attr_gen':
        out_path = f'llava_v1_5_mix865k_{mixture_type}_{variant_type}_{held_out_dataset_name}.json'
    else:
        out_path = f'llava_v1_5_mix865k_{mixture_type}_{held_out_dataset_name}_k-{max_k_per_category}.json'

    if os.path.exists(os.path.join(out_data_dir, out_path)):
        print(f"[ {os.path.join(out_data_dir, out_path)} already exists. ]")
        with open(os.path.join(out_data_dir, out_path), "r") as reader:
            mixture_dataset = json.load(reader)
        return mixture_dataset
    
    data_name_dicts = {
        "inaturalist": "inaturalist",
        "cub-200-2011": "CUB_200_2011",
        "fgvc-aircraft": "fgvc-aircraft-2013b",
        "stanford-dogs": "stanford_dogs",
        "stanford-cars": "stanford_cars",
        "nabirds": "nabrids"
    }
    data_name_dicts.pop(held_out_dataset_name)  # Remove the data from training dataset mixture

    # Load the pruned, combined Finer dataset (e.g., parsed-CUB_200_2011-gpt-4-vision-preview-wiki-text-pruned-lbl_combined.json)
    # Use these attributes to construct additional dataset
    # Load the corresponding image paths from (e.g., unified-cub-200-2011-train-combined.jsonl)
    mixture_dataset = []
    for data_key, data_dir in tqdm(data_name_dicts.items(), desc=f"Constructing FINER-mix (Held-out: {held_out_dataset_name})"):
        
        lbl_attrs_path = os.path.join(root_dir, data_dir, f"parsed-{data_dir}-gpt-4-vision-preview-wiki-text-pruned-lbl_combined.json")
        lbl_imgs_path = os.path.join(root_dir, data_dir, f"unified-{data_key}-train-combined.jsonl")

        # Load the attributes from Wikipedia extracted attributes by GPT-4 
        with open(lbl_attrs_path, "r") as reader:
            print(f"[ Loading Wikipedia-extracted attributes from {lbl_attrs_path} ... ]")
            lbls2attrs = json.load(reader)

        # Construct a Dict for fine-grained concept label to class index
        fine2idx = {}
        for class_idx, attr_dict in enumerate(lbls2attrs):
            if len(attr_dict['fine']) > 1:
                if data_dir == 'inaturalist':
                    binom_common_key = tuple(attr_dict['fine'])
                    fine2imgs[binom_common_key] = class_idx
            elif len(attr_dict['fine']) == 1:
                fine2idx[attr_dict['fine'][0]] = class_idx
            else:
                raise ValueError("Fine-grained Concept Label does not exist for this instance")
        
        # print("fine2idx >>\n", fine2idx)

        # Load the image paths from 'data_dir' - the data instances for training
        with jsonlines.open(lbl_imgs_path, "r") as reader:
            print(f"[ Loading image paths from {lbl_imgs_path} ... ]")
            lbls2imgs = [line for line in reader]
        
        # Construct a Dict for fine-grained concept label to List of image paths
        fine2imgs = defaultdict(list)
        for img_dict in lbls2imgs:
            if len(img_dict['fine-level-lbl']) > 1:
                if data_dir == 'inaturalist':
                    binom_common_key = tuple(img_dict['fine-level-lbl'])
                    fine2imgs[binom_common_key].append(img_dict['img_path'])
            elif len(img_dict['fine-level-lbl']) == 1:
                fine2imgs[img_dict['fine-level-lbl'][0]].append(img_dict['img_path'])
            else:
                raise ValueError("Fine-grained Concept Label does not exist for this instance")

        assert len(fine2idx) == len(fine2imgs)

        # Limit to 2,500 (max_per_cls) instances from each dataset - every class should be included
        # except for inaturalist (divide based on supercategories)
        if data_dir == "inaturalist":
            class_cnt = {'arachnids': 0, 'mammals': 0, 'reptiles': 0, 
                        'animalia': 0, 'mollusks': 0, 'plants': 0, 'amphibians': 0, 
                        'ray-finned fishes': 0, 'birds': 0, 'insects': 0, 'fungi': 0}
        else:
            class_cnt = {cls_lbl: 0 for cls_lbl in list(fine2imgs.keys())}

        # Iterate over the 'lbls2imgs' to construct the 'attr_gen' mixture
        for idx, img_dict in enumerate(tqdm(lbls2imgs, desc=f"Constructing [{data_dir}] subset - {mixture_type} | {variant_type}")):
    
            # Count the classes
            if data_dir == "inaturalist":
                class_lbl = img_dict['basic-level-lbl'].lower().strip()
            else:
                class_lbl = img_dict['fine-level-lbl'][0]
            # If the selected class exceeds pre-defined threshold, pass
            if class_cnt[class_lbl] >= (max_per_cls // len(class_cnt)):
                continue

            superordinate_name = img_dict['basic-level-lbl']
            coarse_names = img_dict['coarse-level-lbl']
            fine_names = img_dict['fine-level-lbl']
            class_idx = fine2idx[fine_names[0]] if data_dir != 'inaturalist' else fine2idx[tuple(fine_names)]
            img_path = img_dict['img_path']

            # print("IMG_DICT: ", img_dict)
            # print("="*30)
            # print(fine2imgs.keys())
            # print(f"DATASET : {data_dir}")
            # print("fine2idx: ", fine2idx)
            # print("class_idx: ", class_idx)
            # print(lbls2attrs[class_idx])

            if mixture_type == "attr_gen":
                '''
                attr_gen

                Three-step concept prediction + attribute generation task
                1) Coarse-grained concept classification
                2) Attribute generation (fine-grained image analysis)
                3) Fine-grained concept classification
                '''

            if variant_type == 'fine_answer':
                input_style = random.choice(input_clf_style).replace("{concept_placeholder}", superordinate_name)
                output_style = random.choice(output_clf_style).replace("{concept_placeholder}", random.choice(coarse_names))
                
                attr_input = random.choice(attr_gen_input_styles).replace("{concept_placeholder}", random.choice(coarse_names))
                attr_output = random.choice(attr_gen_output_styles).replace("{concept_placeholder}", random.choice(coarse_names))
                attr_output = attr_output.replace("{attribute_placeholder}", linearize_attr("", lbls2attrs[class_idx]['required'], include_concept_name=False))
                
                attr_interm = random.choice(attr_gen_interm_styles).replace("{concept_placeholder}", random.choice(coarse_names))
                fine_output_style = random.choice(output_clf_style).replace("{concept_placeholder}", random.choice(fine_names))
                
                obj_dict = {
                    "id": f"{data_dir}-sample_{idx}-class_{class_idx}",
                    "image": img_path,
                    "conversations":[
                        {
                            "from": "human",
                            "value": f"<image>\n{input_style}" if random.random() < 0.5 else f"{input_style}\n<image>" # Change the position of <image> token to avoid position bias
                        },
                        {
                            "from": "gpt",
                            "value": output_style  # coarse-grained / basic-level output
                        },
                        {
                            "from": "human",
                            "value": attr_input
                        },
                        {
                            "from": "gpt",
                            "value": attr_output  # generates linearized set of attributes
                        },
                        {
                            "from": "human",
                            "value": attr_interm
                        },
                        {
                            "from": "gpt",
                            "value": fine_output_style
                        }
                    ]
                }

            elif variant_type == 'no_answer':
                obj_dict = {
                    "id": f"{data_dir}-sample_{idx}-class_{class_idx}",
                    "image": img_path,
                    "conversations":[
                        {
                            "from": "human",
                            "value": f"<image>\n{input_style}" if random.random() < 0.5 else f"{input_style}\n<image>" # Change the position of <image> token to avoid position bias
                        },
                        {
                            "from": "gpt",
                            "value": output_style  # coarse-grained / basic-level output
                        },
                        {
                            "from": "human",
                            "value": attr_input
                        },
                        {
                            "from": "gpt",
                            "value": attr_output  # generates linearized set of attributes
                        }
                    ]
                }

            else:  # Regular mixture - inaturalist - Just prediction
                # Sample question template from GPT-4-generated 'input_clf_style' and 'output_clf_style'

                input_style = random.choice(input_clf_style).replace("{concept_placeholder}", superordinate_name)
                output_style = random.choice(output_clf_style).replace("{concept_placeholder}", random.choice(fine_names))
                obj_dict = {
                    "id": f"{data_dir}-sample_{idx}-class_{class_idx}",
                    "image": img_path,
                    "conversations":[
                        {
                            "from": "human",
                            "value": f"<image>\n{input_style}" if random.random() < 0.5 else f"{input_style}\n<image>" # Change the position of <image> token to avoid bias
                        },
                        {
                            "from": "gpt",
                            "value": output_style
                        }
                    ]
                }
        
        mixture_dataset.append(obj_dict)

    # Load the original LLaVA-mix first
    with open(os.path.join(out_data_dir, "llava_v1_5_mix665k.json"), "r") as reader:
        llava_v1_5_mix665k = json.load(reader)
        print(f"[ llava_v1_5_mix665k (length) : {len(llava_v1_5_mix665k)} ]")
    
    # Merge with FINER-mix
    with open(os.path.join(out_data_dir, out_path), "w") as writer:
        json.dump(llava_v1_5_mix665k + mixture_dataset, writer)
        print(f"[ llava_v1_5_mix865k_{mixture_type}_{dataset_name} (length) : {len(llava_v1_5_mix665k + mixture_dataset)} ]")
        print(f"[ llava_v1_5_mix865k_{mixture_type}_{dataset_name} saved in {os.path.join(out_data_dir, out_path)} ]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="/shared/nas/data/m1/jk100/data/")
    parser.add_argument("--held_out_dataset_name", type=str, required=True, default=None, help="inaturalist, cub-200-2011, fgvc-aircraft, stanford-dogs, stanford-cars, nabirds")
    parser.add_argument("--max_per_cls", type=int, default=2500, help="Maximum number of instances from each class")

    args = parser.parse_args()

    root_dir = args.root_dir
    held_out_dataset_name = args.held_out_dataset_name

    print(f"[ Held-Out Dataset: {args.held_out_dataset_name} ]")

    if args.held_out_dataset_name in ['inaturalist', 'cub-200-2011', 'fgvc-aircraft', 'stanford-cars', 'stanford-dogs', 'nabirds']:
        # Unify the loaded dataset formats into 
        convert_to_llava_mix_format(mixture_type='attrgen', 
                                    held_out_dataset_name=held_out_dataset_name, 
                                    root_dir=root_dir, 
                                    out_data_dir="../playground/data")
    else:
        raise ValueError("--dataset-name has to be one of: ['inaturalist', 'cub-200-2011', 'fgvc-aircraft', 'stanford-cars', 'stanford-dogs', 'nabirds]")
