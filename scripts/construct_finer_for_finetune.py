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


def convert_to_llava_mix_format(train_dataset, mixture_type, dataset_name=None, root_dir=None, out_data_dir="./playground/data"):
   # Unify their format into the same format as iNaturalist
    '''
    Convert each 'dataset' to the same format as 'llava_v1_5_mix665k.json' format
    '''
    root_dir = root_dir if root_dir is not None else ""
    out_data_dir = out_data_dir if out_data_dir is not None else ""

    if not os.path.isdir(out_data_dir):
        os.mkdir(out_data_dir)
        print(f"[ {out_data_dir} does not exist! ]")
        return 0
    
    variant_type = 'fine_answer'  # TODO: The last response of the model constitutes the fine-grained label
    # variant_type = 'no_answer'  # TODO: The last response of the model constitutes the attributes present in the concept

    max_k_per_category = 1  # TODO: Increase the 'max_k_per_category' to see the fine-tuning performance {1, 3, 5, 10}

    if mixture_type == 'attr_gen':
        out_path = f'llava_v1_5_mix865k_{mixture_type}_{variant_type}_{dataset_name}.json'
    else:
        out_path = f'llava_v1_5_mix865k_{mixture_type}_{dataset_name}_k-{max_k_per_category}.json'

    if dataset_name == 'inaturalist':
 
        # TODO: mixture_type - attr_gen
        # TODO: Try difference mixtures for fine-tuning (effectiveness of FINER dataset)
        # TODO: 1) What is this? (basic / coarse prediction) -> 2) Attribute prediction -> 3) Fine-grained concept prediction

        if os.path.exists(os.path.join(out_data_dir, out_path)):
            print(f"[ {os.path.join(out_data_dir, out_path)} already exists. ]")
            with open(os.path.join(out_data_dir, out_path), "r") as reader:
                inat_dataset = json.load(reader)
            return inat_dataset
        
        if mixture_type == "attr_gen":
            # Load the attributes from Wikipedia extracted attributes by GPT-4 
            attr_path = "/home/jk100/code/ecole/LLaVA/llava/preds/parsed_inaturalist_outputs/parsed-gpt-4-wiki-text-combined.json"
            with open(attr_path, "r") as reader:
                print(f"[ Loading Wikipedia-extracted attributes from {attr_path} ... ]")
                attr_dict = json.load(reader)
            coarse_family_path = "/shared/nas/data/m1/jk100/data/inaturalist/coarse-classes-family.json"
            with open(coarse_family_path, "r") as reader:
                print(f"[ Loading family-to-coarse label mappings from {coarse_family_path} ... ]")
                coarse_family_dict = json.load(reader)
        
        print(f"\n========== [ Dataset: {dataset_name} | mixture_type: {mixture_type} | variant_type: {variant_type} ] ==========\n")

        max_k_supercategory = 20000
        max_conv_len = 3
        annotations = train_dataset.annos
        images = train_dataset.images
        classes = train_dataset.classes
        inat_dataset = []
        # 11 supercategories in iNaturalist(2021)
        supercategory_cnt = {'arachnids': 0, 'mammals': 0, 'reptiles': 0, 
                             'animalia': 0, 'mollusks': 0, 'plants': 0, 'amphibians': 0, 
                             'ray-finned fishes': 0, 'birds': 0, 'insects': 0, 'fungi': 0}

        finecategory = train_dataset.fine_classes
        finecategory_cnt = {fine_class[0]: 0 for fine_class in finecategory}  # Use the binomial name as the key for 'finecategory_cnt'
        
        for idx, data in enumerate(tqdm(train_dataset, desc="Converting inaturalist to llava_v1_5_mix format")):
            # Sample 20,000 samples per supercategory
            img_idx = annotations[idx]['id']
            img_path = images[idx]['file_name']
            class_idx = annotations[idx]['category_id']
            supercategory = classes[class_idx]['supercategory'].strip()
            family = classes[class_idx]['family'].strip()
            sc_key = classes[class_idx]['supercategory'].lower().strip()
            bin_common_names = [classes[class_idx]['name'], classes[class_idx]['common_name']]  # Use both the binomial and common_name as output labels

            # Every fine-grained concept should have exactly 'max_k_per_category' included in the mixture
            if finecategory_cnt[classes[class_idx]['name']] == 0:
                finecategory_cnt[classes[class_idx]['name']] += 1
            else:
                continue

            # print("="*20)
            # print("\n annotation: ", annotations[idx])
            # print("img_path: ", img_path)
            # print("classes : ", classes[class_idx])
            # print("class_attr (Dict): ", attr_dict[class_idx])
            # print("class_attr (attributes): ", attr_dict[class_idx]['attr_binomial']['required'])
            # print("family-mapping (coarse0) : ", coarse_family_dict[family])
            # print("\n\nlinearized class_attr: ", linearize_attr("", attr_dict[class_idx]['attr_binomial']['required'], include_concept_name=False))
            # print("="*20)
            # exit()

            if supercategory_cnt[sc_key] < max_k_supercategory:
                supercategory_cnt[sc_key] += 1
            else:
                continue


            # TODO: Try difference mixtures for fine-tuning (effectiveness of FINER dataset)
            # TODO: 1) What is this? (basic / coarse prediction) -> 2) Attribute prediction -> 3) Fine-grained concept prediction

            if mixture_type == "attr_gen":
                ## "attr_gen"
                '''
                attr_gen

                Three-step concept prediction + attribute generation task
                1) Coarse-grained concept classification
                2) Attribute generation (fine-grained image analysis)
                3) Fine-grained concept classification
                '''
                coarse_concept_names = coarse_family_dict[family]
                fine_concept_names = [classes[class_idx]['name'], classes[class_idx]['common_name']]
                input_style = random.choice(input_clf_style).replace("{concept_placeholder}", supercategory)
                output_style = random.choice(output_clf_style).replace("{concept_placeholder}", random.choice(coarse_concept_names))
                
                attr_input = random.choice(attr_gen_input_styles).replace("{concept_placeholder}", random.choice(coarse_concept_names))
                attr_output = random.choice(attr_gen_output_styles).replace("{concept_placeholder}", random.choice(coarse_concept_names))
                attr_output = attr_output.replace("{attribute_placeholder}", linearize_attr("", attr_dict[class_idx]['attr_binomial']['required'], include_concept_name=False))
                
                attr_interm = random.choice(attr_gen_interm_styles).replace("{concept_placeholder}", random.choice(coarse_concept_names))
                fine_output_style = random.choice(output_clf_style).replace("{concept_placeholder}", random.choice(fine_concept_names))
                
                if variant_type == 'fine_answer':
                    obj_dict = {
                        "id": img_idx,
                        "image": os.path.join(root_dir, img_path),
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
                        "id": img_idx,
                        "image": os.path.join(root_dir, img_path),
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
                fine_concept_names = [classes[class_idx]['name'], classes[class_idx]['common_name']]
                input_style = random.choice(input_clf_style).replace("{concept_placeholder}", supercategory)
                output_style = random.choice(output_clf_style).replace("{concept_placeholder}", random.choice(fine_concept_names))
                obj_dict = {
                    "id": img_idx,
                    "image": os.path.join(root_dir, img_path),
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
            
            inat_dataset.append(obj_dict)

        assert sum([cnt for _, cnt in finecategory_cnt.items()]) == max_k_per_category * len(train_dataset.fine_classes)
        
        with open(os.path.join(out_data_dir, "llava_v1_5_mix665k.json"), "r") as reader:
            llava_v1_5_mix665k = json.load(reader)
            print(f"[ llava_v1_5_mix665k (length) : {len(llava_v1_5_mix665k)} ]")
        
        with open(os.path.join(out_data_dir, out_path), "w") as writer:
            json.dump(llava_v1_5_mix665k + inat_dataset, writer)
            print(f"[ llava_v1_5_mix865k_{mixture_type}_{dataset_name} (length) : {len(llava_v1_5_mix665k + inat_dataset)} ]")
            print(f"[ llava_v1_5_mix865k_{mixture_type}_{dataset_name} saved in {os.path.join(out_data_dir, out_path)} ]")


    elif dataset_name == "cub-200":
        pass

    elif dataset_name == "fgvc-aircraft":
        pass

    elif dataset_name == "stanford-dogs":
        pass

    elif dataset_name == "stanford-cars":  # TODO: 'stanford-cars' is incomplete
        pass

    elif dataset_name == "nabirds":
        pass

    else:
        raise ValueError("Invalid 'dataset_name'.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="/shared/nas/data/m1/jk100/data/")
    parser.add_argument("--dataset-name", type=str, required=True, default=None, help="inaturalist, cub-200, fgvc-aircraft, stanford-dogs, stanford-cars, nabirds")

    args = parser.parse_args()

    root_dir = args.root_dir
    dataset_name = args.dataset_name

    print(f"[ Dataset: {args.dataset_name} ]")

    train_dataset = None
    test_dataset = None

    if args.dataset_name in ['inaturalist', 'cub-200', 'fgvc-aircraft', 'stanford-cars', 'stanford-dogs', 'nabirds']:
        if dataset_name == 'inaturalist':
            data_path = os.path.join(root_dir, "inaturalist")
            # test_dataset = iNaturalist(data_path, split='val', download=False)
            train_dataset = iNaturalist(data_path, split='train', download=False)
        
        elif dataset_name == 'nabirds':
            data_path = os.path.join(root_dir, "nabirds")
            train_dataset = NABirds(data_path, train=True, download=False)
            test_dataset = NABirds(data_path, train=False, download=False)

        elif dataset_name == "cub-200":
            data_path = os.path.join(root_dir, "CUB_200_2011")
            # For 'CUB_200_2011', the coarse-grained lbls and fine-grained lbls are contained in the same class instantiation
            train_dataset = Cub2011(data_path, train=True, download=False)
            test_dataset = Cub2011(data_path, train=False, download=False)

        elif dataset_name == "fgvc-aircraft":
            data_path = os.path.join(root_dir, "fgvc-aircraft-2013b")

            # Fine-grained = 'class_type == variant / family'
            # combine 'variant' and 'family' into same category - 'fine-grained'
            # each sample will have multiple labels - e.g., [[Boeing 707-max, Boeing 707], [...], ...]
            train_dataset = Aircraft(data_path, train=True, class_type=["variant", "family"], download=False)
            test_dataset = Aircraft(data_path, train=False, class_type=["variant", "family"], download=False)

            # Coarse-grained = 'class_type == manufacturer'
            coarse_train_dataset = Aircraft(data_path, train=True, class_type=["manufacturer"], download=False)
            coarse_test_dataset = Aircraft(data_path, train=False, class_type=["manufacturer"], download=False)

        elif dataset_name == "stanford-cars":
            data_path = os.path.join(root_dir, "stanford_cars")
            train_dataset = Cars(data_path, train=True, download=False)
            test_dataset = Cars(data_path, train=False, download=False)

        elif dataset_name == "stanford-dogs":
            data_path = os.path.join(root_dir, "stanford_dogs")
            train_dataset = Dogs(data_path, train=True, download=False)
            test_dataset = Dogs(data_path, train=False, download=False)

        # Unify the loaded dataset formats into 
        convert_to_llava_mix_format(train_dataset, mixture_type='regmix', dataset_name=dataset_name, root_dir=data_path, out_data_dir="../playground/data")

    else:
        raise ValueError("--dataset-name has to be one of: ['inaturalist', 'cub-200', 'fgvc-aircraft', 'stanford-cars', 'stanford-dogs']")
