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
from finetune_prompts import input_clf_style, output_clf_style


def convert_to_llava_mix_format(train_dataset, dataset_name=None, root_dir=None, out_data_dir="./playground/data"):
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

    if dataset_name == 'inaturalist':
 
        out_path = 'llava_v1_5_mix865k_inat2021.json'

        if os.path.exists(os.path.join(out_data_dir, out_path)):
            print(f"[ {os.path.join(out_data_dir, out_path)} already exists. ]")
            with open(os.path.join(out_data_dir, out_path), "r") as reader:
                inat_200k_dataset = json.load(reader)
            return inat_200k_dataset

        max_k_supercategory = 20000
        max_conv_len = 3
        annotations = train_dataset.annos
        images = train_dataset.images
        classes = train_dataset.classes
        inat_200k_dataset = []
        # 11 supercategories in iNaturalist(2021)
        supercategory_cnt = {'arachnids': 0, 'mammals': 0, 'reptiles': 0, 
                             'animalia': 0, 'mollusks': 0, 'plants': 0, 'amphibians': 0, 
                             'ray-finned fishes': 0, 'birds': 0, 'insects': 0, 'fungi': 0}
        
        for idx, data in enumerate(tqdm(train_dataset, desc="Converting inaturalist to llava_v1_5_mix format")):
            # Sample 20,000 samples per supercategory
            img_idx = annotations[idx]['id']
            img_path = images[idx]['file_name']
            class_idx = annotations[idx]['category_id']
            supercategory = classes[class_idx]['supercategory']
            sc_key = classes[class_idx]['supercategory'].lower().strip()

            if supercategory_cnt[sc_key] < max_k_supercategory:
                supercategory_cnt[sc_key] += 1
            else:
                continue

            # Sample question template from GPT-4-generated 'input_clf_style' and 'output_clf_style'
            concept_names = [classes[class_idx]['name'], classes[class_idx]['common_name']]
            input_styles = [random.choice(input_clf_style).replace("{concept_placeholder}", supercategory) for _ in range(max_conv_len)]
            output_styles = [random.choice(output_clf_style).replace("{concept_placeholder}", random.choice(concept_names)) for _ in range(max_conv_len)]
            
            if idx == 0:
                print("\n\n== input_styles & output_styles ==\n\n")
                for i in range(len(input_styles)):
                    print("INPUT: ", input_styles[i])
                    print("OUTPUT: ", output_styles[i])

            obj_dict = {  # TODO: Try difference mixtures for fine-tuning (effectiveness of FINER dataset)
                # TODO: 1) What is this? (basic / coarse prediction) -> 2) Attribute prediction -> 3) Fine-grained concept prediction
                "id": img_idx,
                "image": os.path.join(root_dir, img_path),
                "conversations":[
                    {
                        "from": "human",
                        "value": f"<image>\n{input_styles[0]}"
                    },
                    {
                        "from": "gpt",
                        "value": output_styles[0]
                    },
                    {
                        "from": "human",
                        "value": f"{input_styles[1]}"
                    },
                    {
                        "from": "gpt",
                        "value": output_styles[1]
                    },
                    {
                        "from": "human",
                        "value": f"{input_styles[2]}"
                    },
                    {
                        "from": "gpt",
                        "value": output_styles[2]
                    }
                ]
            }
            inat_200k_dataset.append(obj_dict)
        
        with open(os.path.join(out_data_dir, "llava_v1_5_mix665k.json"), "r") as reader:
            llava_v1_5_mix665k = json.load(reader)
            print(f"[ llava_v1_5_mix665k (length) : {len(llava_v1_5_mix665k)} ]")
        
        with open(os.path.join(out_data_dir, out_path), "w") as writer:
            json.dump(llava_v1_5_mix665k + inat_200k_dataset, writer)
            print(f"[ llava_v1_5_mix865k_inat2021 (length) : {len(llava_v1_5_mix665k + inat_200k_dataset)} ]")
            print(f"[ llava_v1_5_mix865k_inat2021 saved in {os.path.join(out_data_dir, out_path)} ]")


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
        convert_to_llava_mix_format(train_dataset, dataset_name=dataset_name, root_dir=data_path, out_data_dir="../playground/data")

    else:
        raise ValueError("--dataset-name has to be one of: ['inaturalist', 'cub-200', 'fgvc-aircraft', 'stanford-cars', 'stanford-dogs']")

