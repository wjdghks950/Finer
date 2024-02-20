import os
import pandas as pd
import scipy
import numpy as np
import argparse
import json
import warnings
import jsonlines
import time
import re

from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, extract_archive, verify_str_arg, check_integrity, download_file_from_google_drive, list_dir
from torch.utils.data import Dataset

from llava.utils import openai_gpt_call

# OpenAI query module
from openai import OpenAI


'''
For the original code, credits to: "https://github.com/lvyilin/pytorch-fgvc-dataset"

Run 'data_loader.py' to unify all the datasets in the following list of datasets
'''


def unify_data_format(fine_train_dataset, fine_test_dataset, 
                      coarse_train_dataset, coarse_test_dataset,
                      dataset_name=None, root_dir=None):
    # CUB-200-2011, Stanford-Cars, -Dogs and FGVC-Aircraft, NABirds, iNaturalist (2021)
    # Unify their format into the same format as iNaturalist
    '''
    Data output format:

    {'idx': i,
     'basic-level-lbl': ...,
     'coarse-level-lbl': ...,
     'fine-level-lbl': ...,
     'img_path': ...,
     'metadata': {...},}

    'metadata' may be different per dataset
    '''
    root_dir = root_dir if root_dir is not None else ""
    test_out_path = os.path.join(root_dir, f"unified-{dataset_name}-test-combined.jsonl")
    train_out_path = os.path.join(root_dir, f"unified-{dataset_name}-train-combined.jsonl")

    if dataset_name == 'inaturalist':

        fine_classes = fine_test_dataset.fine_classes
        coarse_classes = fine_test_dataset.coarse_classes_family # iNaturalist dataset combines the fine and coarse classes into one dataset instance
        classes = fine_test_dataset.classes  # 'categories' of inaturalist dataset

        print("[ test-dataset ] \n", fine_test_dataset)
        print("[ train-dataset ] \n", fine_train_dataset)

        test_unified_dicts = []
        if not os.path.exists(test_out_path):
            with jsonlines.open(test_out_path, "a") as writer:
                for idx, (img_path, tgt) in enumerate(tqdm(fine_test_dataset, desc='[ Unifying iNaturalist_2021 (test) ]')):
                    supercategory = classes[tgt]['supercategory'].strip()
                    family = classes[tgt]['family'].strip()
                    coarse_lbls = coarse_classes[family]

                    obj_dict = {'idx': idx, 
                                'basic-level-lbl': supercategory, 
                                'coarse-level-lbl': coarse_lbls,
                                'fine-level-lbl': fine_classes[tgt],
                                'img_path': img_path,
                                'metadata': {}
                                }
                    
                    writer.write(obj_dict)
                    test_unified_dicts.append(obj_dict)
        else:
            with open(test_out_path, "r") as reader:
                for line in reader:
                    test_unified_dicts.append(json.loads(line))
    
        train_unified_dicts = []
        if not os.path.exists(train_out_path):
            with jsonlines.open(train_out_path, "a") as writer:
                for idx, (img_path, tgt) in enumerate(tqdm(fine_train_dataset, desc='[ Unifying iNaturalist_2021 (train) ]')):
                    supercategory = classes[tgt]['supercategory'].strip()
                    family = classes[tgt]['family'].strip()
                    coarse_lbls = coarse_classes[family]

                    obj_dict = {'idx': idx, 
                                'basic-level-lbl': supercategory, 
                                'coarse-level-lbl': coarse_lbls,
                                'fine-level-lbl': fine_classes[tgt],
                                'img_path': img_path,
                                'metadata': {}
                                }
                    writer.write(obj_dict)
                    train_unified_dicts.append(obj_dict)
        else:
            with open(train_out_path, "r") as reader:
                for line in reader:
                    train_unified_dicts.append(json.loads(line))

    elif dataset_name == "cub-200":

        fine_classes = fine_test_dataset.fine_classes
        coarse_classes = fine_test_dataset.coarse_classes # CUB_200_2011 dataset combines the fine and coarse classes into one dataset instance

        print("[ test-dataset ] \n", fine_test_dataset)
        print("[ train-dataset ] \n", fine_train_dataset)

        test_unified_dicts = []
        if not os.path.exists(test_out_path):
            with jsonlines.open(test_out_path, "a") as writer:
                for idx, (img_path, tgt) in enumerate(tqdm(fine_test_dataset, desc='[ Unifying CUB_200_2011 (test) ]')):
                    obj_dict = {'idx': idx, 
                                'basic-level-lbl': "Bird", 
                                'coarse-level-lbl': [coarse_classes[tgt]],
                                'fine-level-lbl': [fine_classes[tgt]],
                                'img_path': img_path,
                                'metadata': {}
                                }
                    writer.write(obj_dict)
                    test_unified_dicts.append(obj_dict)
        else:
            with open(test_out_path, "r") as reader:
                for line in reader:
                    test_unified_dicts.append(json.loads(line))

    
        train_unified_dicts = []
        if not os.path.exists(train_out_path):
            with jsonlines.open(train_out_path, "a") as writer:
                for idx, (img_path, tgt) in enumerate(tqdm(fine_train_dataset, desc='[ Unifying CUB_200_2011 (train) ]')):
                    obj_dict = {'idx': idx, 
                                'basic-level-lbl': "Bird", 
                                'coarse-level-lbl': [coarse_classes[tgt]],
                                'fine-level-lbl': [fine_classes[tgt]],
                                'img_path': img_path,
                                'metadata': {}
                                }
                    writer.write(obj_dict)
                    train_unified_dicts.append(obj_dict)
        else:
            with open(train_out_path, "r") as reader:
                for line in reader:
                    train_unified_dicts.append(json.loads(line))

    elif dataset_name == "fgvc-aircraft":

        fine_classes = list(fine_test_dataset.classes)  # variant & family
        coarse_classes = list(coarse_test_dataset.classes)  # manufacturer

        # variant, family and manufacturer are NOT aligned
        # First, set up a dict with image_id as keys and coarse_lbls - 'img2class_dict'
        train_img2class_dict, test_img2class_dict = {}, {}
        train_img2class_dict_path = "train-fgvc-aircraft-img2class-dict.json"
        test_img2class_dict_path = "test-fgvc-aircraft-img2class-dict.json"

        img_root_dir = "/shared/nas/data/m1/jk100/data/fgvc-aircraft-2013b/data/images"
        ext = ".jpg"
        
        if not os.path.exists(os.path.join(root_dir, test_img2class_dict_path)):
            print("[ [split: Test] Integrating variant, family, manufacture types in fgvc-aircraft ... ]")
            for i, data in enumerate(tqdm(fine_test_dataset)):
                sample, tgt_idx = data
                img_path, extension = os.path.splitext(sample)
                img_id = img_path.split("/")[-1]
                if img_id not in test_img2class_dict:
                    test_img2class_dict[img_id] = {"fine": [fine_classes[tgt_idx]]}
                else:
                    test_img2class_dict[img_id]["fine"].append(fine_classes[tgt_idx])

            for i, data in enumerate(tqdm(coarse_test_dataset)):
                sample, tgt_idx = data
                img_path, extension = os.path.splitext(sample)
                img_id = img_path.split("/")[-1]
                if img_id not in test_img2class_dict:
                    test_img2class_dict[img_id]["coarse"] = [coarse_classes[tgt_idx]]
                else:
                    if "coarse" not in test_img2class_dict[img_id]:
                        test_img2class_dict[img_id]["coarse"] = [coarse_classes[tgt_idx]]
                    else:
                        test_img2class_dict[img_id]["coarse"].append(coarse_classes[tgt_idx])

            for img_id, _ in test_img2class_dict.items():
                test_img2class_dict[img_id]['fine'] = list(set(test_img2class_dict[img_id]['fine']))
                test_img2class_dict[img_id]['coarse'] = list(set(test_img2class_dict[img_id]['coarse']))
                        
            with open(os.path.join(root_dir, test_img2class_dict_path), "w") as writer:
                json.dump(test_img2class_dict, writer)
                print(f"[ test_img2class_dict saved to {os.path.join(root_dir, test_img2class_dict_path)} ]")

        else:
            print(f"[ test_img2class_dict loaded from {os.path.join(root_dir, test_img2class_dict_path)} ]")
            with open(os.path.join(root_dir, test_img2class_dict_path), "r") as reader:
                test_img2class_dict = json.load(reader)


        if not os.path.exists(os.path.join(root_dir, train_img2class_dict_path)):
            print("[ [split: Train] Integrating variant, family, manufacture types in fgvc-aircraft ... ]")
            for i, data in enumerate(tqdm(fine_train_dataset)):
                sample, tgt_idx = data
                img_path, extension = os.path.splitext(sample)
                img_id = img_path.split("/")[-1]
                if img_id not in train_img2class_dict:
                    train_img2class_dict[img_id] = {"fine": [fine_classes[tgt_idx]]}
                else:
                    train_img2class_dict[img_id]["fine"].append(fine_classes[tgt_idx])

            for i, data in enumerate(tqdm(coarse_train_dataset)):
                sample, tgt_idx = data
                img_path, extension = os.path.splitext(sample)
                img_id = img_path.split("/")[-1]
                if img_id not in train_img2class_dict:
                    train_img2class_dict[img_id]["coarse"] = [coarse_classes[tgt_idx]]
                else:
                    if "coarse" not in train_img2class_dict[img_id]:
                        train_img2class_dict[img_id]["coarse"] = [coarse_classes[tgt_idx]]
                    else:
                        train_img2class_dict[img_id]["coarse"].append(coarse_classes[tgt_idx])

            for img_id, _ in train_img2class_dict.items():
                train_img2class_dict[img_id]['fine'] = list(set(train_img2class_dict[img_id]['fine']))
                train_img2class_dict[img_id]['coarse'] = list(set(train_img2class_dict[img_id]['coarse']))
                        
            with open(os.path.join(root_dir, train_img2class_dict_path), "w") as writer:
                json.dump(train_img2class_dict, writer)
                print(f"[ train_img2class_dict saved to {os.path.join(root_dir, train_img2class_dict_path)} ]")

        else:
            print(f"[ train_img2class_dict loaded from {os.path.join(root_dir, train_img2class_dict_path)} ]")
            with open(os.path.join(root_dir, train_img2class_dict_path), "r") as reader:
                train_img2class_dict = json.load(reader)
        
        print("\n======= test_img2class_dict =======\n")
        print(test_img2class_dict)
        print("test_img2class_dict (length) >> ", len(test_img2class_dict))

        print("\n======= train_img2class_dict =======\n")
        print(train_img2class_dict)
        print("train_img2class_dict (length) >> ", len(train_img2class_dict))

        test_unified_dicts = []
        if not os.path.exists(test_out_path):
            with jsonlines.open(test_out_path, "a") as writer:
                for idx, (img_id, gdict) in enumerate(tqdm(test_img2class_dict.items(), desc="[ Unifying fgvc-aircraft-2013b test ]")):
                    obj_dict = {'idx': idx, 
                                'basic-level-lbl': "Airplane", 
                                'coarse-level-lbl': gdict['coarse'],
                                'fine-level-lbl': gdict['fine'],
                                'img_path': os.path.join(img_root_dir, img_id) + ext,
                                'metadata': {}
                                }
                    writer.write(obj_dict)
                    test_unified_dicts.append(obj_dict)
        else:
            with open(test_out_path, "r") as reader:
                for line in reader:
                    test_unified_dicts.append(json.loads(line))

        print("test_unifed_dicts (samples): ", test_unified_dicts[:3])

        train_unified_dicts = []
        if not os.path.exists(train_out_path):
            with jsonlines.open(train_out_path, "a") as writer:
                for idx, (img_id, gdict) in enumerate(tqdm(train_img2class_dict.items(), desc="[ Unifying fgvc-aircraft-2013b train ]")):
                    obj_dict = {'idx': idx, 
                                'basic-level-lbl': "Airplane", 
                                'coarse-level-lbl': gdict['coarse'],
                                'fine-level-lbl': gdict['fine'],
                                'img_path': os.path.join(img_root_dir, img_id) + ext,
                                'metadata': {}
                                }
                    writer.write(obj_dict)
                    train_unified_dicts.append(obj_dict)
        else:
            with open(train_out_path, "r") as reader:
                for line in reader:
                    train_unified_dicts.append(json.loads(line))

        print("train_unifed_dicts (samples): ", train_unified_dicts[:3])

    elif dataset_name == "stanford-dogs":
        fine_classes = fine_test_dataset.fine_classes
        coarse_classes = fine_test_dataset.coarse_classes # Stanford Dogs dataset combines the fine and coarse classes into one dataset instance

        test_unified_dicts = []
        if not os.path.exists(test_out_path):
            with jsonlines.open(test_out_path, "a") as writer:
                for idx, (img_path, tgt) in enumerate(tqdm(fine_test_dataset, desc='[ Unifying Stanford Dogs (test) ]')):
                    obj_dict = {'idx': idx, 
                                'basic-level-lbl': "Dog", 
                                'coarse-level-lbl': [coarse_classes[tgt]],
                                'fine-level-lbl': [fine_classes[tgt]],
                                'img_path': img_path,
                                'metadata': {}
                                }
                    writer.write(obj_dict)
                    test_unified_dicts.append(obj_dict)
        else:
            with open(test_out_path, "r") as reader:
                for line in reader:
                    test_unified_dicts.append(json.loads(line))
    
        train_unified_dicts = []
        if not os.path.exists(train_out_path):
            with jsonlines.open(train_out_path, "a") as writer:
                for idx, (img_path, tgt) in enumerate(tqdm(fine_train_dataset, desc='[ Unifying Stanford Dogs (train) ]')):
                    obj_dict = {'idx': idx, 
                                'basic-level-lbl': "Dog", 
                                'coarse-level-lbl': [coarse_classes[tgt]],
                                'fine-level-lbl': [fine_classes[tgt]],
                                'img_path': img_path,
                                'metadata': {}
                                }
                    writer.write(obj_dict)
                    train_unified_dicts.append(obj_dict)
        else:
            with open(train_out_path, "r") as reader:
                for line in reader:
                    train_unified_dicts.append(json.loads(line))

    elif dataset_name == "stanford-cars":  # TODO: 'stanford-cars' is incomplete
        fine_classes = fine_test_dataset.fine_classes
        coarse_classes = fine_test_dataset.coarse_classes # Stanford Cars dataset combines the fine and coarse classes into one dataset instance

        test_unified_dicts = []
        if not os.path.exists(test_out_path):
            with jsonlines.open(test_out_path, "a") as writer:
                for idx, (img_path, tgt) in enumerate(tqdm(fine_test_dataset, desc='[ Unifying Stanford Cars (test) ]')):
                    obj_dict = {'idx': idx, 
                                'basic-level-lbl': "Car", 
                                'coarse-level-lbl': [coarse_classes[tgt]],
                                'fine-level-lbl': [fine_classes[tgt]],
                                'img_path': img_path,
                                'metadata': {}
                                }
                    writer.write(obj_dict)
                    test_unified_dicts.append(obj_dict)
        else:
            with open(test_out_path, "r") as reader:
                for line in reader:
                    test_unified_dicts.append(json.loads(line))
    
        train_unified_dicts = []
        if not os.path.exists(train_out_path):
            with jsonlines.open(train_out_path, "a") as writer:
                for idx, (img_path, tgt) in enumerate(tqdm(fine_train_dataset, desc='[ Unifying Stanford Cars (train) ]')):
                    obj_dict = {'idx': idx, 
                                'basic-level-lbl': "Car",
                                'coarse-level-lbl': [coarse_classes[tgt]],
                                'fine-level-lbl': [fine_classes[tgt]],
                                'img_path': img_path,
                                'metadata': {}
                                }
                    writer.write(obj_dict)
                    train_unified_dicts.append(obj_dict)
        else:
            with open(train_out_path, "r") as reader:
                for line in reader:
                    train_unified_dicts.append(json.loads(line))

    elif dataset_name == "nabirds":

        fine_classes = fine_test_dataset.fine_classes
        coarse_classes = fine_test_dataset.coarse_classes # NABirds dataset combines the fine and coarse classes into one dataset instance

        print("[ test-dataset ] \n", fine_test_dataset)
        print("[ train-dataset ] \n", fine_train_dataset)

        test_unified_dicts = []
        if not os.path.exists(test_out_path):
            with jsonlines.open(test_out_path, "a") as writer:
                for idx, (img_path, tgt) in enumerate(tqdm(fine_test_dataset, desc='[ Unifying NABirds (test) ]')):
                    obj_dict = {'idx': idx, 
                                'basic-level-lbl': "Bird", 
                                'coarse-level-lbl': coarse_classes[tgt],
                                'fine-level-lbl': [fine_classes[tgt][1]],
                                'img_path': img_path,
                                'metadata': {}
                                }
                    
                    writer.write(obj_dict)
                    test_unified_dicts.append(obj_dict)
        else:
            with open(test_out_path, "r") as reader:
                for line in reader:
                    test_unified_dicts.append(json.loads(line))
    
        train_unified_dicts = []
        if not os.path.exists(train_out_path):
            with jsonlines.open(train_out_path, "a") as writer:
                for idx, (img_path, tgt) in enumerate(tqdm(fine_train_dataset, desc='[ Unifying NABirds (train) ]')):
                    obj_dict = {'idx': idx, 
                                'basic-level-lbl': "Bird", 
                                'coarse-level-lbl': coarse_classes[tgt],
                                'fine-level-lbl': [fine_classes[tgt][1]],
                                'img_path': img_path,
                                'metadata': {}
                                }
                    writer.write(obj_dict)
                    train_unified_dicts.append(obj_dict)
        else:
            with open(train_out_path, "r") as reader:
                for line in reader:
                    train_unified_dicts.append(json.loads(line))

    else:
        raise ValueError("Invalid 'dataset_name'.")


class iNaturalist(VisionDataset):
    """`iNaturalist 2021 Dataset.

        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    train_folder = 'train/'
    val_folder = 'val/'
    file_list = {
        'train_imgs': ('https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz',
                       'train.tar.gz', ''),
        'val_imgs': ('https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz',
                       'val.tar.gz', ''),
        'train_annos': ('https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz',
                       'train.json.tar.gz', ''),
        'val_annos': ('https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz',
                       'val.json.tar.gz', ''),
    }

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(iNaturalist, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            if not (os.path.exists(os.path.join(self.root, self.file_list['train_imgs'][1]))
                    and os.path.exists(os.path.join(self.root, self.file_list['train_annos'][1]))
                    and os.path.exists(os.path.join(self.root, self.file_list['val_imgs'][1]))
                    and os.path.exists(os.path.join(self.root, self.file_list['val_annos'][1]))):
                print('Downloading...')
                self._download()
            print('Extracting...')
            extract_archive(os.path.join(self.root, self.file_list['train_imgs'][1]))
            extract_archive(os.path.join(self.root, self.file_list['train_annos'][1]))
            extract_archive(os.path.join(self.root, self.file_list['val_imgs'][1]))
            extract_archive(os.path.join(self.root, self.file_list['val_annos'][1]))
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        
        anno_filename = split + '.json'
        with open(os.path.join(self.root, anno_filename), 'r') as fp:
            all_annos = json.load(fp)

        self.annos = all_annos['annotations']
        self.images = all_annos['images']
        self.classes = all_annos['categories']
        self.fine_classes = [[cname['name'].strip(), cname['common_name'].strip()] for cname in self.classes]

        print("self.annos (length) : ", len(self.annos))
        print("self.images (length) : ", len(self.images))
        print("self.classes (length) : ", len(self.classes))

        demo_idx = 305
        print("self.annos (sample) : ", self.annos[demo_idx])
        print("self.images (length) : ", self.images[demo_idx])
        print("\nself.classes (length) : ", self.classes[demo_idx])

        self.class_supercategory = list(set([cls['supercategory'] for cls in self.classes]))  # Basic-level lbl
        self.family =  list(set([cls['family'].strip() for cls in self.classes]))

        # Coarse-classes need to be generated by GPT-4
        coarse_class_path = "coarse-classes.txt"
        coarse_family_aggr_path = "coarse-classes-family.json"  # 'self.coarse_classes' aggregated by 'family' name
        self.coarse_classes = []
        self.coarse_classes_family = defaultdict(list)  # Key - 'family' name / Value - GPT-4-generated coarse-grained label

        if not os.path.exists(os.path.join(self.root, coarse_class_path)):
            load_dotenv()  # Load environment variables from .env file
            api_key = os.getenv("OPENAI_API_KEY")  # Access the API key using the variable name defined in the .env file
            model = 'gpt-4-vision-preview'
            
            client = OpenAI(api_key=api_key)

            system_prompt = "You are a helpful assistant for fine-grained species classification."
            user_prompt = '''
            Generate a coarse-grained label for the following fine-grained animal/insect/mollusk/plant types. \
            For example, if the organism is a "Domestic Sheep (Ovis aries)" generate "Sheep", and if the organism is "The Dear-Headed Chihuahua (Canis Lupus)", generate "Dog".
            Output format is as follows:

            Fine-grained Concept Name: Domestic Sheep (Ovis aries)
            Coarse-grained Name: Sheep

            Fine-grained Concept Name: Northern beardless tyrannulet (Camptostoma imberbe) 
            Coarse-grained Name: Flycatcher

            Fine-grained Concept Name: Pagoda Dogwood (Cornus alternifolia)
            Coarse-grained Name: Tree
            '''
            with open(os.path.join(self.root, coarse_class_path), "a") as writer:
                for idx, cname in enumerate(tqdm(self.fine_classes, desc='[ Generating coarse-grained labels for iNaturalist (2021)... ]')):
                    inp_user_prompt = user_prompt.strip() + "\n\n" + "Fine-grained Concept Name: " + cname[0] + f" ({cname[1]})" + "\nCoarse-grained Name:"
                    _, response = openai_gpt_call(client, system_prompt, inp_user_prompt, model)
                    print(f"[({idx+1}) = Response: {response} ]")
                    self.coarse_classes.append(response.strip())
                    writer.write(f"{idx+1} {response.strip()}\n")
        else:
            print('[ Loading coarse-grained labels for iNaturalist (2021)... ]')
            df = pd.read_csv(os.path.join(self.root, coarse_class_path), names=['raw_line'])
            self.coarse_classes = df['raw_line'].apply(lambda x: ' '.join(x.split(' ')[1:])).tolist()
            print(f"[ self.coarse_classes : {self.coarse_classes[:10]}]")
            print(f"[ self.coarse_class (length) : {len(self.coarse_classes)} ]")
            
        # Aggregate GPT-4-generated labels according to same 'family'
        # Include the 'family' label as one of the multiple labels as well
        for idx, cname in enumerate(tqdm(self.coarse_classes, desc='Mapping generated coarse-grained labels to family names')):
            family_name = self.classes[idx]['family'].strip()
            self.coarse_classes_family[family_name].append(cname.strip())
        
        self.coarse_classes_family = {family_name: list(set(coarse_lbls)) for family_name, coarse_lbls in self.coarse_classes_family.items()}
        
        if not os.path.exists(os.path.join(self.root, coarse_family_aggr_path)):
            with open(os.path.join(self.root, coarse_family_aggr_path), "w") as writer:
                json.dump(self.coarse_classes_family, writer)
                print(f"[ Saved to {os.path.join(self.root, coarse_family_aggr_path)}]")
        else:
            with open(os.path.join(self.root, coarse_family_aggr_path), "r") as reader:
                self.coarse_classes_family = json.load(reader)
                print(f"[ Loaded from {os.path.join(self.root, coarse_family_aggr_path)}]")


    def __getitem__(self, index):
        path = os.path.join(self.root, self.images[index]['file_name'])
        target = self.annos[index]['category_id']

        # TODO: Comment out loader when constructing - since I'm only using 'path' anyways
        # image = self.loader(path)
        # if self.transform is not None:
        #     image = self.transform(image)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # return image, target
        return path, target

    def __len__(self):
        return len(self.images)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, self.train_folder)) and os.path.exists(os.path.join(self.root, self.val_folder)))

    def _download(self):
        for url, filename, md5 in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
            if not check_integrity(os.path.join(self.root, filename), md5):
                raise RuntimeError("File not found or corrupted.")


class NABirds(VisionDataset):
    """`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'images'
    filename = 'nabirds.tar.gz'
    md5 = 'df21a9e4db349a14e2b08adfd45873bd'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=None):
        super(NABirds, self).__init__(root, transform=transform, target_transform=target_transform)
        if download is True:
            msg = ("The dataset is no longer publicly accessible. You need to "
                   "download the archives externally and place them in the root "
                   "directory.")
            raise RuntimeError(msg)
        elif download is False:
            msg = ("The use of the download flag is deprecated, since the dataset "
                   "is no longer publicly accessible.")
            warnings.warn(msg, RuntimeWarning)

        dataset_path = root
        if not os.path.isdir(dataset_path):
            if not check_integrity(os.path.join(root, self.filename), self.md5):
                raise RuntimeError('Dataset not found or corrupted.')
            extract_archive(os.path.join(root, self.filename))
        self.loader = default_loader
        self.train = train

        image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
                                  sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        
        # Since the raw labels are non-continuous, map them to new ones
        self.label_map = self.get_continuous_class_map(image_class_labels['target'])  # 0~554 classes
        train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = image_paths.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        # Load in the train / test split
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        # Load in the class data
        self.class_names = self.load_class_names(dataset_path)  # Contains both the coarse- and fine-grained class labels
        self.class_hierarchy = self.load_hierarchy(dataset_path)

        # Split fine-grained (classes.txt) and coarse-grained classes (hierarchy.txt)
        self.fine_classes = [(cidx, cname) for cidx, cname in self.class_names.items() if int(cidx) in self.label_map]
        
        self.coarse_classes = self._process_coarse_lbls()  # Assign top-most labels to disambiguate overlapping labels
        self.fine_classes = self._process_fine_lbls()  # Remove the (Adult male), (Female/juvenile) from the 'self.fine_classes' labels
        # demo_idx = 10
        # print("[ self.coarse_classes ]\n", self.coarse_classes[361])
        # print("[ self.fine_classes ]\n", self.fine_classes[361])
        # print("[ label_map ] : ", self.label_map[817])
        # exit()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = self.label_map[sample.target]
        # img = self.loader(path)

        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        # return img, target
        return path, target


    def get_continuous_class_map(self, class_labels):
        label_set = set(class_labels)
        return {int(k): i for i, k in enumerate(label_set)}


    def load_class_names(self, dataset_path=''):
        names = {}

        with open(os.path.join(dataset_path, 'classes.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                class_id = pieces[0]
                names[class_id] = ' '.join(pieces[1:])

        return names


    def load_hierarchy(self, dataset_path=''):
        parents = {}

        with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
            for line in f:
                pieces = line.strip().split()
                child_id, parent_id = pieces
                parents[child_id] = parent_id

        return parents

    def _process_fine_lbls(self):
        if len(self.fine_classes) == 0:
            raise Exception("Error: self.fine_classes need to processed first.")
        
        pattern = re.compile(r'\(.+?\)')  # Detect any additional category; e.g., (Adult Male), (Female/juvenile)
        
        fine_grained_lbls = []
        for cidx, cname in self.fine_classes:
            pattern_found = pattern.findall(cname)
            if len(pattern_found) > 0:
                cname = cname.replace(pattern_found[0], "").strip()
            fine_grained_lbls.append((cidx, cname))

        return fine_grained_lbls
    
    def _process_coarse_lbls(self):
        if len(self.fine_classes) == 0:
            raise Exception("Error: self.fine_classes need to processed first.")
        
        pattern = re.compile(r'\(.+?\)')  # Detect any additional category; e.g., (Adult Male), (Female/juvenile)
        
        coarse_grained_lbls = []
        for cidx, cname in self.fine_classes:
            # Check if there are any additional category (e.g., (Adult Male), (Female/juvenile)) - sub-branch of fine-grained category
            parent_idx = self.class_hierarchy[cidx]

            # If so, take the top-most granularity label as the coarse_grained label
            if self.class_names[parent_idx].strip().lower() in cname.strip().lower() or len(pattern.findall(cname)) > 0:
                grand_parent_idx = self.class_hierarchy[parent_idx]
                coarse_grained_lbl_str = self.class_names[grand_parent_idx]
            else:
                coarse_grained_lbl_str = self.class_names[parent_idx]

            if ',' in coarse_grained_lbl_str:
                coarse_lbls = [s.replace("and", "").strip() for s in coarse_grained_lbl_str.split(',')]
            elif " and " in coarse_grained_lbl_str:
                coarse_lbls = [s.strip() for s in coarse_grained_lbl_str.split('and')]
            else:
                coarse_lbls = [coarse_grained_lbl_str.strip()]

            coarse_grained_lbls.append(coarse_lbls)
                
        return coarse_grained_lbls


class Dogs(VisionDataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Dogs, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = os.path.join(self.root, 'Images')
        self.annotations_folder = os.path.join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        self._flat_breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]

        print(f"[ self._breeds (length: {len(self._breeds)}): {self._breeds[:10]} ]")
        print(f"[ self._flat_breed_images (length: {len(self._flat_breed_images)}): {self._flat_breed_images[:10]} ]")
        self.fine_classes = list(set([(' '.join(img_path.split('/')[0].split('-')[-1].split('_')), tgt) for img_path, tgt in self._flat_breed_images]))
        self.fine_classes = [class_lbl for class_lbl, _ in sorted(self.fine_classes, key=lambda x: x[1])]
        print(f"[ self.fine_classes (length: {len(self.fine_classes)}): {self.fine_classes[:10]} ]")

        # Coarse-classes need to be generated by GPT-4 (parsing with '_' alone can be in accurate)
        coarse_class_path = "coarse-classes.txt"

        if not os.path.exists(os.path.join(self.root, coarse_class_path)):
            self.coarse_classes = []
            load_dotenv()  # Load environment variables from .env file
            api_key = os.getenv("OPENAI_API_KEY")  # Access the API key using the variable name defined in the .env file
            model = 'gpt-4-vision-preview'
            
            client = OpenAI(api_key=api_key)

            system_prompt = "You are a helpful assistant for fine-grained dog classification."
            user_prompt = '''
            Generate a coarse-grained label for the following fine-grained dog types. \
            For example, if the dog is a "Cavalier King Charles Spaniel" generate "Spaniel", and if the dog is "The Dear-Headed Chihuahua", generate "Chihuahua".
            Output format is as follows:

            Fine-grained Dog Name: Cavalier King Charles Spaniel
            Dog Name: Spaniel

            Fine-grained Dog Name: Curly-coated retriever
            Dog Name: Retriever

            Fine-grained Dog Name: Newfoundland
            Dog Name: Newfoundland
            '''
            with open(os.path.join(self.root, coarse_class_path), "a") as writer:
                for idx, cname in enumerate(tqdm(self.fine_classes, desc='[ Generating coarse-grained labels for Stanford Dogs... ]')):
                    inp_user_prompt = user_prompt.strip() + "\n\n" + "Fine-grained Dog Name: " + cname + "\nDog Name:"
                    raw_response, response = openai_gpt_call(client, system_prompt, inp_user_prompt, model)
                    print(f"[({idx+1}) = Response: {response} ]")
                    self.coarse_classes.append(response.strip())
                    writer.write(f"{idx+1} {response.strip()}\n")
        else:
            print('[ Loading coarse-grained labels for Stanford Dogs... ]')
            df = pd.read_csv(os.path.join(self.root, coarse_class_path), names=['raw_line'])
            self.coarse_classes = df['raw_line'].apply(lambda x: ' '.join(x.split(' ')[1:])).tolist()
            print(f"[ self.coarse_classes : {self.coarse_classes}]")
            print(f"[ self.coarse_classes (class #) : {len(set(self.coarse_classes))}]")

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        image_name, target = self._flat_breed_images[index]
        image_path = os.path.join(self.images_folder, image_name)
        image = self.loader(image_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # return image, target
        return image_path, target

    def download(self):
        import tarfile

        if os.path.exists(os.path.join(self.root, 'Images')) and os.path.exists(os.path.join(self.root, 'Annotation')):
            if len(os.listdir(os.path.join(self.root, 'Images'))) == len(os.listdir(os.path.join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + os.path.join(self.root, tar_filename))
            with tarfile.open(os.path.join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(os.path.join(self.root, tar_filename))

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)" % (len(self._flat_breed_images), len(counts.keys()),
                                                                     float(len(self._flat_breed_images)) / float(
                                                                         len(counts.keys()))))

        return counts


class Cars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    file_list = {
        'train_imgs': ('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', 'cars_test'),
        'test_imgs': ('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', 'cars_train'),
        'annos': ('http://imagenet.stanford.edu/internal/car196/cars_annos.mat', 'cars_annos.mat')
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")
        
        super(Cars, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self._split = "train" if train else "test"

        self.root = root
        devkit = os.path.join(self.root, "devkit")

        if self._split == "train":
            self._annotations_mat_path = os.path.join(devkit, "cars_train_annos.mat")
            self._images_base_path = os.path.join(self.root, "cars_train")
        else:
            self._annotations_mat_path = os.path.join(self.root, "cars_test_annos_withlabels.mat")
            self._images_base_path = os.path.join(self.root, "cars_test")

        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        
        self.samples = [
            (
                str(os.path.join(self._images_base_path, annotation['fname'])),
                annotation['class'] - 1,   # Original target mapping starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)['annotations']
        ]

        self.fine_classes = sio.loadmat(os.path.join(devkit, "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.fine_classes)}

        coarse_categories = ["sedan", "SUV", "coupe", "convertible", "pickup", "hatchback", "van"]
        self.coarse_classes = []
        coarse_class_path = "coarse-classes.txt"

        load_dotenv()  # Load environment variables from .env file
        api_key = os.getenv("OPENAI_API_KEY")  # Access the API key using the variable name defined in the .env file
        model = 'gpt-4-vision-preview'
        client = OpenAI(api_key=api_key)

        system_prompt = "You are a helpful assistant for fine-grained car classification."
        user_prompt = '''
        Generate a coarse-grained label for the following fine-grained car types. \
        The coarse-grained car types are as follows: ["sedan", "SUV", "coupe", "convertible", "pickup", "hatchback", "van"]. \
        For example, if the car is a "Ford F-150 Regular Cab 2012" generate "pickup", and if the car is "Chrysler 300 SRT-8 2010", generate "sedan".
        Output format is as follows:

        Fine-grained Car Name: Ford F-150 Regular Cab 2012
        Car Name: pickup

        Fine-grained Car Name: Chrysler 300 SRT-8 2010
        Car Name: sedan

        Fine-grained Car Name: Hyundai Santa Fe 2008
        Car Name: SUV
        '''

        if not os.path.exists(os.path.join(self.root, coarse_class_path)):
            with open(os.path.join(self.root, coarse_class_path), "a") as writer:
                for idx, fine_cname in enumerate(tqdm(self.fine_classes, desc='[ Generating coarse-grained labels for Stanford Cars... ]')):
                    coarse_cname = [cname for cname in coarse_categories if cname.lower().strip() in fine_cname.lower().strip()]
                    if len(coarse_cname) == 0:
                        # TODO: Generate coarse-grained label if it doesn't exist in fine-grained classes
                        inp_user_prompt = user_prompt.strip() + "\n\n" + "Fine-grained Car Name: " + fine_cname + "\nCar Name:"
                        raw_response, response = openai_gpt_call(client, system_prompt, inp_user_prompt, model)
                        print(f"[({idx+1}) = Coarse-grained lbl (generated): {response} ]")
                        coarse_cname = response.strip()
                    else:  # Coarse-grained label already exists inside the fine-grained label
                        coarse_cname = coarse_cname[0]
                        print(f"[({idx+1}) = Coarse-grained lbl: {coarse_cname} ]")
                    self.coarse_classes.append(coarse_cname)
                    writer.write(f"{idx+1} {coarse_cname.strip()}\n")
        else:
            print('[ Loading coarse-grained labels for Stanford Cars... ]')
            df = pd.read_csv(os.path.join(self.root, coarse_class_path), names=['raw_line'])
            self.coarse_classes = df['raw_line'].apply(lambda x: ' '.join(x.split(' ')[1:])).tolist()
            print(f"[ self.coarse_classes : {self.coarse_classes[:10]}]")
        
        # print(f"=== self.coarse_classes === (# classes: {len(set(self.coarse_classes))})\n{self.coarse_classes}")
        # print(f"\n\n=== self.fine_classes ===(# classes: {len(set(self.fine_classes))})\n{self.fine_classes}")
        # assert len(self.coarse_classes) == len(self.fine_classes)

        print(f"=== [ Split: {self._split} ] ===")
        print("self.samples (length): ", len(self.samples))
        print("self.samples (sample): ", self.samples[:5])
        print("self.fine_classes (sample 0): ", self.fine_classes[self.samples[0][1]])
        print("self.coarse_classes(sample 0): ", self.coarse_classes[self.samples[0][1]])
        print("-"*30)
        print("self.fine_classes (sample 1): ", self.fine_classes[self.samples[1][1]])
        print("self.coarse_classes(sample 1): ", self.coarse_classes[self.samples[1][1]])

    def __getitem__(self, index):
        path, target = self.samples[index]
        path = os.path.join(self.root, path)

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # return image, target
        return path, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, self.file_list['train_imgs'][1]))
                and os.path.exists(os.path.join(self.root, self.file_list['test_imgs'][1]))
                and os.path.exists(os.path.join(self.root, self.file_list['annos'][1])))

    def _download(self):
        print('Downloading...')
        for url, filename in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
        print('Extracting...')
        archive = os.path.join(self.root, self.file_list['imgs'][1])
        extract_archive(archive)


class Aircraft(VisionDataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    
    # Basic-level: Airplane
    # Coarse-grained (manufacturer): e.g., Boeing / Airbus / British Aerospace
    # Fine-grained (family / variant): e.g., Boeing 707-320 / Boeing 707
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(self, root, train=True, class_type=['variant'], transform=None,
                 target_transform=None, download=False):
        
        super(Aircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        split = 'trainval' if train else 'test'
        
        self.loader = default_loader
        self.img_folder = os.path.join(root, 'data', 'images')

        if download:
            self.download()

        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        
        if len(class_type) > 1:  # Multiple class_types - e.g., ['variant', 'family']
            # Combine the multiple class_types into one single dataset
            assert all([ctype in self.class_types for ctype in class_type])
            
            image_ids, targets, classes, class_to_idx, samples = [], [], [], [], []

            for idx, ctype in enumerate(class_type):
                self.class_type = ctype
                self.split = split
                self.classes_file = os.path.join(self.root, 'data', 'images_%s_%s.txt' % (self.class_type, self.split))

                (img_ids, tgts, cls, cls2idx) = self.find_classes()
                if idx == 0:
                    prev_cls_cnt = len(cls2idx)
                else:
                    tgts = [t + prev_cls_cnt for t in tgts]
                
                smpls = self.make_dataset(img_ids, tgts)

                image_ids.extend(img_ids)
                targets.extend(tgts)
                classes.extend(cls)
                class_to_idx.append(cls2idx)
                samples.extend(smpls)

        elif len(class_type) != 0 and class_type[0] in self.class_types:

            self.class_type = class_type[0]
            self.split = split
            self.classes_file = os.path.join(self.root, 'data',
                                            'images_%s_%s.txt' % (self.class_type, self.split))

            (image_ids, targets, classes, class_to_idx) = self.find_classes()
            samples = self.make_dataset(image_ids, targets)

            # print(f"\n====== SPLIT: [ {split} ] ======\n")
            # print(f"image_ids (length: {len(image_ids)}): {image_ids[:2]}")
            # print(f"targets (length: {len(targets)}): {targets[:2]}")
            # print(f"targets (set) (length: {len(targets)}): {set(targets)}")
            # print(f"classes (length: {len(classes)}): {classes[:2]}")
            # print(f"class_to_idx (length: {len(class_to_idx)}): {class_to_idx}")
            # print(f"samples (length: {len(samples)}): {samples[:2]}\n")

        elif class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # return sample, target
        return path, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s...' % self.url)
        tar_name = self.url.rpartition('/')[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print('Extracting %s...' % tar_path)
        extract_archive(tar_path)
        print('Done!')

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]).strip())

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images


class Cub2011(VisionDataset):
    """`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    img_folder = 'images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Cub2011, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        print("[ self.data ]\n", self.data)

        class_names = pd.read_csv(os.path.join(self.root, 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        self.fine_classes = [" ".join(cname.split('.')[1].split('_')) if '_' in cname \
                                    else "".join(cname.split('.')[1]) for cname in self.class_names]
        
        # Coarse-classes need to be generated by GPT-4 (parsing with '_' alone can be in accurate)
        coarse_class_path = "coarse-classes.txt"

        if not os.path.exists(os.path.join(self.root, coarse_class_path)):
            self.coarse_classes = []
            load_dotenv()  # Load environment variables from .env file
            api_key = os.getenv("OPENAI_API_KEY")  # Access the API key using the variable name defined in the .env file
            model = 'gpt-4-vision-preview'
            
            client = OpenAI(api_key=api_key)

            system_prompt = "You are a helpful assistant for fine-grained bird classification."
            user_prompt = '''
            Generate a coarse-grained label for the following fine-grained bird types. \
            For example, if the bird is a "bald eagle (Haliaeetus leucocephalus)" generate "Eagle", and if the bird is "Pine grosbeak", generate "Finch".
            Output format is as follows:

            Fine-grained Bird Name: Bald eagle
            Bird Name: Eagle

            Fine-grained Bird Name: Pine grosbeak
            Bird Name: Finch

            Fine-grained Bird Name: The black backed woodpecker
            Bird Name: Woodpecker
            '''
            with open(os.path.join(self.root, coarse_class_path), "a") as writer:
                for idx, cname in enumerate(tqdm(self.fine_classes, desc='[ Generating coarse-grained labels for CUB-200-2011... ]')):
                    inp_user_prompt = user_prompt.strip() + "\n\n" + "Fine-grained Bird Name: " + cname + "\nBird Name:"
                    raw_response, response = openai_gpt_call(client, system_prompt, inp_user_prompt, model)
                    print(f"[({idx+1}) = Response: {response} ]")
                    self.coarse_classes.append(response.strip())
                    writer.write(f"{idx+1} {response.strip()}\n")

        else:
            print('[ Loading coarse-grained labels for CUB-200-2011... ]')
            self.coarse_classes = pd.read_csv(os.path.join(self.root, coarse_class_path), sep=' ', names=['class_name'], usecols=[1])['class_name'].tolist()

        # print("[ self.fine_classes ]\n")
        # print(self.fine_classes)
        # print(f"\n\n[ self.coarse_classes (class cnt: {len(set(self.coarse_classes))}) ]\n")
        # print(self.coarse_classes)
    
        assert len(self.fine_classes) == len(self.coarse_classes)

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        # try:
        #     self._load_metadata()
        # except Exception as e:
        #     print(e)
        #     return False
        self._load_metadata()

        print("!!! self.data \n", self.data)

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.img_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_file_from_google_drive(self.file_id, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.img_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return path, target
        # return img, target
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="/shared/nas/data/m1/jk100/data/")
    parser.add_argument("--dataset-name", type=str, required=True, default=None, help="inaturalist, cub-200, fgvc-aircraft, stanford-dogs, stanford-cars, nabirds")

    args = parser.parse_args()

    root_dir = args.root_dir
    dataset_name = args.dataset_name

    print(f"[ Dataset: {args.dataset_name} ]")

    fine_train_dataset = None
    fine_test_dataset = None
    coarse_train_dataset = None
    coarse_test_dataset = None

    if args.dataset_name in ['inaturalist', 'cub-200', 'fgvc-aircraft', 'stanford-cars', 'stanford-dogs', 'nabirds']:
        if dataset_name == 'inaturalist':
            data_path = os.path.join(root_dir, "inaturalist")
            fine_test_dataset = iNaturalist(data_path, split='val', download=False)
            fine_train_dataset = iNaturalist(data_path, split='train', download=False)
        
        elif dataset_name == 'nabirds':
            data_path = os.path.join(root_dir, "nabirds")
            fine_train_dataset = NABirds(data_path, train=True, download=False)
            fine_test_dataset = NABirds(data_path, train=False, download=False)

        elif dataset_name == "cub-200":
            data_path = os.path.join(root_dir, "CUB_200_2011")
            # For 'CUB_200_2011', the coarse-grained lbls and fine-grained lbls are contained in the same class instantiation
            fine_train_dataset = Cub2011(data_path, train=True, download=False)
            fine_test_dataset = Cub2011(data_path, train=False, download=False)

        elif dataset_name == "fgvc-aircraft":
            data_path = os.path.join(root_dir, "fgvc-aircraft-2013b")

            # Fine-grained = 'class_type == variant / family'
            # combine 'variant' and 'family' into same category - 'fine-grained'
            # each sample will have multiple labels - e.g., [[Boeing 707-max, Boeing 707], [...], ...]
            fine_train_dataset = Aircraft(data_path, train=True, class_type=["variant", "family"], download=False)
            fine_test_dataset = Aircraft(data_path, train=False, class_type=["variant", "family"], download=False)

            # Coarse-grained = 'class_type == manufacturer'
            coarse_train_dataset = Aircraft(data_path, train=True, class_type=["manufacturer"], download=False)
            coarse_test_dataset = Aircraft(data_path, train=False, class_type=["manufacturer"], download=False)

        elif dataset_name == "stanford-cars":
            data_path = os.path.join(root_dir, "stanford_cars")
            fine_train_dataset = Cars(data_path, train=True, download=False)
            fine_test_dataset = Cars(data_path, train=False, download=False)

        elif dataset_name == "stanford-dogs":
            data_path = os.path.join(root_dir, "stanford_dogs")
            fine_train_dataset = Dogs(data_path, train=True, download=False)
            fine_test_dataset = Dogs(data_path, train=False, download=False)

        # Unify the loaded dataset formats into 
        unify_data_format(fine_train_dataset, 
                          fine_test_dataset, 
                          coarse_train_dataset,
                          coarse_test_dataset,
                          dataset_name=dataset_name,
                          root_dir=data_path)

    else:
        raise ValueError("--dataset-name has to be one of: ['inaturalist', 'cub-200', 'fgvc-aircraft', 'stanford-cars', 'stanford-dogs']")


