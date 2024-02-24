# Finer: A Benchmark for Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models


**Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models** [[Paper](https://arxiv.org/abs/2310.03744)] <br>
[Jeonghwan Kim](https://wjdghks950.github.io/), [Heng Ji](https://blender.cs.illinois.edu/hengji.html)

## Contents
- [Install](#install)
- [Dataset](#dataset)
- [Evaluation](#evaluation)

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
**Usage and License Notices**: This dataset inherits the Usage and License Notices from LLaVA (Liu et al., 2023). The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.


## Install

1. Clone the current repository to reproduce the Finer evaluation
```bash
git clone https://github.com/wjdghks950/Finer.git
cd Finer
```

2. Install and setup the `llava` conda environment from the official [LLaVA repository](https://github.com/haotian-liu/LLaVA) to run LLaVA.

3. Install and setup the `lavis` conda environment from the official [LAVIS repository](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) to run InstructBLIP and BLIP-2.


## Dataset
- First, set up a separate `data` directory in the same directory as the `LLaVA` and `LAVIS` dirs.
- Download the datasets for each of the dataset from the following [link](https://drive.google.com/drive/folders/1s0-g9cWA3yUufe4jkEq-01SPko4DIASl?usp=drive_link) and structure them into the format below, where `...` indicates the downloaded dataset files including images and their annotations:
```
├── inaturalist
│   └── ...
├── fgvc-aircraft-2013b
│   └── ...
├── CUB_200_2011
│   └── ...
├── nabirds
│   └── ...
└── stanford_cars
│   └── ...
└── stanford_dogs
│   └── ...
```
- If you'd like to set the datasets up yourself, under the `data` directory, set up the following directories separately
    - `data/inaturalist` - Download the evaluation dataset [images](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz) / [annotations](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz) and the training dataset [images](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz) / [annotations](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz);
    - `data/fgvc-aircraft-2013b` - Download the dataset from the following link: [dataset link](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)
    - `data/nabirds` - Download the dataset from the following link: [dataset link](https://dl.allaboutbirds.org/nabirds). You need to agree to the Terms of Use and get the downloadable link manually; follow the instructions in the nabirds dataset link.
    - `data/CUB_200_2011` - Download the dataset from the following link: [dataset link](https://data.caltech.edu/records/20098)
    - `data/stanford_dogs` - Download the dataset from the following link: [images](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar) / [annotations](http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar) / [train/test_split](http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar)
    - `data/stanford_cars` - Download the dataset from the following Kaggle link: [dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
- In each of the dataset (e.g., `data/stanford_cars`) there is a concept to attribute dictionary of file format (`parsed-{dataset_name}-{model_name}-wiki-text-combined.json`) in the following format:
```json
    {
        "id": 41,
        "name": "Acura ZDX Hatchback 2012",
        "attr_binomial": {
            "required": [
                "Five-door coupe-like hatchback body style",
                "Unique, sloping roofline that tapers towards the rear",
                "Shield-shaped front grille with the Acura logo",
                "Angular headlight design with integrated daytime running lights",
                "Distinctive, raised rear end with a high-mounted spoiler",
                "Dual exhaust outlets at the rear",
                "Sharp character lines along the sides"
            ],
            "likely": [
                "LED taillights",
                "19-inch alloy wheels",
                "Panoramic glass roof",
                "Chrome door handle accents",
                "Body-colored side mirrors with integrated turn signals",
                "Sculpted hood design"
            ]
        }
    }
```
- For the superordinate, coarse-level and fine-level labels, within each `data/{dataset_name}` folder, there is a file name `unified-{dataset_name}-{split}-combined.jsonl`
```json
{
    "idx": 278, 
    "basic-level-lbl": "Airplane", 
    "coarse-level-lbl": ["Boeing"], 
    "fine-level-lbl": ["Boeing 737", "737-800"], 
    "img_path": "{data_dir_path}/fgvc-aircraft-2013b/data/images/1935750.jpg", "metadata": {}}
```


## Instruction-Tuning
Fine-tune the LLaVA-v1.5 (7B) model using the finer-mixture (`LLaVA/playground/data/llava_v1_5_mix865k_attr_gen_fine_answer_inaturalist.json`), which was built on top of the llava instruction-tuning mixture, as follows:
```bash
cd LLaVA/scripts/v1_5
./finetune_lora.sh
```
It took approximately ~28 hours to fine-tune the LLaVA-v1.5(7B) on the finer-mixture on 4 V100s (16G)


## Citation

If you find Finer useful for your research and applications, please cite using this BibTeX:
```bibtex

@misc{kim2024finer,
      title={Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models}, 
      author={Kim, Jeonghwan and Ji, Heng},
      publisher={arXiv:2310.03744},
      year={2024},
}
```