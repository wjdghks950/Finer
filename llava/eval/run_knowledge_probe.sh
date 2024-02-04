#!/bin/bash

DATA_PATH="val.json"
CTGR2IMG_PATH="ctgr2img_dict.json"

echo "dataset name ['inaturalist', 'cub_200_2011', 'fgvc_aircraft', 'nabirds', 'stanford_dogs', 'stanford_cars'] >> "
read dataset

echo "model_name ['llava-7b', 'llava-13b', 'instructblip-7b', 'instructblip-13b', 'gpt-4']"
read model_name

if [ "$dataset" = "inaturalist" ]; then
	dataset_name="inaturalist"
elif [ "$dataset" = "cub_200_2011" ]; then
	dataset_name="CUB_200_2011"
elif [ "$dataset" = "fgvc_aircraft" ]; then
	dataset_name="fgvc-aircraft-2013b"
elif [ "$dataset" = "nabirds" ]; then
	dataset_name="nabirds"
elif [ "$dataset" = "stanford_cars" ]; then
	dataset_name="stanford_cars"
elif [ "$dataset" = "stanford_dogs" ]; then
	dataset_name="stanford_dogs"
fi

if [ "$model_name" = "llava-7b" ]; then
	model_path="liuhaotian/llava-v1.5-7b"
elif [ "$model_name" = "llava-13b" ]; then
	model_path="liuhaotian/llava-v1.5-13b"
elif [ "$model_name" = "instructblip-7b" ]; then
	model_path="blip2_vicuna_instruct"
elif [ "$model_name" = "instructblip-13b" ]; then
	model_path="blip2_vicuna_instruct"
elif [ "$model_name" = "gpt-4" ]; then
	model_path="gpt-4-vision-preview"
fi

python -m llava.eval.eval_knowledge_probe \
    --model-path $model_path \
    --data_dir /shared/nas/data/m1/jk100/data \
    --dataset_name $dataset_name \
    --data_path $DATA_PATH \
    --max-new-tokens 256
