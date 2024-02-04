#!/bin/bash

DATA_PATH="val.json"

echo "dataset_name ['inaturalist', 'cub_200_2011', 'fgvc_aircraft', 'nabirds', 'stanford_dogs', 'stanford_cars'] >> "
read dataset_name

echo "model_name ['llava-7b', 'llava-13b', 'instructblip-7b', 'instructblip-13b', 'gpt-4']"
read model_name

if [ "$dataset_name" = "inaturalist" ]; then
	DATA_PATH="inaturalist"
elif [ "$dataset_name" = "cub_200_2011" ]; then
	DATA_PATH="CUB_200_2011"
elif [ "$dataset_name" = "fgvc_aircraft" ]; then
	DATA_PATH="fgvc-aircraft-2013b"
elif [ "$dataset_name" = "nabirds" ]; then
	DATA_PATH="nabirds"
elif [ "$dataset_name" = "stanford_cars" ]; then
	DATA_PATH="stanford_cars"
elif [ "$dataset_name" = "stanford_dogs" ]; then
	DATA_PATH="stanford_dogs"
fi

if [ "$model_name" = "llava-7b" ]; then
	model_name_var="llava-v1.5-7b"
elif [ "$model_name" = "llava-13b" ]; then
	model_name_var="llava-v1.5-13b"
elif [ "$model_name" = "instructblip-7b" ]; then
	model_name_var="instructblip-7b" # model_name_var="blip2_vicuna_instruct_vicuna7b"
elif [ "$model_name" = "instructblip-13b" ]; then
	model_name_var="instructblip-13b" # model_name_var="blip2_vicuna_instruct_vicuna7b"
elif [ "$model_name" = "gpt-4" ]; then
	model_name_var="gpt-4"
fi

echo "use_prompt [true/false] >> "
read use_prompt

# Only ask for prompt_type_id if use_prompt is True
if [ "$use_prompt" == true ]; then
	echo "prompt_type_id [\"cot_0shot: 0\", \"cot_fewshot: 1\", \"attr_seek: 2\"] >> "
	read prompt_type_id
	python eval_fgvc.py \
		--prompt_type_id $prompt_type_id \
		--use_prompt \
		--data_dir /home/jk100/data/data/$DATA_PATH \
		--preds_dir /shared/nas/data/m1/jk100/code/ecole/LLaVA/llava/preds \
		--model_name $model_name_var
else
	prompt_type_id=0  # Set prompt_type_id to an empty string if use_prompt is not True
	python eval_fgvc.py \
		--dataset_name $dataset_name \
		--data_dir /home/jk100/data/data/$DATA_PATH \
		--preds_dir /shared/nas/data/m1/jk100/code/ecole/LLaVA/llava/preds \
		--model_name $model_name_var
fi
