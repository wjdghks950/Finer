#!/bin/bash

CTGR2IMG_PATH="ctgr2img_dict.json"

echo "task_type_id [\"high_coarse: 0\", \"coarse: 1\", \"fine: 2\", \"attr_gen: 3\", \"Conversation Mode: 4\"] >> "
read task_type_id

echo "dataset name ['inaturalist', 'cub_200_2011', 'fgvc_aircraft', 'nabirds', 'stanford_dogs', 'stanford_cars'] >> "
read dataset

echo "model_name ['llava-7b', 'llava-13b', 'llava-7b-regmix', 'llava-7b-attrfine' 'instructblip-7b', 'instructblip-13b', 'gpt-4']"
read model_name

if [ "$dataset" = "inaturalist" ]; then
  dataset_name="inaturalist"
  DATA_PATH="val.json"
elif [ "$dataset" = "cub_200_2011" ]; then
	dataset_name="CUB_200_2011"
  DATA_PATH=""unified-cub-200-test-combined.jsonl""
elif [ "$dataset" = "fgvc_aircraft" ]; then
	dataset_name="fgvc-aircraft-2013b"
  DATA_PATH="unified-fgvc-aircraft-test-combined.jsonl"
elif [ "$dataset" = "nabirds" ]; then
	dataset_name="nabirds"
  DATA_PATH="unified-nabirds-test-combined.jsonl"
elif [ "$dataset" = "stanford_cars" ]; then
	dataset_name="stanford_cars"
  DATA_PATH="unified-stanford-cars-test-combined.jsonl"
elif [ "$dataset" = "stanford_dogs" ]; then
	dataset_name="stanford_dogs"
  DATA_PATH="unified-stanford-dogs-test-combined.jsonl"
else
    echo "Not a valid input; Terminating..."
fi


if [ "$model_name" = "llava-7b" ]; then
	model_path="liuhaotian/llava-v1.5-7b"
  model_base="null"
elif [ "$model_name" = "llava-13b" ]; then
	model_path="liuhaotian/llava-v1.5-13b"
  model_base="null"
elif [ "$model_name" = "llava-7b-regmix" ]; then
	model_path="../../checkpoints/llava-v1.5-7b-lora-865k-inat2021-regmix/checkpoint-8000"
  model_base="lmsys/vicuna-7b-v1.5"
elif [ "$model_name" = "llava-7b-attrfine" ]; then
	model_path="../../checkpoints/llava-v1.5-7b-lora-865k-inat2021-attr_gen_fine_answer"
  model_base="lmsys/vicuna-7b-v1.5"
elif [ "$model_name" = "instructblip-7b" ]; then
	model_path="blip2_vicuna_instruct"
  model_base="null"
elif [ "$model_name" = "instructblip-13b" ]; then
	model_path="blip2_vicuna_instruct"
  model_base="null"
elif [ "$model_name" = "gpt-4" ]; then
	model_path="gpt-4-vision-preview"
  if [ "$task_type_id" = 3 ]; then
    echo "Use Wikipedia documents as input for attribute extraction? [y/n] >> "
    read use_wiki
  fi
else
    echo "Not a valid input; Terminating..."
fi

# Check if task_type_id is 0, 1, or 2
if [[ "$task_type_id" == "0" || "$task_type_id" == "1" || "$task_type_id" == "2" ]]; then
  echo "use_prompt [true/false] >> "
  read use_prompt

  # Only ask for $prompt_type_id if $use_prompt is True
  if [ "$use_prompt" = true ]; then
    echo "prompt_type_id [\"cot_0shot: 0\", \"cot_fewshot: 1\", \"attr_seek: 2\"] >> "
    read prompt_type_id
    # Execute LLaVA
    python -m llava.serve.cli \
      --model-path $model_path \
      --model-base $model_base \
      --data_dir /home/jk100/data/data/$dataset_name \
      --data_path $DATA_PATH \
      --image-file None \
      --task_type_id $task_type_id \
      --prompt_type_id $prompt_type_id \
      --use_prompt \
      --use_gold_coarse
  else
    prompt_type_id=0  # Set prompt_type_id to an empty string if use_prompt is not True
    python -m llava.serve.cli \
      --model-path $model_path \
      --model-base $model_base \
      --data_dir /home/jk100/data/data/$dataset_name \
      --data_path $DATA_PATH \
      --image-file None \
      --dialogue-mode single \
      --task_type_id $task_type_id \
      --use_gold_coarse
  fi

# If it's not for prompting, it's for generating attributes per concept
elif [ "$task_type_id" = 3 ]; then
  echo "Generating attributes..."
  if [[ "$modality" = "text" && "$dataset" = "inaturalist" ]]; then
    echo "Use binomial or common_name [binomial/common] ? >> "
    read input_type
  else
    input_type="common"
  fi

  if [ "$use_wiki" = "n" ]; then
    echo "Use text or image [text/image] ? >> "
    read modality
	  python -m llava.serve.cli \
          --model-path $model_path \
          --model-base $model_base \
          --data_dir /home/jk100/data/data/$dataset_name \
          --data_path $DATA_PATH \
          --image-file None \
          --task_type_id $task_type_id \
          --input_type $input_type \
          --modality $modality \
          --ctgr2img-path $CTGR2IMG_PATH \
          --max-new-tokens 256
          # --load-8bit
  else
	  python -m llava.serve.cli \
          --model-path $model_path \
          --model-base $model_base \
          --data_dir /home/jk100/data/data/$dataset_name \
          --data_path $DATA_PATH \
          --image-file None \
          --task_type_id $task_type_id \
          --input_type $input_type \
          --ctgr2img-path $CTGR2IMG_PATH \
          --max-new-tokens 256 \
          --use_wiki
          # --load-8bit
  fi
fi