#!/bin/bash

DATA_PATH="unified-stanford-dogs-test-combined.jsonl"
CTGR2IMG_PATH="ctgr2img_dict.json"

echo "task_type_id [\"high_coarse: 0\", \"coarse: 1\", \"fine: 2\", \"attr_gen: 3\"] >> "
read task_type_id

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
      --model-path liuhaotian/llava-v1.5-13b \
      --data_dir /home/jk100/data/data/stanford_dogs \
      --data_path $DATA_PATH \
      --image-file None \
      --task_type_id $task_type_id \
      --prompt_type_id $prompt_type_id \
      --use_prompt \
      --use_gold_coarse
  else
    prompt_type_id=0  # Set prompt_type_id to an empty string if use_prompt is not True
    python -m llava.serve.cli \
      --model-path liuhaotian/llava-v1.5-13b \
      --data_dir /home/jk100/data/data/stanford_dogs \
      --data_path $DATA_PATH \
      --image-file None \
      --task_type_id $task_type_id \
      --use_gold_coarse
  fi

# If it's not for prompting, it's for generating attributes per concept
elif [ "$task_type_id" = 3 ]; then
  echo "Running generating attribute..."
  echo "Use text or image [text/image] ? >> "
  read modality
  if [ "$modality" = "text" ]; then
    echo "Use binomial or common_name [binomial/common] ? >> "
    read input_type
  else
    echo "Input type is empty!"
    input_type="null"
  fi
  echo "Model name: [ llava-7b / llava-13b / gpt-4 ] >> "
  read model
  if [ "$model" = "llava-7b" ]; then
    model_path="liuhaotian/llava-v1.5-7b"
  elif [ "$model" = "llava-13b" ]; then
    model_path="liuhaotian/llava-v1.5-13b"
  elif [ "$model" = "gpt-4" ]; then
    model_path="gpt-4-vision-preview"
    echo "Use Wikipedia documents as input for attribute extraction? [y/n] >> "
    read use_wiki
  fi
  if [ "$use_wiki" = "n" ]; then
	  python -m llava.serve.cli \
          --model-path $model_path \
          --data_dir /home/jk100/data/data/stanford_dogs \
          --data_path $DATA_PATH \
          --image-file None \
          --task_type_id $task_type_id \
          --input_type $input_type \
          --modality $modality \
          --model $model \
          --ctgr2img-path $CTGR2IMG_PATH \
          --max-new-tokens 256
          # --parse_attr
          # --load-8bit
          # TODO: Arbitrarily set `max-new-tokens` to 256 to avoid over-generation by llava-7b
  else
	  python -m llava.serve.cli \
          --model-path $model_path \
          --data_dir /home/jk100/data/data/stanford_dogs \
          --data_path $DATA_PATH \
          --image-file None \
          --task_type_id $task_type_id \
          --input_type $input_type \
          --modality $modality \
          --model $model \
          --ctgr2img-path $CTGR2IMG_PATH \
          --max-new-tokens 256 \
          --use_wiki
          # --parse_attr
          # --load-8bit
          # TODO: Arbitrarily set `max-new-tokens` to 256 to avoid over-generation by llava-7b
  fi
fi