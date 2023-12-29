#!/bin/bash

DATA_PATH="val_noplant_64.json"

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
            --model-path liuhaotian/llava-v1.5-7b \
            --data_dir /home/jk100/data/data/inaturalist \
            --data_path $DATA_PATH \
            --image-file None \
            --task_type_id $task_type_id \
            --prompt_type_id $prompt_type_id \
            --use_prompt \
            --use_gold_coarse \
            --load-8bit
  else
    prompt_type_id=0  # Set prompt_type_id to an empty string if use_prompt is not True
          python -m llava.serve.cli \
            --model-path liuhaotian/llava-v1.5-7b \
            --data_dir /home/jk100/data/data/inaturalist \
            --data_path $DATA_PATH \
            --image-file None \
            --task_type_id $task_type_id \
            --use_gold_coarse \
            --load-8bit
  fi

# If it's not for prompting, it's for generating attributes per concept
elif [ "$task_type_id" = 3 ]; then
  echo "Running generating attribute..."
  echo "Use binomial or common_name [binomial/common] ? >> "
  read input_type
	python -m llava.serve.cli \
          --model-path liuhaotian/llava-v1.5-7b \
          --data_dir /home/jk100/data/data/inaturalist \
          --data_path $DATA_PATH \
          --image-file None \
          --task_type_id $task_type_id \
          --input_type $input_type \
          --load-8bit
fi