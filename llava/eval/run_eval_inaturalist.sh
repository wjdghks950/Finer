#!/bin/bash

echo "task_type_id [\"high_coarse: 0\", \"coarse: 1\", \"fine: 2\"] >> "
read task_type_id

echo "use_prompt [true/false] >> "
read use_prompt

# Only ask for prompt_type_id if use_prompt is True
if [ "$use_prompt" == true ]; then
	echo "prompt_type_id [\"cot_0shot: 0\", \"cot_fewshot: 1\", \"attr_seek: 2\"] >> "
	read prompt_type_id
	python eval_inaturalist.py --task_type_id $task_type_id --prompt_type_id $prompt_type_id --use_prompt
else
	prompt_type_id=0  # Set prompt_type_id to an empty string if use_prompt is not True
	python eval_inaturalist.py --task_type_id $task_type_id
fi
