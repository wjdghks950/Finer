#!/bin/bash

echo "task_type_id [\"high_coarse: 0\", \"coarse: 1\", \"fine: 2\"] >> "
read task_type_id

echo "prompt_type_id [\"cot_0shot: 0\", \"cot_fewshot: 1\", \"attr_seek: 2\"] >> "
read prompt_type_id

python eval_inaturalist.py --task_type_id $task_type_id --prompt_type_id $prompt_type_id
