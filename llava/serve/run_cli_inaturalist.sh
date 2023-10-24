#!/bin/bash

# Execute LLaVA
python -m llava.serve.cli \
	  --model-path liuhaotian/llava-v1.5-7b \
	  --data_dir /home/jk100/data/data/inaturalist \
	  --data_path val_noplant_64.json \
	  --dialogue-mode single \
	  --image-file None \
	  --load-8bit
