#!/bin/bash

# Execute LLaVA
python -m llava.serve.cli \
	  --model-path liuhaotian/llava-v1.5-7b \
	  --image-file "https://llava-vl.github.io/static/images/view.jpg" \
	  --load-4bit
