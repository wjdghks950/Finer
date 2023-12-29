#!/bin/bash

# Execute LLaVA
python -m llava.serve.cli \
	  --model-path liuhaotian/llava-v1.5-13b \
	  --image-file "/shared/nas/data/m1/jk100/data/inaturalist/val/00927_Animalia_Arthropoda_Insecta_Lepidoptera_Crambidae_Pyrausta_acrionalis/05fa4316-56b7-4d80-8d4c-4bf250895ad3.jpg" \
	  --task_type_id 0 \
	  --load-4bit