high_coarse = '''
What is the name of the organism that appears in this image? \
Provide your answer after "Answer:" from one of the four categories: {birds, mammals, insects, reptiles, amphibians}.
'''

coarse = '''
What is the name of the {concept_placeholder} that appears in this image? \
Provide your answer after "Answer:".
'''

fine = '''
What is the name of the {concept_placeholder} that appears in this image? \
Provide your answer after "Answer:" and make sure to follow the binomial nomenclature format (e.g., genus-species).
'''

# Multi-turn dialogue for attribute extraction & concept classification
attr_seek = '''
What kind of physical attributes do you see in the {concept_placeholder}? \
Provide the set of detailed physical attributes after "Attributes:".
'''

attr_seek_coarse = '''
Now, tell me what kind of {concept_placeholder} it is. \
Provide your answer after "Answer:".
'''

attr_seek_fine = '''
Now, tell me what kind of {concept_placeholder} it is. \
Provide your answer after "Answer:" and make sure to follow the binomial nomenclature format (e.g., genus-species).
'''

cot_0shot_coarse = '''
What is the name of the {concept_placeholder} that appears in this image? \
Provide your answer after "Answer:".
Let's think step by step.
'''

cot_0shot_fine = '''
What is the name of the {concept_placeholder} that appears in this image? \
Provide your answer after "Answer:" and make sure to follow the binomial nomenclature format (e.g., genus-species).
Let's think step by step. 
'''

# TODO: Ansel's original prompt
# attr_gen = '''
# What are useful visual features for distinguishing {concept_placeholder} in a photo? Provide the answer as lists of required and likely attributes. For example, for a school bus you might say:
# Required:
# - yellow
# - black
# - wheels
# - windows
# - bus

# Likely:
# - school children
# - stop sign
# - school bus lights

# Provide your response in the above format, saying nothing else. If there are no useful visual features, simply write "none". For example, if there are no useful required features, say:

# Required:
# none
# '''


# Revised version of prompt
attr_gen = '''
What are useful visual features for distinguishing {concept_placeholder} in a photo? \
Provide the answer as lists of required and likely attributes. For example, for a bengal tiger (Felis Tigris) you might say:

Required:
- yellow to light orange coat
- dark brown to black stripes
- black rings on the tail
- inner legs and belly are white
- 21 to 29 stripes

Likely:
- lives in mangrove, wooded habitat
- amber, yellow eyes
- large, padded paws
- long tail
- stout teeth

In the required (Required:) set, do not include relative attributes like size or weight. \
Provide your response in the above format, saying nothing else. If there are no useful visual features, simply write "none". \
'''

# Prompt to extract descriptive, physical attributes that pertain to the organism's Wikipedia page
attr_gen_wiki ='''
What are useful visual features for distinguishing {concept_placeholder} in a photo? \
Given an input document (Document:) that may talk about {concept_placeholder}, provide the answer as lists of required and likely attributes. \
For example, for a bengal tiger (Felis Tigris) you might say:

Required:
- yellow to light orange coat
- dark brown to black stripes
- black rings on the tail
- inner legs and belly are white
- 21 to 29 stripes

Likely:
- lives in mangrove, wooded habitat
- amber, yellow eyes
- large, padded paws
- long tail
- stout teeth

In the required (Required:) set, do not include relative, non-visual attributes like size or weight. \
If no document is given, generate from what you already know about {concept_placeholder}.
Provide your response in the above format, saying nothing else. If there are no useful visual features, simply write "none".
'''


attr_gen_image = '''
What are useful visual features for distinguishing the {concept_placeholder} in the photo? \
Provide the answer as lists of required and likely attributes. For example, for a bengal tiger (Felis Tigris) you might say:

Required:
- yellow to light orange coat
- dark brown to black stripes
- black rings on the tail
- inner legs and belly are white
- 21 to 29 stripes

Likely:
- lives in mangrove, wooded habitat
- amber, yellow eyes
- large, padded paws
- long tail
- stout teeth

In the required (Required:) set, do not include relative attributes like size or weight. \
Provide your response in the above format, saying nothing else. If there are no useful visual features, simply write "none".
'''

PROMPT_DICT = {}
PROMPT_DICT['high_coarse'] = high_coarse.strip()
PROMPT_DICT['coarse'] = coarse.strip()
PROMPT_DICT['fine'] = fine.strip()
PROMPT_DICT['attr_seek'] = attr_seek.strip()
PROMPT_DICT['attr_seek_coarse'] = attr_seek_coarse.strip()
PROMPT_DICT['attr_seek_fine'] = attr_seek_fine.strip()
PROMPT_DICT['cot_0shot_coarse'] = cot_0shot_coarse.strip()
PROMPT_DICT['cot_0shot_fine'] = cot_0shot_fine.strip()
PROMPT_DICT['attr_gen'] = attr_gen.strip()
PROMPT_DICT['attr_gen_image'] = attr_gen_image.strip()
PROMPT_DICT['attr_gen_wiki'] = attr_gen_wiki.strip()
