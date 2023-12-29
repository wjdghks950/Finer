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

# TODO: Think out the `cot_fewshot` prompt
cot_fewshot = '''
...
'''

attr_gen = '''
What are useful visual features for distinguishing {concept_placeholder} in a photo? Provide the answer as lists of required and likely attributes. For example, for a school bus you might say:
Required:
- yellow
- black
- wheels
- windows
- bus

Likely:
- school children
- stop sign
- school bus lights

Provide your response in the above format, saying nothing else. If there are no useful visual features, simply write "none". For example, if there are no useful required features, say:

Required:
none
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
PROMPT_DICT['cot_fewshot'] = cot_fewshot.strip()
PROMPT_DICT['attr_gen'] = attr_gen.strip()

def prompt_parser(text, parse_mode='attr_gen'):
    if parse_mode == 'attr_gen':
        pass