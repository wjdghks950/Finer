high_coarse = '''
What is the name of the organism that appears in this image? \
Provide your answer after "Answer: " from one of the four categories: {plants, birds, mammals, insects}.
'''

coarse = '''
What is the name of the {plants/birds/mammals/insects} that appears in this image? \
Provide your answer after "Answer: ".
'''

fine = '''

'''

PROMPT_DICT = {}
PROMPT_DICT['high_coarse'] = high_coarse.strip()
PROMPT_DICT['coarse'] = coarse.strip()
PROMPT_DICT['fine'] = fine.strip()