# iNaturalist
high_coarse_inaturalist = '''
What is the name of the organism that appears in this image? \
Provide your answer after "Answer:" from one of the following categories: ['Arachnids', 'Mammals', 'Reptiles', 'Animalia', 'Mollusks', 'Plants', 'Amphibians', 'Ray-finned Fishes', 'Birds', 'Insects', 'Fungi'].
'''

coarse_inaturalist = '''
What is the name of the {concept_placeholder} that appears in this image? \
For example, if it's a picture of a bengal tiger, give a coarse-grained label for the image 'Tiger'.
Provide your answer after "Answer:".
'''

fine_inaturalist = '''
What is the name of the {concept_placeholder} that appears in this image? \
For example, if it's a picture of a bengal tiger, give a fine-grained label for the image 'Bengal Tiger' or use its binomial nomenclature 'Panthera tigris tigris'.
Provide your answer after "Answer:".
'''

# FGVC-Aircraft
high_coarse_fgvc_aircraft = '''
What is the name of the object that appears in this image? \
Provide your answer after "Answer:" from one of the following categories: ['Airplane', 'Car', 'Train', 'Bicycle', 'Cell Phone', 'Plants', 'Dogs', 'Birds', 'Trucks'].
'''
 
coarse_fgvc_aircraft = '''
What is the manufacturer of the {concept_placeholder} that appears in this image? \
Provide your answer after "Answer:" from one of the following categories: ['Embraer', 'Lockheed Corporation', 'Douglas Aircraft Company', 'Cirrus Aircraft', 'Airbus', 'Antonov', 'de Havilland', 'Eurofighter', 'Cessna', 'Tupolev', 'Dornier', 'Yakovlev', 'Panavia', 'Robin', 'ATR', 'Beechcraft', 'Dassault Aviation', 'Fairchild', 'McDonnell Douglas', 'Fokker', 'Gulfstream Aerospace', 'Boeing', 'Saab', 'Canadair', 'Lockheed Martin', 'Supermarine', 'Ilyushin', 'British Aerospace', 'Piper', 'Bombardier Aerospace'].
'''

fine_fgvc_aircraft = '''
What is the name of the airplane model made by {concept_placeholder} that appears in this image? \
For example, if it's a picture of a Boeing 787 Dreamliner, give a fine-grained label for the image 'Boeing 787 Dreamliner'.
Provide your answer after "Answer:".
'''

# CUB-200-2011
high_coarse_cub_200_2011 = '''
What is the name of the organism that appears in this image? \
Provide your answer after "Answer:" from one of the following categories: ['Arachnids', 'Mammals', 'Reptiles', 'Animalia', 'Mollusks', 'Plants', 'Amphibians', 'Ray-finned Fishes', 'Birds', 'Insects', 'Fungi'].
'''
 
coarse_cub_200_2011 = '''
What is the name of the {concept_placeholder} that appears in this image? \
For example, if it's a picture of a Owl Parrot, give a coarse-grained label for the image 'Parrot'.
Provide your answer after "Answer:".
'''

fine_cub_200_2011 = '''
What is the name of the {concept_placeholder} that appears in this image? \
For example, if it's a picture of a Owl Parrot, give a fine-grained label for the image 'Owl Parrot'.
Provide your answer after "Answer:".
'''

# Stanford Dogs
high_coarse_stanford_dogs = '''
What is the name of the organism that appears in this image? \
Provide your answer after "Answer:" from one of the following categories: ['Arachnids', 'Dogs', 'Reptiles', 'Mollusks', 'Plants', 'Amphibians', 'Ray-finned Fishes', 'Birds', 'Insects', 'Fungi'].
'''
 
coarse_stanford_dogs = '''
What is the name of the {concept_placeholder} that appears in this image? \
For example, if it's a picture of a Golden Retriever, give a coarse-grained label for the image 'Retriever'.
Provide your answer after "Answer:".
'''

fine_stanford_dogs = '''
What is the name of the {concept_placeholder} that appears in this image? \
For example, if it's a picture of a Golden Retriever, give a coarse-grained label for the image 'Golden Retriever'.
Provide your answer after "Answer:".
'''

# Stanford Cars
high_coarse_stanford_cars = '''
What is the name of the object that appears in this image? \
Provide your answer after "Answer:" from one of the following categories: ['Airplane', 'Car', 'Train', 'Bicycle', 'Cell Phone', 'Plants', 'Dogs', 'Birds', 'Trucks'].
'''
 
coarse_stanford_cars = '''
What is the name of the {concept_placeholder} that appears in this image? \
Provide your answer after "Answer:" from one of the following categories: ['Sedan', 'SUV', 'Coupe', 'Convertible', 'Pickup', 'Hatchback', 'Van']
'''

fine_stanford_cars = '''
What is the name of the {concept_placeholder} that appears in this image? \
For example, if it's a picture of a 2006 Honda Civic LX Coupe, give a fine-grained label for the image '2006 Honda Civic LX Coupe'.
Provide your answer after "Answer:".
'''

# NAbirds
high_coarse_nabirds = '''
What is the name of the organism that appears in this image? \
Provide your answer after "Answer:" from one of the following categories: ['Arachnids', 'Mammals', 'Reptiles', 'Animalia', 'Mollusks', 'Plants', 'Amphibians', 'Ray-finned Fishes', 'Birds', 'Insects', 'Fungi'].
'''
 
coarse_nabirds = '''
What is the name of the {concept_placeholder} that appears in this image? \
For example, if it's a picture of a Owl Parrot, give a coarse-grained label for the image 'Parrot'.
Provide your answer after "Answer:".
'''

fine_nabirds = '''
What is the name of the {concept_placeholder} that appears in this image? \
For example, if it's a picture of a Owl Parrot, give a fine-grained label for the image 'Owl Parrot'.
Provide your answer after "Answer:".
'''

# Multi-turn dialogue for attribute extraction & concept classification
attr_seek = '''
What kind of external descriptive attributes do you see from the {concept_placeholder} in this image? \
Provide the set of detailed physical attributes after "Attributes:".
'''

attr_seek_coarse = '''
Now, tell me what kind of {concept_placeholder} it is. \
For example, if it's a picture of a bengal tiger, give a coarse-grained label for the image 'Tiger'.
Provide your answer after "Answer:".
'''

attr_seek_fine = '''
Now, tell me what kind of {concept_placeholder} it is. \
For example, if it's a picture of a bengal tiger, give a fine-grained label for the image 'Bengal Tiger' or use its binomial nomenclature 'Panthera tigris tigris'.
Provide your answer after "Answer:".
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

'Required' attributes are a set of external, physical attributes that allows a human to distinguish it from other similar looking concepts.
'Likely' attributes are a set of attributes that may or may not be visible or are not one of the most discriminative features of the concept.
In the required (Required:) set, do not include relative, non-visual attributes like size or weight, only the external, visually distinguishable attributes. \
Provide your response in the above format, saying nothing else. If there are no useful visual features, simply write "none". \
'''

# Prompt to extract descriptive, physical attributes that pertain to the organism's Wikipedia page
attr_gen_wiki ='''
What are useful visual, external features for distinguishing {concept_placeholder} in a photo? \
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

'Required' attributes are a set of external, physical attributes that allows a human to distinguish it from other similar looking concepts.
'Likely' attributes are a set of attributes that may or may not be visible or are not one of the most discriminative features of the concept.
In the required (Required:) set, do not include relative, non-visual attributes like size or weight, only the external, visually distinguishable attributes. \
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

'Required' attributes are a set of external, physical attributes that allows a human to distinguish it from other similar looking concepts.
'Likely' attributes are a set of attributes that may or may not be visible or are not one of the most discriminative features of the concept.
In the required (Required:) set, do not include relative, non-visual attributes like size or weight, only the external, visually distinguishable attributes. \
Provide your response in the above format, saying nothing else. If there are no useful visual features, simply write "none".
'''

knowledge_probe_inaturalist = '''
Can you guess the specific name (specific epithet) of an organism in the following taxonomic category given its physical attributes?
Provide your answer after "Specific Epithet:".

Physical Attributes: {attribute_placeholder}

Supercategory: {supercategory_placeholder}
Kingdom: {kingdom_placeholder}
Phylum: {phylum_placeholder}
Class: {class_placeholder}
Order: {order_placeholder}
Family: {family_placeholder}
Genus: {genus_placeholder}
Specific Epithet:
'''

knowledge_probe_fgvc_aircraft = '''
Can you guess the specific name (specific type) of an Airplane in the following taxonomic category given its physical attributes?
Provide your answer after "Specific Airplane:".

Physical Attributes: {attribute_placeholder}

Supercategory: {supercategory_placeholder}
Coarse-grained Category: {coarse_placeholder}
Specific Airplane:
'''

knowledge_probe_stanford_dogs = '''
Can you guess the specific name (specific type) of a Dog in the following taxonomic category given its physical attributes?
Provide your answer after "Specific Dog:".

Physical Attributes: {attribute_placeholder}

Supercategory: {supercategory_placeholder}
Coarse-grained Category: {coarse_placeholder}
Specific Dog:
'''

PROMPT_DICT = {}
PROMPT_DICT['high_coarse_inaturalist'] = high_coarse_inaturalist.strip()
PROMPT_DICT['coarse_inaturalist'] = coarse_inaturalist.strip()
PROMPT_DICT['fine_inaturalist'] = fine_inaturalist.strip()

PROMPT_DICT['high_coarse_fgvc_aircraft'] = high_coarse_fgvc_aircraft.strip()
PROMPT_DICT['coarse_fgvc_aircraft'] = coarse_fgvc_aircraft.strip()
PROMPT_DICT['fine_fgvc_aircraft'] = fine_fgvc_aircraft.strip()

PROMPT_DICT['high_coarse_cub_200_2011'] = high_coarse_cub_200_2011.strip()
PROMPT_DICT['coarse_cub_200_2011'] = coarse_cub_200_2011.strip()
PROMPT_DICT['fine_cub_200_2011'] = fine_cub_200_2011.strip()

PROMPT_DICT['high_coarse_stanford_dogs'] = high_coarse_stanford_dogs.strip()
PROMPT_DICT['coarse_stanford_dogs'] = coarse_stanford_dogs.strip()
PROMPT_DICT['fine_stanford_dogs'] = fine_stanford_dogs.strip()

PROMPT_DICT['high_coarse_stanford_cars'] = high_coarse_stanford_cars.strip()
PROMPT_DICT['coarse_stanford_cars'] = coarse_stanford_cars.strip()
PROMPT_DICT['fine_stanford_cars'] = fine_stanford_cars.strip()

PROMPT_DICT['high_coarse_nabirds'] = high_coarse_nabirds.strip()
PROMPT_DICT['coarse_nabirds'] = coarse_nabirds.strip()
PROMPT_DICT['fine_nabirds'] = fine_nabirds.strip()

PROMPT_DICT['attr_seek'] = attr_seek.strip()
PROMPT_DICT['attr_seek_coarse'] = attr_seek_coarse.strip()
PROMPT_DICT['attr_seek_fine'] = attr_seek_fine.strip()
PROMPT_DICT['cot_0shot_coarse'] = cot_0shot_coarse.strip()
PROMPT_DICT['cot_0shot_fine'] = cot_0shot_fine.strip()
PROMPT_DICT['attr_gen'] = attr_gen.strip()
PROMPT_DICT['attr_gen_image'] = attr_gen_image.strip()
PROMPT_DICT['attr_gen_wiki'] = attr_gen_wiki.strip()

PROMPT_DICT['knowledge_probe_inaturalist'] = knowledge_probe_inaturalist.strip()
PROMPT_DICT['knowledge_probe_fgvc_aircraft'] = knowledge_probe_fgvc_aircraft.strip()
PROMPT_DICT['knowledge_probe_stanford_dogs'] = knowledge_probe_stanford_dogs.strip()
