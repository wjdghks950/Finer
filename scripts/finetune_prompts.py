# Diversify the input space of LLM for finetuning on FINER dataset
input_clf_style = [
    "What is kind of {concept_placeholder} appear in this image?",
    "Can you identify the {concept_placeholder} shown in this picture?",
    "What's the name of the {concept_placeholder} depicted in this image?",
    "Could you point out what the {concept_placeholder} in this image is?",
    "I'm curious, what is this {concept_placeholder} shown here?",
    "Do you know what the {concept_placeholder} in this photograph is?",
    "What do we call the {concept_placeholder} that's in this image?",
    "Please tell me, what is the {concept_placeholder} here in this image?",
    "I'm trying to figure out the {concept_placeholder} in this image, can you help?",
    "What term would you use for the {concept_placeholder} in this picture?",
    "Could you clarify what the {concept_placeholder} in this picture is called?",
    "How would you classify the {concept_placeholder} in this picture?",
    "What classification would you give to the {concept_placeholder} in this picture?",
    "What type of {concept_placeholder} is displayed here?",
    "Can you define the class of {concept_placeholder} in this photograph?",
    "How would you describe the category of the {concept_placeholder} in this image?",
    "What's your classification for the {concept_placeholder} in this image?",
    "Could you specify the type of {concept_placeholder} shown here in this picture?",
    "In your opinion, what classification does the {concept_placeholder} in this image belong to?",
    "How would you name the {concept_placeholder} appearing in this image?"
]

# Diversify the output space of LLM for finetuning on FINER dataset
output_clf_style = [
    "It is a {concept_placeholder}.",
    "It's called a {concept_placeholder}.",
    "The concept in this image's name is {concept_placeholder}.",
    "This is known as a {concept_placeholder}.",
    "We refer to this as a {concept_placeholder}.",
    "The term for this concept is {concept_placeholder}.",
    "This object is identified as a {concept_placeholder}.",
    "In this context, it's a {concept_placeholder}.",
    "They call this a {concept_placeholder}.",
    "This representation is of a {concept_placeholder}.",
    "One might describe this as a {concept_placeholder}.",
    "This illustrates the concept of a {concept_placeholder}.",
    "Here, we see an example of a {concept_placeholder}.",
    "This depicts a {concept_placeholder}.",
    "In the image, you'll notice a {concept_placeholder}.",
    "The focus here is on the {concept_placeholder}.",
    "You're looking at a {concept_placeholder}.",
    "A {concept_placeholder} is shown in this image."
]

# (i) Attribute generation per image input (ii) Generate the end concept with the attributes generated
attr_gen_input_styles = [
    "What kind of external descriptive attributes do you see from the {concept_placeholder} in this image?",
    "From the {concept_placeholder}, describe the discriminative, physical characteristics you see.",
    "Can you identify and describe the external features visible on the {concept_placeholder} in this picture?",
    "Observe the {concept_placeholder} and detail the unique, physical traits you notice.",
    "What observable qualities can you point out about the {concept_placeholder} in this image?",
    "In this image, what are the notable exterior aspects of the {concept_placeholder}?",
    "Please describe the visible characteristics that stand out for the {concept_placeholder} here.",
    "Examine the {concept_placeholder} and comment on its distinctive physical features.",
    "What external attributes of the {concept_placeholder} in the image catch your attention?",
    "Identify and elaborate on the physical qualities of the {concept_placeholder} seen in this picture.",
    "Regarding the {concept_placeholder}, what discernible features can you describe from this image?",
    "What are the standout external features you notice on the {concept_placeholder} depicted here?",
    "Can you detail the observable physical characteristics of the {concept_placeholder} in this photograph?",
    "In this depiction, what distinct external traits of the {concept_placeholder} can you identify?",
    "Describe the noticeable, physical attributes of the {concept_placeholder} in this visual representation."
]

attr_gen_output_styles = [
    "The {concept_placeholder} in the image exhibits {attribute_placeholder}.",
    "I can see that the {concept_placeholder} has {attribute_placeholder}.",
    "The {concept_placeholder} shown displays {attribute_placeholder}.",
    "It's evident that the {concept_placeholder} possesses {attribute_placeholder}.",
    "One can observe that the {concept_placeholder} features {attribute_placeholder}.",
    "The {concept_placeholder} in this depiction is characterized by {attribute_placeholder}.",
    "Noticeably, the {concept_placeholder} holds {attribute_placeholder}.",
    "The {concept_placeholder} here presents with {attribute_placeholder}.",
    "In this representation, the {concept_placeholder} clearly has {attribute_placeholder}.",
    "It appears that the {concept_placeholder} is endowed with {attribute_placeholder}.",
    "You can discern that the {concept_placeholder} includes {attribute_placeholder}.",
    "This particular {concept_placeholder} is marked by {attribute_placeholder}.",
    "The {concept_placeholder} is visibly equipped with {attribute_placeholder}.",
    "In the image, the {concept_placeholder} evidently has {attribute_placeholder}.",
    "It's clear that the {concept_placeholder} in the picture has {attribute_placeholder}."
]

attr_gen_interm_styles = [
    "Now, what is the specific name of the {concept_placeholder} considering their physical attributes?",
    "Given the set of attributes, tell me the specific name of the {concept_placeholder}.",
    "What is the exact name of the {concept_placeholder} now that you have identified their physical characteristics?",
    "Can you identify the {concept_placeholder} by its physical features and provide its specific name?",
    "What specific name is given to the {concept_placeholder} based on its physical attributes?",
    "Now that you're aware of their physical traits, what is the {concept_placeholder}'s precise name?",
    "Considering their physical characteristics, what is the {concept_placeholder} specifically called?",
    "After noting the physical attributes, could you specify the name of the {concept_placeholder}?",
    "What exact name corresponds to the {concept_placeholder} in light of their physical traits?",
    "Identify the specific name of the {concept_placeholder} given its physical features.",
    "Based on the physical characteristics, what is the exact designation of the {concept_placeholder}?",
    "Can you determine the precise name of the {concept_placeholder} from its physical attributes?",
    "What is the detailed name of the {concept_placeholder} based on their physical description?",
    "With the physical attributes outlined, what is the {concept_placeholder}'s specific nomenclature?",
    "Given the physical characteristics you've noted, what is the specific name of the {concept_placeholder}?",
    "Name the {concept_placeholder} specifically, based on the physical traits you have identified."
]


input_attr_clf_style = "Judging by the {attributes} in the concept, it's a {concept_placeholder}"