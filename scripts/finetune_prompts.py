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

input_attr_clf_style = "Judging by the {attributes} in the concept, it's a {concept_placeholder}"