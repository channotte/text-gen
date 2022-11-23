from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

HUGGING_FACE_PSEUDO = 'channotte'
HUGGING_FACE_DS_NAME = 'Georges_Sand'

# Transform original dataset to dataset of token
context_length = 100
dataset = load_dataset(HUGGING_FACE_PSEUDO+"/"+ HUGGING_FACE_DS_NAME)


#pretrained or custom tokenizer

pretrained_tokenizer = AutoTokenizer.from_pretrained("benjamin/gpt2-wechsel-french")

def tokenize(element):
    outputs = pretrained_tokenizer(
        element["review"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=False,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = dataset.map(
    tokenize, batched=True, remove_columns=dataset["train"].column_names
)

print(tokenized_datasets)