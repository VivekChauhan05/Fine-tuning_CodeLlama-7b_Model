dataset = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1" , split = "train")
# Some other datasets
# dataset = load_dataset("sahil2801/code_instructions_120k" , split = "train")
# dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca" , split = "train")
#dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K",split = "train")
# dataset = load_dataset("WizardLM/WizardLM_evol_instruct_70k", split = "train")
# dataset = load_dataset("mlabonne/CodeLlama-2-20k" , split = "train")
# dataset = load_dataset("VMware/open-instruct-v1-oasst-dolly-hhrlhf", split = "train")


training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length" : 2048
    },
    "datasets": {
        # "use_hf": use_hf,
        "path": dataset
    },
    "verbose": True # a boolean indicating whether to output detailed information during the process.
}

def tokenize_function(examples):
    if "question" in examples and "answer" in examples:
      text = examples["question"][0] + examples["answer"][0]
    elif "input" in examples and "output" in examples:
      text = examples["input"][0] + examples["output"][0]
    elif "instruction" in examples and "response" in examples:
      text = examples["instruction"][0] + examples["response"][0]
    elif "instruction" in examples and "completion" in examples:
      text = examples["instruction"][0] + examples["completion"][0]
    elif "instruction" in examples and "output" in examples:
      text = examples["instruction"][0] + examples["output"][0]
    else:
      text = examples["text"][0]
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
    )

    max_length = min(
        tokenized_inputs["input_ids"].shape[1],
        2048
    )
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=max_length
    )

    return tokenized_inputs


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1,
    drop_last_batch=True
)
total = sum(len(sequence) for sequence in tokenized_dataset)
print(total)

print(tokenized_dataset)



#Splitting the dataset into train,validation,test dataset
total_samples = len(tokenized_dataset)
train_ratio = 0.6  # 60% for training
val_ratio = 0.2    # 20% for validation
test_ratio = 0.2   # 20% for testing

train_size = int(total_samples * train_ratio)
val_size = int(total_samples * val_ratio)
test_size = int(total_samples * test_ratio)

# Define ranges for each split
train_range = range(train_size)
val_range = range(train_size, train_size + val_size)
test_range = range(train_size + val_size, total_samples)

# Create datasets based on the defined ranges
train_dataset = tokenized_dataset.select(train_range)
val_dataset = tokenized_dataset.select(val_range)
test_dataset = tokenized_dataset.select(test_range)

# Print the sizes of each dataset
print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
print("Test dataset size:", len(test_dataset))

print(train_dataset)
print(val_dataset)
print(test_dataset)