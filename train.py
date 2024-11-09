import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import os
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import matplotlib.pyplot as plt

model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", load_in_8bit=True, device_map="auto")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

parent_dir = "./processed/"
csv_files = os.listdir(parent_dir)
csv_files = [parent_dir + path for path in csv_files]

df = pd.concat([pd.read_csv(f, sep='\t', index_col=False) for f in csv_files], ignore_index=True)

df['string_length'] = df['reference'].apply(len)

# Plot histogram
plt.hist(df['string_length'], bins=10, edgecolor='black')
plt.xlabel('Length of Strings')
plt.ylabel('Frequency')
plt.title('Histogram of String Lengths in Column')
plt.savefig('hist.png')

max_length = 40  # Adjust this as needed

# Filter rows where both 'pred' and 'true' are within the max length
filtered_df = df[df['prediction'].str.len().le(max_length) & df['reference'].str.len().le(max_length)]
df = filtered_df

ds = Dataset.from_pandas(df)
ds = ds.remove_columns(["string_length", "__index_level_0__"])


ds = ds.train_test_split(test_size=0.2)

print(ds)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prediction'])):
        text = f"### Prediction: {example['prediction'][i]}\n ### Corrected: {example['reference'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Corrected:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


# PEFT config
lora_alpha = 16
lora_dropout = 0.1
lora_r = 32  # 64
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
    modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"],
)

training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1, # increase if you can
    per_device_eval_batch_size=1, 
    gradient_accumulation_steps=8, # increase when memeory issue
    optim="adamw_torch",
    save_steps=5,
    logging_steps=5,
    learning_rate=2e-4,
    fp16=False, # Change this to True for efficiency
    max_grad_norm=0.3,
    max_steps=100,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,  # gradient checkpointing
    #report_to="wandb",
)

trainer = SFTTrainer(
    model,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    args=training_arguments,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
