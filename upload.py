
"""
before running, update your package
>>> pip install --upgrade huggingface_hub

make sure to be logged in
>>> huggingface-cli login
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import pandas as pd
import numpy as np
import os


print("loading data")

parent_dir = "./processed/"
csv_files = os.listdir(parent_dir)
csv_files = [parent_dir + path for path in csv_files]

df = pd.concat([pd.read_csv(f, sep='\t', index_col=False) for f in csv_files], ignore_index=True)

df['string_length'] = df['reference'].apply(len)

# max_length = 40  # Adjust this as needed
max_length = np.inf

# Filter rows where both 'pred' and 'true' are within the max length
filtered_df = df[df['prediction'].str.len().le(max_length) & df['reference'].str.len().le(max_length)]
df = filtered_df

ds = Dataset.from_pandas(df)
ds = ds.remove_columns(["string_length", "__index_level_0__"])


print(ds)

ds = ds.train_test_split(test_size=0.05, shuffle=False)

print("saving dataset")
ds.push_to_hub("slugroom/rjochtwurd-dataset")



print("loading model and tokenizer")

model_id = "llama_finetuned"
model_id = "./results/checkpoint-5"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# to save the model and tokenizer to huggingface hub

print("saving model and tokenizer")
model.push_to_hub("slugroom/llama_rjochtwurd")
tokenizer.push_to_hub("slugroom/llama_rjochtwurd")
