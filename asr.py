import pandas as pd
import numpy as np
import torch
import torchaudio
from datasets import load_dataset, Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from concurrent.futures import ThreadPoolExecutor

test_dataset = pd.read_csv("./corpus/fy-NL/test.tsv", sep="\t")
parent_folder = "./corpus/fy-NL/clips/"
test_dataset["path"] = parent_folder + test_dataset["path"]

test_dataset = Dataset.from_pandas(test_dataset)

processor = Wav2Vec2Processor.from_pretrained("wietsedv/wav2vec2-large-xlsr-53-frisian", torch_dtype=torch.bfloat16)
model = Wav2Vec2ForCTC.from_pretrained("wietsedv/wav2vec2-large-xlsr-53-frisian", torch_dtype=torch.bfloat16)
model.eval()

resampler = torchaudio.transforms.Resample(48_000, 16_000)

def speech_file_to_array_fn(batch):
    speech_array, _ = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze(0).numpy().astype(np.float32)
    return batch

def process_dataset_in_parallel(dataset, func, num_workers=4):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        dataset = list(executor.map(func, dataset))
    return Dataset.from_list(dataset)
    
test_dataset = process_dataset_in_parallel(test_dataset, speech_file_to_array_fn, num_workers=8)

batch_size = 16
data_chunk_size = batch_size * 2
predictions = []
references = []

for i in range(0, 256, batch_size):
    try:
        batch = test_dataset[i:i + batch_size]
        inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True).to(torch.bfloat16)

        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predictions.extend(processor.batch_decode(predicted_ids))
        references.extend(batch["sentence"])
        
        if (i - batch_size) % data_chunk_size == 0:
            cur_file_index = i // data_chunk_size
            results_df = pd.DataFrame({"prediction": predictions, "reference": references})
            results_df.to_csv("./processed/processed_data_" + str(cur_file_index) + ".csv", index=False, sep="\t")
            
            predictions = []
            references = []
    except Exception as e:
        print(e)