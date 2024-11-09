import pandas as pd
import torch
import torchaudio
from datasets import load_dataset, Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

test_dataset = pd.read_csv('./cv-corpus-19.0-2024-09-13/fy-NL/test.tsv', sep='\t')

parent_folder = './cv-corpus-19.0-2024-09-13/fy-NL/clips/'
test_dataset["path"] = test_dataset["path"].apply(lambda x: parent_folder + x)


test_dataset = Dataset.from_pandas(test_dataset)

processor = Wav2Vec2Processor.from_pretrained("wietsedv/wav2vec2-large-xlsr-53-frisian")
model = Wav2Vec2ForCTC.from_pretrained("wietsedv/wav2vec2-large-xlsr-53-frisian")

resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"][:2], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset["sentence"][:2])

