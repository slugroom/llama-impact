import pandas as pd
import numpy as np
import torch
import torchaudio
from datasets import load_dataset, Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from concurrent.futures import ThreadPoolExecutor
from evaluate import load
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

wer_metric = load("wer")
normalizer = BasicTextNormalizer()

processor = Wav2Vec2Processor.from_pretrained("wietsedv/wav2vec2-large-xlsr-53-frisian")
model = Wav2Vec2ForCTC.from_pretrained("wietsedv/wav2vec2-large-xlsr-53-frisian")
resampler = torchaudio.transforms.Resample(48_000, 16_000)
model.eval()

def speech_file_to_array_fn(batch):
        speech_array, _ = torchaudio.load(batch["path"])
        batch["speech"] = resampler(speech_array).squeeze(0).numpy()
        return batch

def process_dataset_in_parallel(dataset, func, num_workers=4):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        dataset = list(executor.map(func, dataset))
    return Dataset.from_list(dataset)

batch_size = 64
data_chunk_size = batch_size * 16
pass_size = data_chunk_size * 8
pass_count = 0

while True:

    test_dataset = pd.read_csv("./corpus/fy-NL/other.tsv", sep="\t")
    if pass_count * pass_size >= len(test_dataset):
        break
    test_dataset = test_dataset[(pass_count * pass_size):((pass_count + 1) * pass_size)]
    parent_folder = "./corpus/fy-NL/clips/"
    test_dataset["path"] = parent_folder + test_dataset["path"]

    test_dataset = Dataset.from_pandas(test_dataset)
        
    test_dataset = process_dataset_in_parallel(test_dataset, speech_file_to_array_fn, num_workers=8)

    predictions = []
    references = []
    print(len(test_dataset))
    for i in range(0, len(test_dataset), batch_size):
    
        batch = test_dataset[i:i + batch_size]
        inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)

        prediction_array = processor.batch_decode(predicted_ids)
        reference_array = batch["sentence"]

        for j in range(0, len(prediction_array)):
            prediction = prediction_array[j]
            reference = reference_array[j]
            normalized_prediction = normalizer(prediction)
            normalized_reference = normalizer(reference)
            
            wer = wer_metric.compute(predictions=[normalized_prediction], references=[normalized_reference])
            
            if wer >= 0.25:
                predictions.append(prediction)
                references.append(reference)
        
        print("Batch " + str(i // batch_size))
        
        if (i + batch_size) % data_chunk_size == 0:
            cur_file_index = i // data_chunk_size + int(pass_count * data_chunk_size / batch_size)
            results_df = pd.DataFrame({"prediction": predictions, "reference": references})
            results_df.to_csv("./processed/processed_data_" + str(cur_file_index) + ".csv", index=False, sep="\t")
            print("Wrote file " + str(cur_file_index + 1) + " with index i at " + str(i) + " and " + str(len(predictions)) + " predictions and chunk size " + str(data_chunk_size) + ".")
            predictions = []
            references = []
        
    pass_count += 1