from evaluate import load
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from pred import Llama_frisian
from datasets import load_dataset
import matplotlib.pyplot as plt


wer_metric = load("wer")
cer_metric = load("cer")
normalizer = BasicTextNormalizer()

llama = Llama_frisian("slugroom/llama_rjochtwurd")
ds = load_dataset("slugroom/rjochtwurd-dataset")

filtered_data = []

test_data = ds["test"].select(range(2))

def evaluation(example):
    prediction = example["prediction"]
    reference = example["reference"]
    
    if isinstance(prediction, str) and isinstance(reference, str):
        corrected = llama.error_correct(prediction)
        
        wer_pred = wer_metric.compute(predictions=[prediction], references=[reference])
        wer_corr = wer_metric.compute(predictions=[prediction], references=[reference])
        cer_pred = cer_metric.compute(predictions=[corrected], references=[reference])
        cer_corr = cer_metric.compute(predictions=[corrected], references=[reference])
        
        example["corrected"] = corrected
        example["reference"] = reference

        example["wer_corr"] = wer_corr
        example["wer_pred"] = wer_pred

        example["cer_pred"] = cer_pred
        example["cer_corr"] = cer_corr

        return example


test_data = test_data.map(evaluation, desc="Evaluating")

print(test_data[:2])

test_data = test_data.filter(lambda x: x is not None, desc="Filtering out None values")




