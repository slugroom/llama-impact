from evaluate import load
from pred import Llama_frisian
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import os


wer_metric = load("wer")
cer_metric = load("cer")

llama = Llama_frisian("slugroom/llama_rjochtwurd")


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



def save_predictions(max_samples=None):
    ds = load_dataset("slugroom/rjochtwurd-dataset")

    if max_samples is not None and isinstance(max_samples, int):
        test_data = ds["test"].select(range(max_samples))
    else:
        test_data = ds["test"]

    test_data = test_data.map(evaluation, desc="Evaluating")

    test_data = test_data.filter(lambda x: x is not None, desc="Filtering out None values")


    df = test_data.to_pandas()

    df.to_csv("evaluation.csv")

    return df


def load_predictions():
    df = pd.read_csv("evaluation.csv")
    return df


if __name__ == "__main__":
    if not os.path.exists("evaluation.csv"):
        df = save_predictions()
    else:
        df = load_predictions()

    print(df.head())


    # statistics

    stats = df.describe()
    stats.to_csv("statistics.csv")

    print(stats)

