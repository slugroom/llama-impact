from evaluate import load
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from pred import Llama_frisian


wer_metric = load("wer")
cer_metric = load("cer")
normalizer = BasicTextNormalizer()

llama = Llama_frisian("slugroom/llama_rjochtwurd")

index = 0
filtered_data = []
while True:
    file_path = f"./testing_set/processed_data_" + str(index) + ".csv"
    
    if not os.path.exists(file_path):
        break

    
    print("File " + str(index))
    dataset = pd.read_csv(file_path, sep="\t")
    
    for _, row in dataset.iterrows():
        prediction = row["prediction"]
        reference = row["reference"]
        wer_pred = row["wer"]
        
        if isinstance(prediction, str) and isinstance(reference, str):
            corrected = llama.error_correct(prediction)
            normalized_correction = normalizer(corrected)
            normalized_reference = normalizer(reference)
            
            wer_corr = wer_metric.compute(predictions=[normalized_correction], references=[normalized_reference])
            cer_pred = cer_metric.compute(predictions=[normalized_correction], references=[normalized_reference])
            cer_corr = cer_metric.compute(predictions=[normalized_correction], references=[normalized_reference])
            
            
            filtered_data.append({
                    "prediction": prediction,
                    "corrected": corrected,
                    "reference": reference,
                    "wer_pred": wer_pred,
                    "wer_corr": wer_corr,
                    "cer_pred": cer_corr,
                    "cer_corr": cer_corr
                })
    
    index += 1


new_file_path = f"./measurements.csv"
filtered_df = pd.DataFrame(filtered_data)

filtered_df.to_csv(new_file_path, sep="\t", index=False)
