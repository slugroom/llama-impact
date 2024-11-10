from evaluate import load
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer



class Llama_frisian:
    def __init__(self, model_id="./llama3"):
        # model_id = "meta-llama/Llama-3.2-1B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def error_correct(self, txt):

        in_txt = f"The following is a Frisian audio transcription, some parts of the transcription may be incorrect. Correct the transcription by making it grammatically and phonetically accurate.\n ### Transcription: {txt} \n ### Corrected: "
        model_inputs = self.tokenizer([in_txt], return_tensors="pt")
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=50, num_beams=4, do_sample=True, eos_token_id=self.tokenizer.eos_token_id)

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


index = 0
llama = Llama_frisian("llama_finetuned")

wer_metric = load("wer")
cer_metric = load("cer")
normalizer = BasicTextNormalizer()

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
