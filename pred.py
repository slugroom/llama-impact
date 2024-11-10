from transformers import AutoModelForCausalLM, AutoTokenizer

class Llama_frisian:
    def __init__(self, model_id="./llama3"):
        # model_id = "meta-llama/Llama-3.2-1B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def error_correct(self, txt):
        in_txt = f"### Prediction: {txt} ### Corrected: "
        model_inputs = self.tokenizer([in_txt], return_tensors="pt")
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=50, num_beams=4, do_sample=True)

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]



llama = Llama_frisian("llama_finetuned")

corrected = llama.error_correct("dy soarch is foarn mer jochte op it rela te follen fan passin")

print(corrected)