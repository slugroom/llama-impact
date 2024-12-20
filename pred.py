from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch
from io import BytesIO

torch.backends.cuda.matmul.allow_tf32 = True

class Llama_frisian:
    def __init__(self, model_id="slugroom/llama_rjochtwurd"):
        # model_id = "meta-llama/Llama-3.2-1B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        # self.model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu") # for Josh

    def error_correct(self, txt):

        in_txt = f"The following is a Frisian audio transcription, some parts of the transcription may be incorrect. Correct the transcription by making it grammatically and phonetically accurate.\n ### Transcription: {txt} \n ### Corrected:"
        model_inputs = self.tokenizer([in_txt], return_tensors="pt").to("cpu")

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=50, num_beams=4, do_sample=True, eos_token_id=self.tokenizer.eos_token_id)

        # remove prompt from generated text
        corrected = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

        try:
            corrected = corrected.split(" ### Corrected:\xa0")[1].split("  ")[0]
        except IndexError:
            corrected = corrected
        finally:
            return corrected


class Wave2Vec2_frisian:
    def __init__(self, model_id="wietsedv/wav2vec2-large-xlsr-53-frisian"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.resampler = torchaudio.transforms.Resample(48_000, 16_000)

    def preprocess_audio_data(self, audio_data):
        audio_stream = BytesIO(audio_data)
        speech_array, _ = torchaudio.load(audio_stream, format="mp3")

        speech = self.resampler(speech_array).squeeze(0).numpy()
        return speech

    def speech_to_text(self, speech):
        inputs = self.processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)
        # inputs = {k: v.to("cpu") for k, v in inputs.items()}  # Ensure tensors are on CPU

        with torch.no_grad():
            logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)

        return self.processor.batch_decode(predicted_ids)[0]

    def predict(self, audio_data):
        speech = self.preprocess_audio_data(audio_data)
        return self.speech_to_text(speech)


# llama = Llama_frisian()
# corrected = llama.error_correct("de snoeskjirne nei fleurich blinkend yn it middeislacht")
# print(corrected)

# wav2vec2 = Wave2Vec2_frisian()
# text = wav2vec2.predict("corpus/fy-NL/clips/common_voice_fy-NL_40989988.mp3")
# print(text)

