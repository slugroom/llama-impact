from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gradio as gr

from peft import PeftModel

lora_wheights = "slugroom/llama_rjochtwurd"
base_model_id = "meta-llama/Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(model, lora_wheights, torch_dtype=torch.float16, quantization_config=BitsAndBytesConfig(load_in_8bit=True)).to("cpu")

asr = pipeline("automatic-speech-recognition", "wietsedv/wav2vec2-large-xlsr-53-frisian", device="cpu")

def speech_to_text(speech):
    text = asr(speech)["text"]
    return text

def correct_errors(text):
    in_txt = f"The following is a Frisian audio transcription, some parts of the transcription may be incorrect. Correct the transcription by making it grammatically and phonetically accurate.\n ### Transcription: {text} \n ### Corrected:"
    model_inputs = tokenizer([in_txt], return_tensors="pt")

    generated_ids = model.generate(**model_inputs, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=0.5, top_k=50)

    # remove prompt from generated text
    generated_ids = generated_ids[:, model_inputs["input_ids"].shape[-1]:]

    corrected = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(corrected)
    if "###" in corrected:
        corrected = corrected.split("###")[0]


    return corrected


demo = gr.Blocks()

with demo:
    gr.HTML("<h1>Frisian Speech-to-Text Error Correction</h1>")
    gr.HTML("<p>Upload an audio file in Frisian and press 'Transcribe Audio' to transcribe the audio. The transcription will appear in the text box below. You can also type Frisian text in the text box to be corrected and press 'Correct Errors' to correct the text.</p>")
    gr.HTML("<p>Note that correcting the errors may take up to a minute.</p>")

    audio_file = gr.Audio(type="filepath", label="Upload Audio File")
    gr.Examples([["./samples/1.mp3"], ["./samples/2.mp3"], ["./samples/3.mp3"], ["./samples/4.mp3"]], audio_file, label="Example Audio Files")
    text = gr.Textbox(placeholder="Transcription will appear here... Additionally type Frisian here to be corrected", label="Transcription")
    gr.Examples([["ik wit net wêrt ûnder skoraaf"],["hiy it is mei jammer"], ["ast myn frou de earenijen hinne e litte wol is it hare saak"], ["dêr wiene hiel wat sasjalisten yn fryslân"]], text, label="Examples of Frisian Transcriptions")
    audio_file.upload(fn=speech_to_text, inputs=audio_file, outputs=text)

    corrected_text = gr.Textbox(interactive=False, label="Corrected Transcription")

    with gr.Row():
        b1 = gr.Button("Transcribe Audio")
        b1.click(speech_to_text, inputs=audio_file, outputs=text)

        b2 = gr.Button("Correct Errors")
        b2.click(correct_errors, inputs=text, outputs=corrected_text)


demo.launch(server_port=5000)
