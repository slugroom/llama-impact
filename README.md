# Frisian Speech-to-Text Error Correction (a.k.a RjochtWurd)

By: Efe, Josh, Saker, Sigurdur Haukur

*Names sorted alphabetically*

## Overview

This is the project page for the [Llama Impact Hackathon](https://lablab.ai/event/llama-impact-hackathon/)


The live demo of the project can be found [here](https://llama-impact.sigurdurhaukur.com/).

### Summary

Frisian is a low-resource language spoken by around 400,000 thousand people in the Netherlands. The goal of this project is to build a speach-to-text error correction model for Frisian using the [Common Voice Dataset](https://commonvoice.mozilla.org/en/datasets) and the Llama3.2 3B model.

We ran wav2vec2-large-xlsr-53-frisian by wietsedv a speech-to-text model for Frisian (available [here](wietsedv/wav2vec2-large-xlsr-53-frisian) on the validation set of the Frisian Common Voice Dataset and calculated the normalized word error rate (WER) and normalized character error rate (CER) to evaluate the model. Then we fine-tuned  Llama3.2 3B  to correct the errors in the transcription. We also evaluated the model using the WER and CER metrics. See table below for the results, ASR refers to the original transcription and Llama refers to the corrected transcription.

| Statistic | WER (ASR)  | WER (Llama)  | CER (ASR)  | CER (LLama) |
|-----------|-----------|-----------|-----------|-----------|
| count     | 100.00000 | 100.00000 | 100.00000 | 100.00000 |
| mean      |  0.795456 | 0.603064  | 0.205363  | 0.408049  |
| std       | 0.736767  | 0.192863  | 0.080553  | 0.323054  |
| min       | 0.000000  | 0.250000  | 0.064516  | 0.000000  |
| 25%       | 0.351190  | 0.500000  | 0.139134  | 0.181446  |
| 50%       |  0.563492 | 0.577381  | 0.198361  | 0.365642  |
| 75%       | 1.000000  | 0.666667  | 0.255319  | 0.564381  |
| max       | 3.500000  | 1.250000  | 0.523810  | 1.692308  |

The model was able to correct the errors in the transcription, but the WER and CER metrics were higher than expected. This could be due to the low-resource nature of the Frisian language and the lack of training data. The high standard deviation of the WER and CER metrics also indicates that the model is not consistent in correcting the errors in the transcription. In fact, when looking at the minimum and maximum values of the WER and CER metrics, we can see that the model was able to correct some transcriptions perfectly, while others were corrected poorly. This suggests that the model is not robust and may not be suitable for real-world applications.

Despite the limitations of the model, we believe that it has the potential to be improved with more training data and fine-tuning. We also believe that the model could be useful for researchers and developers working on speech-to-text error correction for low-resource languages.


## Evaluation

Plotting the WER and CER metrics for the wav2vec2-large-xlsr-53-frisian by wietsedv model and the fine-tuned LLama model on a bar chart, with the error bars representing the standard deviation of the metrics, we can see that the post-processing via the fine-tuned Llama model decreases the accuracy of the transcription on average. However, the error bars are quite large, and in some cases the fine-tuned Llama model increases the accuracy of the transcription (see the figure below).





## Installation

To run the web app, you need to install the required packages. As well as torch and torchaudio

```sh
pip install -r requirements.txt
```

### Flask Frontend

Then you can run the web app with the following command:

```sh
flask run
```

Afterwards, you can access the web app by navigating to `http://localhost:5000/` in your web browser.

### Gradio Frontend

To run the Gradio frontend, you can run the following command:

```sh
python3 demo.py
```

Afterwards, you can access the web app by navigating to `http://localhost:7860/` in your web browser.


## Contact Information

If you have any questions or inquiries, please reach out to us at:

- Sigurdur Haukur: [linkedin](https://www.linkedin.com/in/sigurdur-haukur-birgisson)
