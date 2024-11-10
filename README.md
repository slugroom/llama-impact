# Team name: Reindeer

> When winter approached the Llama model decided to wear a coat and put on horns

<img src="./media/reindeer.jpeg" style="width: 400px;">


Team members: 

- Efe
- Josh
- Saker
- Sigurdur Haukur

*Names sorted alphabetically*


---

## Overview

This is the project page for the [Llama Impact Hackathon](https://lablab.ai/event/llama-impact-hackathon/)


The live demo of the project can be found [here](https://llama-impact.sigurdurhaukur.com/). Unfortunately, we were not able to run inference on the server due to computational limitation. However, you can run the web app locally by following the instructions below.


### Summary

Frisian is a low-resource language spoken by around 400,000 thousand people in the Netherlands. The goal of this project is to build a speach-to-text error correction model for Frisian using the Common Voice Dataset and the Llama3.2 3B model.

We used ran a speech-to-text model for Frisian on the validation set of the Frisian Common Voice Dataset and calculated the word error rate (WER) and character error rate (CER) to evaluate the model. Then we fine-tuned the Llama3.2 3B model to correct the errors in the transcription. We also evaluated the model using the WER and CER metrics.

[5 rows x 9 columns]
       Unnamed: 0         wer    wer_corr    wer_pred    cer_pred    cer_corr
count  100.000000  100.000000  100.000000  100.000000  100.000000  100.000000
mean    49.500000    0.461790    0.795456    0.603064    0.205363    0.408049
std     29.011492    0.137044    0.736767    0.192863    0.080553    0.323054
min      0.000000    0.250000    0.000000    0.250000    0.064516    0.000000
25%     24.750000    0.375000    0.351190    0.500000    0.139134    0.181446
50%     49.500000    0.444444    0.563492    0.577381    0.198361    0.365642
75%     74.250000    0.509615    1.000000    0.666667    0.255319    0.564381
max     99.000000    0.800000    3.500000    1.250000    0.523810    1.692308

## Installation

To run the web app, you need to install the required packages. As well as torch and torchaudio

```sh
pip install -r requirements.txt
```

Then you can run the web app with the following command:
```sh
flask run
```

data: https://commonvoice.mozilla.org/en/datasets


## Contact Information

If you have any questions or inquiries, please reach out to us at:

- Sigurdur Haukur: [linkedin](https://www.linkedin.com/in/sigurdur-haukur-birgisson)
