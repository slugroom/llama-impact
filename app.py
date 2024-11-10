from flask import request, jsonify
import pred

speechToText = pred.Wave2Vec2_frisian
llamaFrisian = pred.Llama_frisian

app = flask.Flask(__name__)

@app.route("/")
def main_page():
    return flask.render_template('index.html')

<<<<<<< HEAD
from flask import request, jsonify
=======
@app.route("/data-send", methods=['POST'])
def file_upload():
    file = request.data
>>>>>>> 8cdce29e6c4bafc39e3956d5fc2ae2200d1f27f5

@app.route("/data-send", methods=['POST'])
def file_upload():
    file = request.files.get("audio_data")
    audio_type = request.form.get("type")
    
    if not file:
        return jsonify({"error": "No file provided"}), 400
    
    audio_data = file.read()

    original_text = speechToText.predict(audio_data)
    llama_corrected_text = llamaFrisian.predict(original_text)

    return jsonify({
        "original": original_text,
        "corrected": llama_corrected_text
    })
