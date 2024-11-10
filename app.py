import flask
import pred

speechToText = pred.Wave2Vec2_frisian
llamaFrisian = pred.Llama_frisian

app = flask.Flask(__name__)

@app.route("/")
def main_page():
    return flask.render_template('index.html')

@app.rout("/data-send", methods=['POST'])
def file_upload():
    file = request.data

    original_text        = speechToText.predict(file)
    llama_corrected_text = llamaFrisian.predict(original_text)

    return {
        "original": original_text,
        "corrected": llama_corrected_text,
    }
