import flask
import threading
import uuid
import pred
from werkzeug.middleware.proxy_fix import ProxyFix 


app = flask.Flask(__name__)
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

speechToText = pred.Wave2Vec2_frisian()
llamaFrisian = pred.Llama_frisian()
tasks = {}

@app.route("/")
def main_page():
    return flask.render_template('index.html')

from flask import request, jsonify

def run_llama_prediction(task_id, audio_data):
    # try:
    original_text = speechToText.predict(audio_data)
    
    llama_corrected_text = llamaFrisian.error_correct(original_text)
    
    tasks[task_id]["status"] = "completed"
    tasks[task_id]["result"] = {
        "original": original_text,
        "corrected": llama_corrected_text
    }
    # except Exception as e:
        # print(e)
        # tasks[task_id]["status"] = "failed"
        # tasks[task_id]["error"] = str(e)

@app.route("/data-send", methods=['POST'])
def file_upload():
    file = request.files.get("audio_data")
    if not file:
        return jsonify({"error": "No file provided"}), 400
    
    audio_data = file.read()
    
    task_id = str(uuid.uuid4())
    
    tasks[task_id] = {
        "status": "in_progress",
        "result": None
    }
    
    thread = threading.Thread(target=run_llama_prediction, args=(task_id, audio_data))
    thread.start()
    
    return jsonify({"task_id": task_id})

@app.route("/task-status/<task_id>", methods=['GET'])
def task_status(task_id):
    if task_id not in tasks:
        return jsonify({"error": "Invalid task ID"}), 404
    
    task_info = tasks[task_id]
    return jsonify(task_info)
