import flask

app = flask.Flask(__name__)

@app.route("/")
def main_page():
    return flask.render_template('index.html')
