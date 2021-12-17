from flask import Flask

app = Flask(__name__)

@app.route("/check")
def check():
    return "<p>Hello, World!</p>"
