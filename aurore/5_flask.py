from flask import Flask, render_template
import logging

app = Flask(__name__)

@app.route('/')
def home(name=None):
    return render_template('home.html', name=name)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(host="0.0.0.0", port=80)