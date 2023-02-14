from flask import Flask, render_template, request
import logging
from transformers import pipeline
from transformers import AutoTokenizer, TFGPT2LMHeadModel, AutoConfig
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfFolder
from utils import CONFIG_FILE, config

class TextGenerator:
    MODEL_NAME  = 'benjamin/gpt2-wechsel-french'

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(TextGenerator.MODEL_NAME)
        # Configuration du réseau GPT2
        config = AutoConfig.from_pretrained(
        TextGenerator.MODEL_NAME,
            vocab_size=len(self.tokenizer),
            n_ctx=100,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Initialisation of the model =/= from pretrained

        model = TFGPT2LMHeadModel(config)
        self.model = model.from_pretrained(TextGenerator.MODEL_NAME, from_pt=True)
        self.model(self.model.dummy_inputs)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0)

    def get_text(self, prompt):
        return self.pipe([prompt], num_return_sequences=1)[0][0]["generated_text"]


app = Flask(__name__)

text_generator = TextGenerator()


@app.route("/generate", methods=('GET', 'POST'))
def generate_text():

    if request.method == 'POST':
        prompt = request.form['amorce']
        text = text_generator.get_text(prompt)
    else:
        text = ""

    logging.warning(text)
    return render_template('aurore.html', text=text)


@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(host="0.0.0.0", port=80, debug=True)