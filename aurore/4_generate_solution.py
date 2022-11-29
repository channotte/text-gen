from transformers import pipeline
from transformers import AutoTokenizer, TFGPT2LMHeadModel, AutoConfig
from datasets import load_dataset, load_from_disk
from tensorflow.keras.models import load_model
from huggingface_hub import HfFolder
from utils import CONFIG_FILE, config

credentials = config(CONFIG_FILE)
HfFolder.save_token(credentials["token"])

path="aurore/"
file_name="tokenizer"


#---------------- Chargement du DS, Tokenizer et du modèle ----------------------------------

print("\n Chargement du dataset, tokenizer et modèle \n")

#--------- En mode local --------------------------------

dataset = load_from_disk("aurore/data/")['validation']
model = TFGPT2LMHeadModel.from_pretrained("aurore/model/", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(path+file_name)

#----------- En mode HUB -------------------------------
# HUGGING_FACE_PSEUDO = credentials["hugging_face_pseudo"]
# HUGGING_FACE_DS_NAME = 'George_Sand'
# MODEL_NAME = 'gpt2-George-sand'

#tokenizer = AutoTokenizer.from_pretrained("benjamin/gpt2-wechsel-french")
#dataset = load_dataset(HUGGING_FACE_PSEUDO+"/"+ HUGGING_FACE_DS_NAME, split="validation")
#model = TFGPT2LMHeadModel.from_pretrained(HUGGING_FACE_PSEUDO+"/"+ MODEL_NAME)


#------------------- Création de la pipeline de génération ---------------------------------

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device=0
)

#-------------------- Génération de texte --------------------------------------------------

prompts = ["Cette jeune femme était assise","Le soir, elle se rappellait"]

output0=pipe(prompts, num_return_sequences=1)[0][0]["generated_text"]
output1=pipe(prompts, num_return_sequences=1)[1][0]["generated_text"]

print("Pour le texte entré :  ", prompts[0], " le text généré est :")
print(output0)
print("Pour le texte entré :  ", prompts[1], " le text généré est :")
print(output1)