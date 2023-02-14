from transformers import pipeline
from transformers import AutoTokenizer, TFGPT2LMHeadModel, AutoConfig
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfFolder
from utils import CONFIG_FILE, config

credentials = config(CONFIG_FILE)
HfFolder.save_token(credentials["token"])

path="aurore/"
file_name="tokenizer"

MODEL_NAME  = 'benjamin/gpt2-wechsel-french'

#---------------- Chargement du DS, Tokenizer et du modèle ----------------------------------

print("\n Chargement du dataset, tokenizer et modèle \n")

#--------- En mode local : Model Pré-entrainé Hugging Face --------------

dataset = load_from_disk("aurore/data/")['validation']
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Construction de la configuration GPT2
config = AutoConfig.from_pretrained(
    #######,
    vocab_size=####,
    n_ctx=#####,
    bos_token_id=####,
    eos_token_id=####,
)

# Initialisation of the model =/= from pretrained

model = #### Chargement de la config
print("Construction du modèle")
model = #### Chargement du modèle

#--------- En mode local : Model entrainé --------------

# dataset = load_from_disk("aurore/data/")['validation']
# model = TFGPT2LMHeadModel.from_pretrained("aurore/model/", local_files_only=True)
# tokenizer = AutoTokenizer.from_pretrained(path+file_name)


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

prompts = # liste de phrase de prompt

output0= # output du premier élément de la liste prompts après appel à pipe()
output1= # output du second élément de la liste prompts après appel à pipe()

#### Afficher les résultats