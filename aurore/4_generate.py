from transformers import pipeline
from transformers import AutoTokenizer, TFGPT2LMHeadModel, AutoConfig
from huggingface_hub import HfFolder
from utils import CONFIG_FILE, config

credentials = config(CONFIG_FILE)
HfFolder.save_token(credentials["token"])

path="aurore/"
file_name="tokenizer"

MODEL_NAME  = 'benjamin/gpt2-wechsel-french'

#---------------- Chargement du Tokenizer et du modèle ----------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

#TODO Construction de la configuration GPT2
config = AutoConfig.from_pretrained(
    #######,
    vocab_size=####,
    n_ctx=#####,
    bos_token_id=####,
    eos_token_id=####,
)

# TODO instancier le modèle

model = #### Chargement de la config dans le modèle
print("Construction du modèle")
model = #### appel de model et de from_pretrained pour charger le modèle

#--------- En mode local : Model entrainé --------------

# model = TFGPT2LMHeadModel.from_pretrained("aurore/model/", local_files_only=True)
# tokenizer = AutoTokenizer.from_pretrained(path+file_name)


#----------- En mode HUB -------------------------------
# HUGGING_FACE_PSEUDO = credentials["hugging_face_pseudo"]
# MODEL_NAME = 'gpt2-George-sand'

#tokenizer = AutoTokenizer.from_pretrained("benjamin/gpt2-wechsel-french")
#model = TFGPT2LMHeadModel.from_pretrained(HUGGING_FACE_PSEUDO+"/"+ MODEL_NAME)


#------------------- Création de la pipeline de génération ---------------------------------

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device=0
)

#-------------------- Génération de texte --------------------------------------------------

# TODO :

prompts = # liste de phrase de prompt

output0= # output du premier élément de la liste prompts après appel à pipe()
output1= # output du second élément de la liste prompts après appel à pipe()

#### TODO Afficher les résultats
