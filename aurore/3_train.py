# Train from scratch

from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
from transformers import AutoConfig
from transformers import create_optimizer
import tensorflow as tf
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
from transformers.keras_callbacks import PushToHubCallback
from utils import CONFIG_FILE, config

credentials = config(CONFIG_FILE)

path="aurore/"
file_name="tokenizer"
context_length = 100

#------------------ Fonctions de tokenization du dataset -------------------------

def tokenize(element):
    print("Tokenization of the dataset.")
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=False,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

#--------------------- Récupération du dataset et du tokenizer -----------------

# TODO :Récupération en mode local du tokenizer
tokenizer = None
if not tokenizer:
    raise NotImplementedError

# TODO : charger le jeu de données
dataset = None
if not dataset:
    raise NotImplementedError

# Récupération du tokenizer pré-entrainé en mode HUB
#tokenizer = AutoTokenizer.from_pretrained("benjamin/gpt2-wechsel-french")

#-------------------------- Tokénisation du dataset de phrases ------------------

# Appel de la fonction tokenize qui va transformer chaque phrase du dataset en une phrase de token

# TODO : appel de la fonction map sur le dataset et de tokenize
tokenized_datasets = None

if not tokenized_datasets:
    raise NotImplementedError

print("Dataset tokenisé :", tokenized_datasets)

#--------------------- Préparation des lots (batches) pour l'entrainement  ---------

tokenizer.pad_token = tokenizer.eos_token

# TODO : Créer une instance de la classe DataCollator
data_collator = None

if not data_collator:
    raise NotImplementedError

out = data_collator([tokenized_datasets["train"][i] for i in range(5)])

# for key in out:
#     print(f"{key} shape: {out[key].shape}")

# for key in out:
#     print(f"{key}: {out[key][0]}")
    

#--------------------------- ETAPE 3 : Dataset -> TF DATASET --------------------------

print("\n Conversion du dataset tokenisé en dataset tensorflow \n")

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
)
tf_eval_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=32,
)

# Configuration du réseau GPT2
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Initialisation of the model =/= from pretrained

model = TFGPT2LMHeadModel(config)
print("Construction du modèle")
model(model.dummy_inputs)
model.summary()

# ------------------------ Configuration du réseau --------------------------------------

# Appel du data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="tf")

num_train_steps = len(tf_train_dataset)

# lr scheduler pour améliorer la stabilité du réseau
# Au début lr élevé pour trouver l'optimum pour ensuite se stabiliser

optimizer, schedule = create_optimizer(
    init_lr=5e-5,
    num_warmup_steps=1_000,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)

# Compilation du modèle

model.compile(optimizer=optimizer)
tf.keras.mixed_precision.set_global_policy("mixed_float16")


#---------------------------------- ENTRAINEMENT ---------------------------------------------------------------------

print("\n Entrainement du modèle en cours ... \n")

# epochs = Nombre d'itérations. Attention à ne pas faire exploser votre machine :D
model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=100)

print("Fin de l'entrainement, modèle sauvegardé en local ")
model.save_pretrained("aurore/model/")

# -------------------- Pousser le réseau entrainé sur le HUB ---------------------------------

# On doit installer git-lfs pour cela :

# Linux : apt-get install git-lfs
# Windows : Installer git, puis télécharger une version depuis https://github.com/git-lfs/git-lfs/releases/tag/v3.2.0
# MacOS : avec HomeBrew, faire  brew update    puis  brew install git-lfs

# -------- Une fois téléchargé pour tous -------------

# Ensuite lancer : git lfs install

# Et configurer git
#!git config --global user.email "yourmail@gmail.com"
#!git config --global user.name "yourusername"

# ----------- Enregistrement du modèle ----------------

#HUGGING_FACE_PSEUDO = credentials["hugging_face_pseudo"]
#MODEL_NAME = 'gpt2-George-sand'

# Soit une fois pour toute à la fin de l'entrainement :

#model.push_to_hub(HUGGING_FACE_PSEUDO+"/"+MODEL_NAME)

# Soit pendant l'entrainement au fur et à mesure :

#callback = PushToHubCallback(output_dir=HUGGING_FACE_PSEUDO+"/"+MODEL_NAME, tokenizer=tokenizer)
#model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=3, callbacks=[callback])
