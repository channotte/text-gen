# Train from scratch

from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
from transformers import AutoConfig
from transformers import create_optimizer
from transformers import TrainingArguments, Trainer
import tensorflow as tf
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import logging
from datetime import datetime
from pathlib import Path
from utils import CONFIG_FILE, config

credentials = config(CONFIG_FILE)

AURORE_PATH= Path("aurore")
MODEL_PATH = AURORE_PATH / 'model'
file_name="tokenizer"
context_length = 100

#------------------ Fonctions d'entrainement -------------------------------------

class Saver(Callback):
    VAL_LOSS = 'val_loss'

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.best_val_loss = None

        now = datetime.now()
        date = now.strftime("%Y-%m-%d %H:%M:%S")

        self.model_path = MODEL_PATH / date

    def on_epoch_end(self, epoch, logs=None):
        if self.best_val_loss is None:
            self.best_val_loss = logs[Saver.VAL_LOSS]
            self.model.save_pretrained(self.model_path)
            logging.warning(f"\nInitialize saved model at epoch {epoch}\n")
        elif self.best_val_loss > logs[Saver.VAL_LOSS]:
            self.model.save_pretrained(self.model_path)
            logging.warning(f"\nUpdated saved model at epoch {epoch} (previous loss: {self.best_val_loss}, current loss: {logs[Saver.VAL_LOSS]}\n")
            self.best_val_loss = logs[Saver.VAL_LOSS]


def tokenize(element):

    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=False,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length <= context_length:
            input_batch.append(input_ids)
        else:
            input_batch.append(input_ids[:context_length])

    return {"input_ids": input_batch}

#--------------------- R??cup??ration du dataset et du tokenizer -----------------

dataset = load_from_disk("aurore/data/")


# R??cup??ration en mode local
tokenizer = AutoTokenizer.from_pretrained(AURORE_PATH/file_name)

# R??cup??ration du tokenizer pr??-entrain??
#tokenizer = AutoTokenizer.from_pretrained("benjamin/gpt2-wechsel-french")

#-------------------------- Tok??nisation du dataset de phrases ------------------

# Appel de la fonction tokenize qui va transformer chaque phrase du dataset en une phrase de token

print("Dataset inital :", dataset)

tokenized_datasets = dataset.map(
    tokenize, batched=True, remove_columns=dataset["train"].column_names
)

print("Dataset tokenis?? :", tokenized_datasets)

#--------------------- Pr??paration des lots (batches) pour l'entrainement  ---------

tokenizer.pad_token = tokenizer.eos_token

# Chargement des tokens par lot avec le Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="tf")


#--------------------------- ETAPE 3 : Dataset -> TF DATASET --------------------------

print("\n Conversion du dataset tokenis?? en dataset tensorflow \n")

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=8,
)
tf_eval_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=8,
)

# Configuration du r??seau GPT2
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Initialisation of the model =/= from pretrained

model = TFGPT2LMHeadModel(config)
print("Construction du mod??le")
model(model.dummy_inputs)
model.summary()

# ------------------------ Configuration du r??seau --------------------------------------

num_train_steps = len(tf_train_dataset)

# lr scheduler pour am??liorer la stabilit?? du r??seau
# Au d??but lr ??lev?? pour trouver l'optimum pour ensuite se stabiliser

optimizer, schedule = create_optimizer(
    init_lr=5e-5,
    num_warmup_steps=1_000,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)

# # Compilation du mod??le

model.compile(optimizer=optimizer)
tf.keras.mixed_precision.set_global_policy("mixed_float16")


# #---------------------------------- ENTRAINEMENT ---------------------------------------------------------------------

print("\n Entrainement du mod??le en cours ... \n")


# epochs = Nombre d'it??rations. Attention ?? ne pas faire exploser votre machine :D
saver = Saver(model)

model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=1000, callbacks=[saver])


# -------------------- Pousser le r??seau entrain?? sur le HUB ---------------------------------

# # On doit installer git-lfs pour cela :

# # Linux : apt-get install git-lfs
# # Windows : Installer git, puis t??l??charger une version depuis https://github.com/git-lfs/git-lfs/releases/tag/v3.2.0
# # MacOS : avec HomeBrew, faire  brew update    puis  brew install git-lfs

# # -------- Une fois t??l??charg?? pour tous -------------

# # Ensuite lancer : git lfs install

# # Et configurer git
# #!git config --global user.email "yourmail@gmail.com"
# #!git config --global user.name "yourusername"

# # ----------- Enregistrement du mod??le ----------------

# #HUGGING_FACE_PSEUDO = credentials["hugging_face_pseudo"]
# #MODEL_NAME = 'gpt2-George-sand'

# # Soit une fois pour toute ?? la fin de l'entrainement :

# #model.push_to_hub(HUGGING_FACE_PSEUDO+"/"+MODEL_NAME)

# # Soit pendant l'entrainement au fur et ?? mesure :

# #callback = PushToHubCallback(output_dir=HUGGING_FACE_PSEUDO+"/"+MODEL_NAME, tokenizer=tokenizer)
# #model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=3, callbacks=[callback])
