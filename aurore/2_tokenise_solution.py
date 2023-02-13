# Pour la tokenization on a deux approches.
# Soit utiliser un tokenizer pré-entrainé, soit en entrainer un nouveau
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfFolder
from utils import CONFIG_FILE, config

credentials = config(CONFIG_FILE)
HfFolder.save_token(credentials["token"])

MODEL_NAME  = 'benjamin/gpt2-wechsel-french'


# --------------------- Récupération du dataset ---------------------------------------

# Soit en local
downloaded_dataset = load_from_disk("aurore/data/")

# Soit depuis le hub hugging face

# HUGGING_FACE_PSEUDO = credentials["hugging_face_pseudo"]
# HUGGING_FACE_DS_NAME = 'George_Sand'
# downloaded_dataset = load_dataset(HUGGING_FACE_PSEUDO+"/"+ HUGGING_FACE_DS_NAME)

# --------------------------- CAS 1 : Tokenizer pré-entrainé ---------------------------

print("------------------ TOKENIZER PRE ENTRAINE -------------------")
# On donne une taille max de phrase possible
CONTEXT_LENGTH = 100

# On récupère un tokenizer pré entrainé sur du français pour gpt2 (sur hugging face il y en a pleins)
pretrained_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Le vocabulaire a une taille de {len(pretrained_tokenizer)}")

# Passage de la chaine de caractère au token
txt = "Bonjour Madame, je m'appelle George Sand. Et vous ?"
tokens = pretrained_tokenizer(txt)['input_ids']
# On remarque les symboles spéciaux Ġ et Ċ qui indiquent les espaces et les retours à la ligne.
print("Nombre associé à chaque token : \n",tokens)

# On peut reconvertir le string tokenisé en chaine de caractères et voir son découpage
converted = pretrained_tokenizer.convert_ids_to_tokens(tokens)
print("Chaine de caractères convertie en token : \n", converted, "\n")

# Remarquez que le tokenizer pré-entraîné divise la chaîne donnée en une séquence de sous-mots.
# Comme le suggère la documentation officielle, si un modèle de langue n'est pas disponible dans la langue
# qui vous intéresse, ou si votre corpus est très différent de celui sur lequel votre modèle de langue a été entraîné,
# vous voudrez très probablement réentraîner le modèle à partir de zéro en utilisant un tokenizer adapté à vos données. 
# Pour ce faire, vous devrez entraîner un nouveau tokenizer sur votre jeu de données.

# --------------------------- CAS 2 : Tokenizer pré-entrainé ---------------------------

# Créons un générator pour éviter que Python sauvegarde tout mémoire jusqu'au moment nécessaire

def get_training_corpus():
    batch_size = 1000
    return (
        downloaded_dataset["train"][i : i + batch_size]["text"]
        for i in range(0, len(downloaded_dataset["train"]), batch_size)
    )

training_corpus = get_training_corpus()

print("------------------ TOKENIZER CUSTOMISE -------------------")

print("Analyse des phrases pour l'entrainement du tokenizer :")

for i, text in enumerate(get_training_corpus()):
    print(f"Batch {i} : {len(text)} phrases d'entrainement.")

vocab_size = 52000
tokenizer = pretrained_tokenizer.train_new_from_iterator(training_corpus,vocab_size)
print("Le vocabulaire a une taille de ", tokenizer.vocab_size)

txt = "Bonjour Madame, je m'appelle George Sand. Et vous ?"
tokens = tokenizer(txt)['input_ids']
# On remarque les symboles spéciaux Ġ et Ċ qui indiquent les espaces et les retours à la ligne.
print("Nombre associé à chaque token : \n",tokens)

# On peut reconvertir le string tokenisé en chaine de caractères et voir son découpage
converted = tokenizer.convert_ids_to_tokens(tokens)
print("Chaine de caractères convertie en token : \n", converted)


# Lequel est le meilleur ?


print(f"Il y a {len(tokenizer.tokenize(txt))} tokens pour le tokenizer customisé")
print(f"Il y a {len(pretrained_tokenizer.tokenize(txt))} tokens pour le tokenizer pré-entrainé")

# Le tokeniser qui sait le mieux généraliser ou celui qui permet d'avoir un token par mot ?

# ---------------------------- Sauvegarde et chargement du tokenizer localement --------------------------------------------------

path="aurore/"
file_name="tokenizer"

# Sauvegarde du tokenizer en local
tokenizer.save_pretrained(path+file_name)

#Chargement du tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained(path+file_name)

# ---------------------------- Envoie du tokenizer vers le HUB --------------------------------------------------

# HUGGING_FACE_PSEUDO = credentials["hugging_face_pseudo"]
# HUGGING_FACE_TOK_NAME = 'georgesand-ds-mini'
# tokenizer.push_to_hub(HUGGING_FACE_PSEUDO+"/"+ HUGGING_FACE_TOK_NAME)
# downloaded_tokenizer = AutoTokenizer.from_pretrained(HUGGING_FACE_PSEUDO+"/"+ HUGGING_FACE_TOK_NAME)
# print(downloaded_tokenizer.tokenize(txt))