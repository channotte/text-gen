import keras
import nltk
import pandas as pd
import tensorflow as tf

import datasets
from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from huggingface_hub import HfFolder
from utils import CONFIG_FILE, config

credentials = config(CONFIG_FILE)
HfFolder.save_token(credentials["token"])

nltk.download("punkt")

# ----------------------------- Chargement des données --------------------------------

LELIA_URL = "https://www.gutenberg.org/files/39738/39738-0.txt"
LA_PETITE_FADETTE_URL = "https://www.gutenberg.org/cache/epub/34204/pg34204.txt"
GABRIEL_URL = "https://www.gutenberg.org/cache/epub/13380/pg13380.txt"
LETTRE_VOYAGEUR_URL = "https://www.gutenberg.org/files/37989/37989-0.txt"
LA_MARQUISE_URL = "https://www.gutenberg.org/cache/epub/13025/pg13025.txt"
DAME_VERTES_URL = "https://www.gutenberg.org/cache/epub/69098/pg69098.txt"
MEUNIER_ANGIBAULT_URL = "https://www.gutenberg.org/cache/epub/13892/pg13892.txt"
COMPTESSE_RUDOLSTADT_URL = "https://www.gutenberg.org/files/17225/17225-0.txt"
INDIANA_URL = "https://www.gutenberg.org/files/63445/63445-0.txt"
HISTOIRE_DE_MA_VIE_1_URL = "https://www.gutenberg.org/cache/epub/39101/pg39101.txt"
HISTOIRE_DE_MA_VIE_3_URL = "https://www.gutenberg.org/files/42765/42765-0.txt"
MARE_AU_DIABLE_URL = "https://www.gutenberg.org/files/23582/23582-0.txt"
VALENTINE_URL = "https://www.gutenberg.org/files/17251/17251-0.txt"


# Première étape créer un dictionnaire
# Clé du dictionnaire = Titre en chaine de caractères
# Valeur : Url (string) du fichier .txt à télécharger

dict_train = {
    "Lélia": LELIA_URL,
    "La petite fadette": LA_PETITE_FADETTE_URL,
    "Gabriel": GABRIEL_URL,
    "Lettre d'un voyageur": LETTRE_VOYAGEUR_URL,
    "La Marquise": LA_MARQUISE_URL,
    "Les dames vertes": DAME_VERTES_URL,
    "Le meunier d'Angibault": MEUNIER_ANGIBAULT_URL,
    "La comptesse de Rudolstadt": COMPTESSE_RUDOLSTADT_URL,
    'Indiana': INDIANA_URL,
    'Histoire de ma vie, livre 1': HISTOIRE_DE_MA_VIE_1_URL,
    'Histoire de ma vie, livre 3': HISTOIRE_DE_MA_VIE_3_URL,
}

dict_test = {"La Mare au Diable": MARE_AU_DIABLE_URL}

# -------------------- Fonctions de data préparation --------------------------------


def download_files(data_dict: dict) -> list:

    texts_train = []
    
    for key_name, value_url in data_dict.items():
        # print(key_name, value_url)
        filepath = keras.utils.get_file(f"{key_name}.txt", origin=value_url)
        with open(filepath, encoding="utf-8") as file:
            text = file.readlines()
            # On peut aussi enlever les sauts de lignes au début avec
            # text = file.read().splitlines()
            # Soit on le fait plus tard dans le clean
            print("\n")
            print("Livre :", key_name)
            print("Les deux premières lignes du fichier : ", text[:2])
            print("Nombre de lignes dans le fichier : ", len(text))

            # print(text[0:20])
            # On remarque que les 50 premières lignes sont l'intro et la préface de Gutenberg
            # On va les retirer pour avoir le moins de bruit possible


            for row_end in range(-1, -len(text), -1):
                if 'END' in text[row_end] or 'FIN' in text[row_end]:
                    print(text[row_end], row_end)
                    break



            texts_train.extend(text[50: row_end])

            # On remarque aussi que les dernières lignes comportent aussi du bruit et en anglais
            # On retire environ les 200 dernières lignes pour éviter d'incorporer de l'anglais

            # extend est différent d'append. [a].append([b]) = [a,[b]]. [a].extend([b]) = [a,b].

    return texts_train


def split_text_to_list(text_list: list) -> list:
    print(f"nb rows: {len(text_list)}")
    # On retire les espaces à droite et les \n pour chaque ligne
    cleaned_list = [line.rstrip() for line in text_list]

    # On crée une chaine de caractères qui concatène chaque phrase et les sépare d'un espace
    list_as_str = " ".join(cleaned_list)
    list_as_str = list_as_str.replace('_', '')

    # On utilise nltk pour séparer notre chaine en phrase
    # Séparer par un "." n'est pas suffisant pour identifier une phrase
    # Une phrase peut se finir par "!", "?", "...", "etc."
    # On charge  le tokenizer qui sépare le texte en phrase
    tokenizer = nltk.data.load("tokenizers/punkt/PY3/french.pickle")

    # Et on l'applique sur notre liste
    splitted_list = tokenizer.tokenize(list_as_str)
    # print(splitted_list)

    # On enlève le texte vide (des phrases sans aucun mots)
    text_list = list(filter(None, splitted_list))
    # Pourcentage de texte enlevé
    percent = 1 - len(text_list) / len(splitted_list)
    print(f"\n Pourcentage de phrases None enlevées {round(percent*100,2)}")
    print("Nombre de phrases complètes", len(text_list))
    return text_list


def prepare_dataset(data_dict: dict) -> list:
    text_str = download_files(data_dict)
    prepared_text_list = split_text_to_list(text_str)
    return prepared_text_list


def compute_sentence_length(example) -> str:
    return {"sentence_length": len(example["text"].split())}



# On appelle nos fonctions qui nous renvoient une liste de phrases préparées

text_list_train = prepare_dataset(dict_train)

# On fait pareil pour le test

text_list_test = prepare_dataset(dict_test)

# Maintenant on est revenu au début, il mais nous avons un texte nettoyé.
# Il nous suffit de charger les données en créant un objet de la classe Dataset
# Cette fois ci pas en utilisant load_dataset() qui charge depuis un fichier (texte, csv, parquet...)
# Mais en créant une instance de la classe dataset.
# Rapel : Nous avons une liste de texte. Plusieurs options de chargement

# Créer un tableau pandas d'une colonne du nom "train_text" et charger notre liste
# Chaque phrase = 1 ligne du tableau
# On utilise la fonction from_pandas() venant du nom de la bibliothèque de dataframe
dataset_train = Dataset.from_pandas(
    pd.DataFrame({"train_text": text_list_train}))

# Ou Créer un dictionnaire et utiliser la fonction from_dict(), la clé est le nom de la colonne

# On ajoute le type de nos données avec la classe Features de Hugging Face
# On indique que la colonne "text" est de type string
features = Features({"text": Value(dtype="string")})
dataset_train = Dataset.from_dict({"text": text_list_train}, features)

print("Taille du dataset d'entrainement ", dataset_train.shape)

# On fait pareil pour le dataset de test, le typage des données est le même
dataset_test = Dataset.from_dict({"text": text_list_test}, features)

dataset = DatasetDict({"train": dataset_train, "test": dataset_test})
print("Dataset final: \n ", dataset)

# On est revenus à l'état initial mais en ayant modifié nos données
# On peut adapter les modifications selon le niveau de traitement (répartition statistique, découpage spécial)

# Autres traitements possibles...
# On peut calculer la taille de chaque phrase
# On peut utiliser directement des fonctions map sur la classe Dataset pour effectuer un traitement direct

dataset_new_col_train = dataset["train"].map(compute_sentence_length)
print(
    "Nouvelle colonne ajoutée", dataset_new_col_train[0]
)  # Ajout d'une nouvelle colonne

# Certaines phrases ont trop peu de mots
print(
    "Type de phrases trop courtes dans le texte : ",
    dataset_new_col_train.sort("sentence_length")[:3],
)
# On peut les supprimer (par exemple, retirer les phrases qui ont une taille <= 3 mots)
# Cela nous permettra d'avoir des phrases générées plus cohérentes (et plus longues)

dataset_new_col_train = dataset_new_col_train.filter(
    lambda x: x["sentence_length"] > 3)
print("Nombre de phrases dans le train initial : ", dataset_train.num_rows)
print("Nombre de phrases dans le train filtré : ",
      dataset_new_col_train.num_rows)

# On fait pareil pour le dataset de test

dataset_new_col_test = (
    dataset["test"]
    .map(compute_sentence_length)
    .filter(lambda x: x["sentence_length"] > 3)
)

# ----------------------------- TRAIN/VALIDATION/TEST ------------------------------------------------

# On sépare maintenant notre dataset d'entrainement (train) en deux sets
# 1. Un d'entrainement pur (train) qui nous permettra d'apprendre sur la donnée fournie
# 2. Un de vérification (validation) sur lequel on évaluera notre modèle.
# Le dataset de validation nous donne un indice sur notre future performance en conditions réelles (données inconnues)

print("\n ----------- SPLIT DATASET ---------\n")

dataset_train_splitted = dataset_new_col_train.train_test_split(
    train_size=0.9, seed=42)
# Par défaut ce dataset s'appelle test, on le renomme par "validation"
dataset_train_splitted["validation"] = dataset_train_splitted.pop("test")

print(dataset_train_splitted)

# On le sauvegarde en local
dataset_train_splitted.save_to_disk("aurore/data/")

with open("aurore/data/train_text.txt", "w", encoding="utf-8") as f:
    f.writelines(
        [sentence + '\n' for sentence in dataset_train_splitted['train']['text']])

with open("aurore/data/validation_text.txt", "w", encoding="utf-8") as f:
    f.writelines(
        [sentence + '\n' for sentence in dataset_train_splitted['validation']['text']])
print("Dataset saved to disk")


# # ---------------------------- PARTAGE SUR LE HUB DU DATASET --------------------------------------------------

# # HUGGING_FACE_PSEUDO = credentials["hugging_face_pseudo"]
# # HUGGING_FACE_DS_NAME = 'George_Sand'
# # dataset_train_splitted.push_to_hub(HUGGING_FACE_PSEUDO+"/"+ HUGGING_FACE_DS_NAME)
# # downloaded_dataset = load_dataset(HUGGING_FACE_PSEUDO+"/"+ HUGGING_FACE_DS_NAME)
# # print(downloaded_dataset)
