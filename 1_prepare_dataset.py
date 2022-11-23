import os
import re
import tensorflow as tf
import datasets
from datasets import load_dataset, Dataset, DatasetDict, Features, Value
import keras
import pandas as pd
import random

######### PREMIERE PARTIE ###############
lelia_url = "https://www.gutenberg.org/files/39738/39738-0.txt"
la_petite_fadette_url = "https://www.gutenberg.org/cache/epub/34204/pg34204.txt"
gabriel_url = "https://www.gutenberg.org/cache/epub/13380/pg13380.txt"
lettre_voyageur_url = "https://www.gutenberg.org/files/37989/37989-0.txt"
la_mare_au_diable_url = "https://www.gutenberg.org/files/23582/23582-0.txt"

train_paths = [lelia_url, la_petite_fadette_url, gabriel_url, lettre_voyageur_url]
test_path = la_mare_au_diable_url


dataset = load_dataset("text", data_files={"train": train_paths, "test": test_path})

print(dataset)
print(dataset["train"][0:20])
print(dataset["train"][40:50])

# On est encore dans la préface, il y a des lignes vides

# On remarque qu'il y a des headers dans nos datas
# Même s'il existe des outils, charger nos data sans les vérifier = suicide
# Il faut absolument faire un prétraitement sur nos données
# On va créer une classe, et voir la répartition des données

dict_train = {
    "Lélia": lelia_url,
    "La petite fadette": la_petite_fadette_url,
    "Gabriel": gabriel_url,
    "Lettre d'un voyageur": lettre_voyageur_url,
}

dict_test = {"La Mare au Diable": la_mare_au_diable_url}


class DatasetPrepper:
    def __init__(self, data_dict) -> None:
        self.data_dict = data_dict
        # after add data
        self.prepared_text_list = []

    # first step is to create dict
    # then save into a entire str
    def download_files(self):

        for key_name, value_url in self.data_dict.items():
            # print(key_name, value_url)
            filepath = keras.utils.get_file(f"{key_name}.txt", origin=value_url)
            texts_train = []
            with open(filepath, encoding="utf-8") as file:
                text = file.readlines()
                print(text[:3])
                print("nb lignes dans le fichier", len(text))
                # enlever les \n soit au début
                # file.read().splitlines()
                
                # soit plus tard dans le clean
                 
                # print(text[0:20])
                # First 50 lines are the Gutenberg intro and preface
                # Skipping first 50 lines for each book should be approximately
                # removing the intros and prefaces.
                # 50 dernières lignes idem bruit
                #extend est différent d'append
                texts_train.extend(text[50:len(text)-50])
                
        return texts_train

    def split_text_to_list(self, text_list):
        cleaned_list = [line.rstrip() for line in text_list]
        list_as_str = ' '.join(cleaned_list)
        text_list_train = list_as_str.split(".")
        print(f"{len(text_list_train)} in text")

        # On enlève le texte vide
        text_list = list(filter(None, text_list_train))
        # Pourcentage de texte enlevé
        percent = 1 - len(text_list) / len(text_list_train)
        print(f"Percentage of None value removed {round(percent*100,2)}")

        # Shuffle ou pas shuffle ?
        # random.shuffle(text_list)
        print("Nombre de phrases complètes", len(text_list))
        return text_list

    def prepare_dataset(self):
        text_str = self.download_files()
        self.prepared_text_list = self.split_text_to_list(text_str)
        return self.prepared_text_list


train_prepper = DatasetPrepper(dict_train)
text_list_train = train_prepper.prepare_dataset()

# EQUIVALENT A
# text_list_train = DatasetPrepper(dict_train).prepare_dataset()

# On fait pareil pour le test

text_list_test = DatasetPrepper(dict_test).prepare_dataset()

# Maintenant on est revenu au début, il mais nous avons un texte nettoyé.
# Il nous suffit de charger les données en créant un objet de la classe Dataset
# Cette fois ci pas en utilisant load_dataset() qui charge depuis un fichier (texte, csv, parquet...)
# Mais en créant une instance de la classe dataset.
# Rapel : Nous avons une liste de texte. Plusieurs options de chargement

# Créer un tableau pandas d'une colonne du nom "train_text" et charger notre liste
# Chaque phrase = 1 ligne du tableau
# On utilise la fonction from_pandas() venant du nom de la bibliothèque de dataframe
dataset_train = Dataset.from_pandas(pd.DataFrame({"train_text": text_list_train}))

# Ou Créer un dictionnaire et utiliser la fonction from_dict(), la clé est le nom de la colonne
print(len(text_list_train))
features = Features({'text' : Value(dtype='string')})
dataset_train = Dataset.from_dict({"text": text_list_train}, features)

print(dataset_train.shape)

# On fait pareil pour le dataset de test
dataset_test = Dataset.from_dict({"text": text_list_test}, features)

dataset = DatasetDict({"train": dataset_train, "test": dataset_test})
print(dataset)

# On est revenus à l'état initial mais en ayant modifié nos données
# On peut adapter les modifications selon le niveau de traitement (répartition statistique, découpage spécial)

# EXEMPLE A RAJOUTER DANS LA CLASSE DATASET PREPPER #
def compute_sentence_length(example):
    return {"sentence_length": len(example["text"].split())}

dataset_new_col_train= dataset['train'].map(compute_sentence_length)
print(dataset_new_col_train[0]) # new column added

# Certaines phrases ont trop peu de mots
print(dataset_new_col_train.sort("sentence_length")[:3])
# ON retire les phrases qui ont une taille <= 3 mots

dataset_new_col_train = dataset_new_col_train.filter(lambda x: x["sentence_length"] > 3)
print(dataset_new_col_train.num_rows)

# On fait pareil pour le dataset de validation

dataset_new_col_test = dataset['test'].map(compute_sentence_length).filter(lambda x: x["sentence_length"] > 3)

# ----------------------------- TRAIN/VALIDATION/TEST ------------------------------------------------

dataset_train_splitted = dataset_new_col_train.train_test_split(train_size=0.9, seed=42)
# Rename the default "test" split to "validation"
dataset_train_splitted["validation"] = dataset_train_splitted.pop("test")

print(dataset_train_splitted)

# ---------------------------- SHARE DATASET ON HUB --------------------------------------------------

# HUGGING_FACE_PSEUDO = 'channotte'
# HUGGING_FACE_DS_NAME = 'Georges_Sand'
# dataset_train_splitted.push_to_hub(HUGGING_FACE_PSEUDO+"/"+ HUGGING_FACE_DS_NAME)
# downloaded_dataset = load_dataset(HUGGING_FACE_PSEUDO+"/"+ HUGGING_FACE_DS_NAME)
# print(downloaded_dataset)