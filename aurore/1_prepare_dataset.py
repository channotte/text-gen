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

# -------------------- Fonctions de data préparation --------------------------------


def download_files(data_dict: dict) -> list:

    for key_name, value_url in data_dict.items():

        print(key_name, value_url)

        # TODO : Utilise la fonction get_file

        filepath = ""

        if not filepath:
            raise NotImplementedError

        texts_train = []

        # TODO : Ouvrir le fichier téléchargé (filepath) avec la fonction with open

        # TODO : Récupérer les lignes du fichier avec readlines()

        # TODO : Retirer les 50 premières lignes (et les 100 dernières...)

        # Astuce, on peut utiliser extend sur une liste.
        # extend est différent d'append. [a].append([b]) = [a,[b]]. [a].extend([b]) = [a,b].

    if not texts_train:
        raise NotImplementedError

    return texts_train


def split_text_to_list(text_list: list) -> list:

    # TODO : Modifier cleaned list pour retirer les espages et \n en fin de ligne
    cleaned_list = []

    if not cleaned_list:
        raise NotImplementedError

    # TODO : Re-créer le texte original avec la méthode join
    list_as_str = ""

    if not list_as_str:
        raise NotImplementedError

    # On utilise nltk pour séparer notre chaine en phrase
    # Séparer par un "." n'est pas suffisant pour identifier une phrase
    # Une phrase peut se finir par "!", "?", "...", "etc."
    # On charge  le tokenizer qui sépare le texte en phrase

    nltk_transformer = nltk.data.load("tokenizers/punkt/PY3/french.pickle")

    # TODO : Appeler NLTK pour tokenizer la chaine de caractères
    splitted_list = []

    if not splitted_list:
        raise NotImplementedError

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


# ----------------------------- Chargement des données --------------------------------

R1_URL = "https://www.gutenberg.org/cache/epub/60812/pg60812.txt" # Adriani
R2_URL = "https://www.gutenberg.org/cache/epub/12862/pg12862.txt" # Aldo le rimeur
R3_URL = "https://www.gutenberg.org/cache/epub/13431/pg13431.txt" # André
R4_URL = "https://www.gutenberg.org/cache/epub/14372/pg14372.txt" # Autour de la table
R5_URL = "https://www.gutenberg.org/cache/epub/61411/pg61411.txt" # Le beau Laurence
R6_URL = "https://www.gutenberg.org/files/25981/25981-0.txt" # Les beaux messieurs de Bois-Doré
R7_URL = "https://www.gutenberg.org/cache/epub/28977/pg28977.txt" # Cadio
R8_URL = "https://www.gutenberg.org/cache/epub/14564/pg14564.txt" # Césarine Dietrich
R9_URL = "https://www.gutenberg.org/cache/epub/13668/pg13668.txt" # Le château des Désertes
R10_URL = "https://www.gutenberg.org/files/17225/17225-0.txt" # La comtesse de Rudolstadt
R11_URL = "https://www.gutenberg.org/cache/epub/12666/pg12666.txt" # Consuelo, volume 1
R12_URL = "https://www.gutenberg.org/files/13258/13258-0.txt" # Consuelo, volume 2
R13_URL = "https://www.gutenberg.org/files/13374/13374-0.txt" # Consuelo, volume 3
R14_URL = "https://www.gutenberg.org/cache/epub/12338/pg12338.txt" # Contes d'une grand-mère
R15_URL = "https://www.gutenberg.org/cache/epub/12837/pg12837.txt" # Cora
R16_URL = "https://www.gutenberg.org/cache/epub/13629/pg13629.txt" # Correspondance, volume 1
R17_URL = "https://www.gutenberg.org/cache/epub/13837/pg13837.txt" # Correspondance, volume 2
R18_URL = "https://www.gutenberg.org/cache/epub/13838/pg13838.txt" # Correspondance, volume 3
R19_URL = "https://www.gutenberg.org/cache/epub/13875/pg13875.txt" # Correspondance, volume 4
R20_URL = "https://www.gutenberg.org/cache/epub/13839/pg13839.txt" # Correspondance, volume 5
R21_URL = "https://www.gutenberg.org/cache/epub/43889/pg43889.txt" # La Coupe; Lupo Liverani; Le Toast; Garnier; Le Contrebandier; La Rêverie à Paris
R22_URL = "https://www.gutenberg.org/cache/epub/69098/pg69098.txt" # Les dames vertes
R23_URL = "https://www.gutenberg.org/cache/epub/13917/pg13917.txt" # La Daniella, volume 1
R24_URL = "https://www.gutenberg.org/cache/epub/14038/pg14038.txt" # La Daniella, volume 2
R25_URL = "https://www.gutenberg.org/cache/epub/17795/pg17795.txt" # La dernière Aldini ; Simon
R26_URL = "https://www.gutenberg.org/files/13653/13653-0.txt" # Elle et lui
R27_URL = "https://www.gutenberg.org/cache/epub/58299/pg58299.txt" # Evenor et Leucippe; Les amours de l'Âge d'Or; Légende antidéluvienne
R28_URL = "https://www.gutenberg.org/cache/epub/15397/pg15397.txt" # Francia; Un bienfait n'est jamais perdu
R29_URL = "https://www.gutenberg.org/cache/epub/13380/pg13380.txt" # Gabriel
R30_URL = "https://www.gutenberg.org/cache/epub/39101/pg39101.txt" # Histoire de ma Vie, Livre 1
R31_URL = "https://www.gutenberg.org/files/41322/41322-0.txt" # Histoire de ma Vie, Livre 2
R32_URL = "https://www.gutenberg.org/files/42765/42765-0.txt" # Histoire de ma Vie, Livre 3
R33_URL = "https://www.gutenberg.org/files/32640/32640-0.txt" # Histoire du véritable Gribouille
R34_URL = "https://www.gutenberg.org/files/14688/14688-0.txt" # Un hiver à Majorque
R35_URL = "https://www.gutenberg.org/cache/epub/13671/pg13671.txt" # Horace
R36_URL = "https://www.gutenberg.org/cache/epub/13744/pg13744.txt" # Isidora
R37_URL = "https://www.gutenberg.org/cache/epub/13818/pg13818.txt" # Jacques
R38_URL = "https://www.gutenberg.org/cache/epub/15584/pg15584.txt" # Jean Ziska
R39_URL = "https://www.gutenberg.org/cache/epub/17589/pg17589.txt" # Journal d'un voyageur pendant la guerre
R40_URL = "https://www.gutenberg.org/cache/epub/13303/pg13303.txt" # Kourroglou
R41_URL = "https://www.gutenberg.org/cache/epub/13016/pg13016.txt" # Lavinia
R42_URL = "https://www.gutenberg.org/files/17911/17911-0.txt" # Légendes rustiques
R43_URL = "https://www.gutenberg.org/files/39738/39738-0.txt" # Lélia
R44_URL = "https://www.gutenberg.org/cache/epub/15388/pg15388.txt" # Leone Leoni
R45_URL = "https://www.gutenberg.org/files/37989/37989-0.txt" # Lettres d'un voyageur
R46_URL = "https://www.gutenberg.org/cache/epub/16286/pg16286.txt" # Lucrezia Floriani
R47_URL = "https://www.gutenberg.org/cache/epub/18075/pg18075.txt" # Mademoiselle La Quintinie
R48_URL = "https://www.gutenberg.org/cache/epub/20254/pg20254.txt" # Les Maîtres sonneurs
R49_URL = "https://www.gutenberg.org/files/23582/23582-0.txt" # La Mare au Diable
R50_URL = "https://www.gutenberg.org/cache/epub/13025/pg13025.txt" # La Marquise
R51_URL = "https://www.gutenberg.org/cache/epub/12865/pg12865.txt" # Mattea
R52_URL = "https://www.gutenberg.org/files/62787/62787-0.txt" # Mauprat
R53_URL = "https://www.gutenberg.org/cache/epub/12869/pg12869.txt" # Metella
R54_URL = "https://www.gutenberg.org/cache/epub/13892/pg13892.txt" # Le meunier d'Angibault
R55_URL = "https://www.gutenberg.org/cache/epub/15226/pg15226.txt" # Nanon
R56_URL = "https://www.gutenberg.org/cache/epub/13198/pg13198.txt" # Nouvelles lettres d'un voyageur
R57_URL = "https://www.gutenberg.org/cache/epub/15235/pg15235.txt" # Oeuvres illustrées de George Sand
R59_URL = "https://www.gutenberg.org/cache/epub/12448/pg12448.txt" # L'Orco
R60_URL = "https://www.gutenberg.org/cache/epub/12447/pg12447.txt" # Pauline
R61_URL = "https://www.gutenberg.org/files/12367/12367-0.txt" # Le péché de Monsieur Antoine, volume 1
R62_URL = "https://www.gutenberg.org/files/12534/12534-0.txt" # Le péché de Monsieur Antoine, volume 2
R63_URL = "https://www.gutenberg.org/cache/epub/34204/pg34204.txt" # La petite fadette
R64_URL = "https://www.gutenberg.org/files/30831/30831-0.txt" # Le Piccinino
R65_URL = "https://www.gutenberg.org/cache/epub/28623/pg28623.txt" # Le poême de Myrza - Hamlet
R66_URL = "https://www.gutenberg.org/cache/epub/12889/pg12889.txt" # Promenades autour d'un village
R67_URL = "https://www.gutenberg.org/files/26614/26614-0.txt" # Le secrétaire intime
R70_URL = "https://www.gutenberg.org/files/18205/18205-0.txt" # Simon
R71_URL = "https://www.gutenberg.org/files/15239/15239-0.txt" # Spiridion
R72_URL = "https://www.gutenberg.org/files/45753/45753-0.txt" # Tamaris
R73_URL = "https://www.gutenberg.org/cache/epub/15287/pg15287.txt" # Teverino
R74_URL = "https://www.gutenberg.org/files/13592/13592-0.txt" # L'Uscoque
R75_URL = "https://www.gutenberg.org/files/17251/17251-0.txt" # Valentine
R76_URL = "https://www.gutenberg.org/cache/epub/13263/pg13263.txt" # Valvèdre

# Première étape créer un dictionnaire
# Clé du dictionnaire = Titre en chaine de caractères
# Valeur : Url (string) du fichier .txt à télécharger

dict_train = {
    #"R1":R1_URL, # réservé pour le test
    #"R2":R2_URL, # réservé pour le test
    #"R3":R3_URL, # réservé pour le test
    "R4":R4_URL,
    "R5":R5_URL,
    "R6":R6_URL,
    "R7":R7_URL,
    "R8":R8_URL,
    "R9":R9_URL,
    "R10":R10_URL,
    "R11":R11_URL,
    "R12":R12_URL,
    "R13":R13_URL,
    "R14":R14_URL,
    "R15":R15_URL,
    "R16":R16_URL,
    "R17":R17_URL,
    "R18":R18_URL,
    "R19":R19_URL,
    "R20":R20_URL,
    "R21":R21_URL,
    "R22":R22_URL,
    "R23":R23_URL,
    "R24":R24_URL,
    "R25":R25_URL,
    "R26":R26_URL,
    "R27":R27_URL,
    "R28":R28_URL,
    "R29":R29_URL,
    "R30":R30_URL,
    "R31":R31_URL,
    "R32":R32_URL,
    "R33":R33_URL,
    "R34":R34_URL,
    "R35":R35_URL,
    "R36":R36_URL,
    "R37":R37_URL,
    "R38":R38_URL,
    "R39":R39_URL,
    "R40":R40_URL,
    "R41":R41_URL,
    "R42":R42_URL,
    "R43":R43_URL,
    "R44":R44_URL,
    "R45":R45_URL,
    "R46":R46_URL,
    "R47":R47_URL,
    "R48":R48_URL,
    "R49":R49_URL,
    #"R50":R50_URL, # réservé pour le test
    #"R51":R51_URL, # réservé pour le test
    #"R52":R52_URL, # réservé pour le test
    "R53":R53_URL,
    "R54":R54_URL,
    "R55":R55_URL,
    "R56":R56_URL,
    "R57":R57_URL,
    "R58":R58_URL,
    "R59":R59_URL,
    "R60":R60_URL,
    "R61":R61_URL,
    "R62":R62_URL,
    "R63":R63_URL,
    "R64":R64_URL,
    "R65":R65_URL,
    "R66":R66_URL,
    "R67":R67_URL,
    "R68":R68_URL,
    "R69":R69_URL,
    "R70":R70_URL,
    "R71":R71_URL,
    "R72":R72_URL,
    "R73":R73_URL,
    "R74":R74_URL,
    "R75":R75_URL,
    "R76":R_76_URL}

dict_test = {
    "R1": R1_URL,
    "R2": R2_URL,
    "R3": R3_URL,
    "R50": R50_URL,
    "R51": R51_URL,
    "R52": R52_URL}

# On appelle nos fonctions qui nous renvoient une liste de phrases préparées

text_list_train = prepare_dataset(dict_train)

# On fait pareil pour le test

text_list_test = prepare_dataset(dict_test)

# Maintenant on est revenu au début, il mais nous avons un texte nettoyé.
# Il nous suffit de charger les données en créant un objet de la classe Dataset
# Cette fois ci pas en utilisant load_dataset() qui charge depuis un fichier (texte, csv, parquet)
# Mais en créant une instance de la classe dataset.
# Rapel : Nous avons une liste de texte. Plusieurs options de chargement

# Créer un tableau pandas d'une colonne du nom "train_text" et charger notre liste
# Chaque phrase = 1 ligne du tableau
# On utilise la fonction from_pandas() venant du nom de la bibliothèque de dataframe
dataset_train = Dataset.from_pandas(pd.DataFrame({"train_text": text_list_train}))

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
# On peut adapter les modifications selon le niveau de traitement
# (répartition statistique, découpage spécial)

# Autres traitements possibles...
# On peut calculer la taille de chaque phrase
# On peut utiliser directement des fonctions map sur la classe Dataset
# pour effectuer un traitement direct

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

dataset_new_col_train = dataset_new_col_train.filter(lambda x: x["sentence_length"] > 3)
print("Nombre de phrases dans le train initial : ", dataset_train.num_rows)
print("Nombre de phrases dans le train filtré : ", dataset_new_col_train.num_rows)

# On fait pareil pour le dataset de test

dataset_new_col_test = (
    dataset["test"]
    .map(compute_sentence_length)
    .filter(lambda x: x["sentence_length"] > 3)
)

# ----------------------------- TRAIN/VALIDATION/TEST ---------------------------------------------

# On sépare maintenant notre datasein d'entrainement (train) en deux sets
# 1. Un d'entrainement pur (train) qui nous permettra d'apprendre sur la donnée fournie
# 2. Un de vérification (validation) sur lequel on évaluera notre modèle.
# Le dataset de validation nous donne un indice sur notre future performance
# en conditions réelles (données inconnues)

print("\n ----------- SPLIT DATASET ---------\n")

dataset_train_splitted = dataset_new_col_train.train_test_split(train_size=0.9, seed=42)
# Par défaut ce dataset s'appelle test, on le renomme par "validation"
dataset_train_splitted["validation"] = dataset_train_splitted.pop("test")

print(dataset_train_splitted)

# On le sauvegarde en local
dataset_train_splitted.save_to_disk("aurore/data/")

# # ---------------------------- PARTAGE SUR LE HUB DU DATASET -------------------------------------

# # HUGGING_FACE_PSEUDO = credentials["hugging_face_pseudo"]
# # HUGGING_FACE_DS_NAME = 'George_Sand'
# # dataset_train_splitted.push_to_hub(HUGGING_FACE_PSEUDO+"/"+ HUGGING_FACE_DS_NAME)
# # downloaded_dataset = load_dataset(HUGGING_FACE_PSEUDO+"/"+ HUGGING_FACE_DS_NAME)
# # print(downloaded_dataset)
