# Pour la tokenization on a deux approches.
# Soit utiliser un tokenizer pré-entrainé, soit en entrainer un nouveau
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset

HUGGING_FACE_PSEUDO = 'channotte'
HUGGING_FACE_DS_NAME = 'Georges_Sand'

downloaded_dataset = load_dataset(HUGGING_FACE_PSEUDO+"/"+ HUGGING_FACE_DS_NAME)

# --------------------------- CAS 1 : Tokenizer pré-entrainé ---------------------------

context_length = 100  # Taille max phrase
pretrained_tokenizer = AutoTokenizer.from_pretrained("benjamin/gpt2-wechsel-french")

# Nous savons que la plupart des commentaires contiennent plus de 40 tokens,
# donc le simple fait de tronquer les entrées à la longueur maximale éliminerait une grande partie de notre ensemble de données.
# Au lieu de cela, nous allons utiliser l'option return_overflowing_tokens pour tokeniser l'entrée entière et la diviser
# en plusieurs morceaux. Nous utiliserons également l'option return_length pour retourner automatiquement la longueur de chaque
# morceau créé. Souvent, le dernier morceau sera plus petit que la taille du contexte, et nous nous débarrasserons de ces morceaux
# pour éviter les problèmes de remplissage ; nous n'en avons pas vraiment besoin car nous avons beaucoup de données de toute façon.

outputs = pretrained_tokenizer(
    downloaded_dataset["train"][:2]["text"],
    truncation=True,
    max_length=context_length,
    return_overflowing_tokens=False,
    return_length=True,
)

print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {(outputs['length'])}")

print("vocab_size: ", len(pretrained_tokenizer))

# From STR to token
txt = "Bonjour Madame, je m'appelle Georges Sand. Et vous ?"
tokens = pretrained_tokenizer(txt)['input_ids']
# On remarque les symboles spéciaux Ġ et Ċ qui indiquent les espaces et les retours à la ligne.
print(tokens)

# We can convert back the token to string
converted = pretrained_tokenizer.convert_ids_to_tokens(tokens)
print(converted)

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

for text in get_training_corpus():
    print(len(text))

vocab_size = 52000
tokenizer = pretrained_tokenizer.train_new_from_iterator(training_corpus,vocab_size)
print(tokenizer.eos_token_id)
print(tokenizer.vocab_size)

txt = "Bonjour Madame, je m'appelle Georges Sand. Et vous ?"
tokens = tokenizer(txt)['input_ids']
# On remarque les symboles spéciaux Ġ et Ċ qui indiquent les espaces et les retours à la ligne.
print(tokens)

# We can convert back the token to string
converted = tokenizer.convert_ids_to_tokens(tokens)
print(converted)


# Lequel est le meilleur ?


print(len(tokenizer.tokenize(txt)))
print(len(pretrained_tokenizer.tokenize(txt)))

# Le tokeniser qui sait le mieux généraliser ou celui qui permet d'avoir un token par mot ?

# ---------------------------- Save and load tokenizer locally --------------------------------------------------

path="./"
file_name="georgessand-ds-mini"
tokenizer.save_pretrained(path+file_name)
loaded_tokenizer = AutoTokenizer.from_pretrained(path+file_name)
print(tokenizer.tokenize(txt))

# ---------------------------- PUSH TOKENIZER TO HUB--------------------------------------------------

# HUGGING_FACE_PSEUDO = 'channotte'
# HUGGING_FACE_TOK_NAME = 'georgessand-ds-mini'
# tokenizer.push_to_hub(HUGGING_FACE_PSEUDO+"/"+ HUGGING_FACE_TOK_NAME)
# downloaded_tokenizer = AutoTokenizer.from_pretrained(HUGGING_FACE_PSEUDO+"/"+ HUGGING_FACE_TOK_NAME)
# print(downloaded_tokenizer.tokenize(txt))