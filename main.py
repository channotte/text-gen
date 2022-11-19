import tensorflow as tf
from tensorflow import keras
import keras_nlp
import numpy as np
import random
from tensorflow.keras.layers import TextVectorization
from pathlib import Path

lelia_url = "https://www.gutenberg.org/files/39738/39738-0.txt"
la_petite_fadette_url = "https://www.gutenberg.org/cache/epub/34204/pg34204.txt"
gabriel_url = "https://www.gutenberg.org/cache/epub/13380/pg13380.txt"
lettre_voyageur_url = "https://www.gutenberg.org/files/37989/37989-0.txt"

paths = [lelia_url, la_petite_fadette_url, gabriel_url, lettre_voyageur_url]
names = ["Lélia", "La petite fadette", "Gabriel", "Lettre d'un voyageur"]
texts = ""

for index, path in enumerate(paths):
    filepath = keras.utils.get_file(f"{names[index]}.txt", origin=path)
    text = ""
    with open(filepath, encoding="utf-8") as f:
        text = f.read()
        # First 50 lines are the Gutenberg intro and preface
        # Skipping first 10k characters for each book should be approximately
        # removing the intros and prefaces.
        texts += text[10000:]
        print(text[10000:10100])

text_list = texts.split(".")
print(f"{len(text_list)} in text")

text_list = list(filter(None, text_list))
random.shuffle(text_list)

length = len(text_list)
text_train = text_list[: int(0.7 * length)]
text_test = text_list[int(0.7 * length) : int(0.85 * length)]
text_valid = text_list[int(0.85 * length) :]


def custom_standardization(input_string):
    sentence = tf.strings.lower(input_string)
    sentence = tf.strings.regex_replace(sentence, "\n", " ")
    return sentence


maxlen = 50
# You can also set calculate the longest sentence in the data - 25 in this case
# maxlen = len(max(text_list).split(' '))

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)

vectorize_layer.adapt(text_list)
vocab = vectorize_layer.get_vocabulary()

with open("vocab.txt", "w", encoding="utf-8") as fp:
    for word in vocab:
        # write each item on a new line
        fp.write(f"{word}\n")

vocab_size = len(vocab)
index_lookup = dict(zip(range(len(vocab)), vocab))

print(vectorize_layer(["le monde !"]))

batch_size = 64

train_dataset = tf.data.Dataset.from_tensor_slices(text_train)
train_dataset = train_dataset.shuffle(buffer_size=256)
train_dataset = train_dataset.batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices(text_test)
test_dataset = test_dataset.shuffle(buffer_size=256)
test_dataset = test_dataset.batch(batch_size)

valid_dataset = tf.data.Dataset.from_tensor_slices(text_valid)
valid_dataset = valid_dataset.shuffle(buffer_size=256)
valid_dataset = valid_dataset.batch(batch_size)


def preprocess_text(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


train_dataset = train_dataset.map(preprocess_text)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(preprocess_text)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

valid_dataset = valid_dataset.map(preprocess_text)
valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)

embed_dim = 128
num_heads = 4


def create_model():
    inputs = keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    x = keras_nlp.layers.TokenAndPositionEmbedding(vocab_size, maxlen, embed_dim)(
        inputs
    )
    for i in range(4):
        x = keras_nlp.layers.TransformerDecoder(
            intermediate_dim=embed_dim * 2, num_heads=num_heads, dropout=0.5
        )(x)
    do = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(vocab_size, activation="softmax")(do)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[keras_nlp.metrics.Perplexity(), "accuracy"],
    )
    return model


class TextSampler(keras.callbacks.Callback):
    def __init__(self, start_prompt, max_tokens):
        self.start_prompt = start_prompt
        self.max_tokens = max_tokens

    # Helper method to choose a word from the top K probable words with respect to their probabilities
    # in a sequence
    def sample_token(self, logits):
        logits, indices = tf.math.top_k(logits, k=5, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def on_epoch_end(self, epoch, logs=None):
        decoded_sample = self.start_prompt

        for i in range(self.max_tokens - 1):
            tokenized_prompt = vectorize_layer([decoded_sample])[:, :-1]
            predictions = self.model.predict([tokenized_prompt], verbose=0)
            # To find the index of the next word in the prediction array.
            # The tokenized prompt is already shorter than the original decoded sample
            # by one, len(decoded_sample.split()) is two words ahead - so we remove 1 to get
            # the next word in the sequence
            sample_index = len(decoded_sample.strip().split()) - 1

            sampled_token = self.sample_token(predictions[0][sample_index])
            sampled_token = index_lookup[sampled_token]
            decoded_sample += " " + sampled_token

        print(f"\nSample text:\n{decoded_sample}...\n")


# First 5 words of a random sentence to be used as a seed
random_sentence = " ".join(random.choice(text_valid).replace("\n", " ").split(" ")[:4])
sampler = TextSampler(random_sentence, 30)
reducelr = keras.callbacks.ReduceLROnPlateau(patience=10, monitor="val_loss")
model = create_model()
model.summary()
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=10,
    callbacks=[sampler, reducelr],
)


def sample_token(logits):
    logits, indices = tf.math.top_k(logits, k=5, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
    preds = np.asarray(preds).astype("float32")
    return np.random.choice(indices, p=preds)


def generate_text(prompt, response_length=20):
    decoded_sample = prompt
    for i in range(response_length - 1):
        tokenized_prompt = vectorize_layer([decoded_sample])[:, :-1]
        predictions = model.predict([tokenized_prompt], verbose=0)
        sample_index = len(decoded_sample.strip().split()) - 1

        sampled_token = sample_token(predictions[0][sample_index])
        sampled_token = index_lookup[sampled_token]
        decoded_sample += " " + sampled_token
    return decoded_sample


print(f"GENERATED : {generate_text('Cette jeune femme avait pour occupation ')}")
