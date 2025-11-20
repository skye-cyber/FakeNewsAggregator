import numpy as np
import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from arch import model_architecture

data = pd.read_csv("datasets/news.csv")
data = data.drop(["Unnamed: 0"], axis=1)

le = preprocessing.LabelEncoder()
le.fit(data['label'])
data['label'] = le.transform(data['label'])

embedding_dim = 50
max_length = 54
padding_type = 'post'
trunc_type = 'post'
oov_tok = "<OOV>"
training_size = 3000
test_portion = 0.1

title = []
text = []
labels = []


def train():
    try:
        for x in range(training_size):
            title.append(data['title'][x])
            text.append(data['text'][x])
            labels.append(data['label'][x])

        tokenizer1 = Tokenizer()
        tokenizer1.fit_on_texts(title)
        word_index1 = tokenizer1.word_index
        vocab_size1 = len(word_index1)
        sequences1 = tokenizer1.texts_to_sequences(title)
        padded1 = pad_sequences(sequences1, padding=padding_type, truncating=trunc_type)

        split = int(test_portion * training_size)
        training_sequences1 = padded1[split:training_size]
        test_sequences1 = padded1[0:split]
        test_labels = labels[0:split]
        training_labels = labels[split:training_size]

        training_sequences1 = np.array(training_sequences1)
        test_sequences1 = np.array(test_sequences1)

        embedding_index = {}
        with open('src/glove.6B.50d.txt', 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = coefs

        embedding_matrix = np.zeros((vocab_size1 + 1, embedding_dim))

        for word, i in word_index1.items():
            if i < vocab_size1:
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

        model = model_architecture(vocab_size1, embedding_matrix, embedding_dim, max_length)

        # Callbacks
        checkpoint = ModelCheckpoint(
            "checkpoints/model-best.h5",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )

        history = model.fit(
            training_sequences1,
            np.array(training_labels),
            epochs=50,
            validation_data=(test_sequences1, np.array(test_labels)),
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=2
        )

        model.save('models/model.keras')
        return history

    except Exception as e:
        print(f"\033[31mError: {str(e)}\033[0m")
        raise
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        return None


if __name__ == "__main__":
    train()
