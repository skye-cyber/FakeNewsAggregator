import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the tokenizer used during training (you need to save this!)


def load_tokenizer(tokenizer_path='models/tokenizer.pkl'):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

# Or recreate it if you have the training data


def create_tokenizer_from_training_data():
    # You need to replicate the exact same preprocessing as training
    import pandas as pd
    data = pd.read_csv("datasets/news.csv")
    data = data.drop(["Unnamed: 0"], axis=1)

    training_size = 3000
    title = []
    for x in range(training_size):
        title.append(data['title'][x])

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(title)
    return tokenizer


def predict_news(title):
    # Parameters must match training exactly
    max_length = 54
    padding_type = 'post'
    trunc_type = 'post'
    oov_tok = "<OOV>"

    try:
        # Load model
        model = load_model('models/model.keras')

        # Load tokenizer (you need to save this during training!)
        tokenizer = load_tokenizer()  # Or use create_tokenizer_from_training_data()

        # Preprocess exactly like training
        sequences = tokenizer.texts_to_sequences([title])
        sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        # Predict
        prediction = model.predict(sequences, verbose=0)[0][0]

        print(f"Raw prediction: {prediction}")

        if prediction >= 0.5:
            result = "REAL"
        else:
            result = "FAKE"

        print(f"Prediction: {prediction:.4f} -> This news is {result}")
        return prediction, result

    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None


if __name__ == "__main__":
    test_titles = [
        "BREAKING: Shocking discovery reveals that drinking coffee makes you immune to COVID-19! Doctors don't want you to know this simple trick that's going viral worldwide!",
        "President announces new economic policy to address inflation concerns",
        "Scientists discover new species in Amazon rainforest",
        "ALIENS LAND IN TIMES SQUARE: Government covering up extraterrestrial visit that happened last night!"
    ]

    for title in test_titles:
        print(f"\nTesting: {title[:80]}...")
        predict_news(title)
        print("-" * 50)
