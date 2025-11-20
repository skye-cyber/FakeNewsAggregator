import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from arch import model_architecture
from sklearn.utils import class_weight

data = pd.read_csv("datasets/news.csv")
data = data.drop(["Unnamed: 0"], axis=1)

le = preprocessing.LabelEncoder()
le.fit(data['label'])

# print("Class distribution:", data['label'].value_counts())
# shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data['label'] = le.transform(data['label'])

embedding_dim = 50
max_length = 54
padding_type = 'post'
trunc_type = 'post'
oov_tok = "<OOV>"
training_size = min(3000, len(data))
test_portion = 0.1

titles = []
text = []
labels = []


def find_learning_rate(model, train_data, train_labels, start_lr=1e-6, end_lr=1e-1, epochs=5):
    """Find optimal learning rate"""
    original_weights = model.get_weights()

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: start_lr * 10**(epoch / epochs * np.log10(end_lr / start_lr))
    )

    history = model.fit(
        train_data[:1000],  # Use subset for speed
        train_labels[:1000],
        epochs=epochs,
        batch_size=32,
        callbacks=[lr_schedule],
        verbose=0
    )

    # Reset model weights
    model.set_weights(original_weights)
    return history


def train():
    global titles, labels
    try:
        for x in range(training_size):
            titles.append(data['title'][x])
            text.append(data['text'][x])
            labels.append(data['label'][x])

        # Convert to numpy arrays for proper shuffling
        titles = np.array(titles)
        labels = np.array(labels)

        # Shuffle the data PROPERLY
        indices = np.random.permutation(len(titles))
        titles = titles[indices]
        labels = labels[indices]

        # print(f"First 10 labels: {labels[:100]}")
        # print(f"Label distribution: {np.unique(labels, return_counts=True)}")

        tokenizer1 = Tokenizer(oov_token=oov_tok)
        tokenizer1.fit_on_texts(titles)
        word_index1 = tokenizer1.word_index
        vocab_size1 = len(word_index1)
        sequences1 = tokenizer1.texts_to_sequences(titles)

        with open('models/tokenizer.pkl', 'wb') as handle:
            pickle.dump(tokenizer1, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Ensure padding uses the correct max_length
        padded1 = pad_sequences(sequences1, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        # Proper splitting
        training_sequences1, test_sequences1, training_labels, test_labels = train_test_split(
            padded1, labels, test_size=test_portion, random_state=42, stratify=labels
        )

        training_sequences1 = np.array(training_sequences1)
        test_sequences1 = np.array(test_sequences1)

        # Load embeddings
        embedding_index = {}
        with open('src/glove.6B.50d.txt', 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = coefs

        embedding_matrix = np.zeros((vocab_size1 + 1, embedding_dim))
        for word, i in word_index1.items():
            if i <= vocab_size1:  # Changed from < to <=
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

        # Build model
        model = model_architecture(vocab_size1, embedding_matrix, embedding_dim, max_length)

        # FORCE MODEL TO BUILD
        print("\n=== BUILDING MODEL ===")
        dummy_input = tf.constant(training_sequences1[:2], dtype=tf.int32)
        _ = model(dummy_input)  # This forces all layers to build

        print("\n=== MODEL SUMMARY AFTER BUILDING ===")
        model.summary()

        # Test predictions
        print("\n=== TESTING PREDICTIONS ===")
        sample_pred = model.predict(training_sequences1[:5], verbose=0)
        print(f"Sample predictions: {sample_pred.flatten()}")

        # Callbacks
        checkpoint = ModelCheckpoint(
            "checkpoints/model-best.h5",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            mode="max"
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7
        )

        # Calculate class weights
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(training_labels),
            y=training_labels
        )
        class_weight_dict = dict(enumerate(class_weights))

        print("\n=== STARTING TRAINING ===")
        print(f"Class weights: {class_weight_dict}")

        class DetailedMetrics(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                if epoch == 0:
                    # Check initial predictions
                    initial_pred = self.model.predict(training_sequences1[:10], verbose=0)
                    print(f"\nInitial predictions: {initial_pred.flatten()}")

            def on_epoch_end(self, epoch, logs=None):
                if epoch < 5 or epoch % 10 == 0:
                    # Check predictions and gradients
                    sample_pred = self.model.predict(training_sequences1[:5], verbose=0)
                    pred_std = np.std(sample_pred)
                    print(f"Epoch {epoch + 1}: loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}, "
                          f"val_loss={logs['val_loss']:.4f}, pred_std={pred_std:.4f}")

                    # Check if predictions are diversifying
                    if pred_std < 0.01:
                        print("WARNING: Predictions are not diversifying!")

        detailed_metrics = DetailedMetrics()

        history = model.fit(
            training_sequences1,
            np.array(training_labels),
            epochs=50,
            batch_size=32,
            validation_data=(test_sequences1, np.array(test_labels)),
            callbacks=[checkpoint, early_stop, reduce_lr, detailed_metrics],
            class_weight=class_weight_dict,
            verbose=1
        )

        model.save('models/model.keras')
        print("Training completed successfully!")
        return history

    except Exception as e:
        print(f"\033[31mError: {str(e)}\033[0m")
        import traceback
        traceback.print_exc()
        raise
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        return None


if __name__ == "__main__":
    train()
