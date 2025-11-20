from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer


max_length = 54
padding_type = 'post'
trunc_type = 'post'

X = "Karry to go to France in gesture of sympathy"


tokenizer1 = Tokenizer()
# tokenizer1.fit_on_texts(['example sentence'])
sequences = tokenizer1.texts_to_sequences([X])

model = load_model('models/model.keras')

sequences = tokenizer1.texts_to_sequences([X])
sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
if model.predict(sequences, verbose=0)[0][0] >= 0.5:
    print("This news is True")
else:
    print("This news is False")
