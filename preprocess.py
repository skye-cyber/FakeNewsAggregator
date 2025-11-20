import pandas as pd
from sklearn import preprocessing


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
