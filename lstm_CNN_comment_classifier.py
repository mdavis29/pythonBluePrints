
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# check to see if the data is there
from subprocess import check_output
print(check_output(["ls", "input"]).decode("utf8"))
np.random.seed(0)

# load in the data
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

# this for the kaggle comp, six classes to classify the text as one of
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# extract a sentence list from the pandas data frame 'comment_text' section and fill NA with 'unkown'
list_sentences_test = test["comment_text"].fillna("unknown").values
list_sentences_train = train["comment_text"].fillna("unknown").values
y = train[list_classes].values

# set up parameters for the tokenizer (max_features, is max words considered, max_len will ne num cols the of array
max_features = 20000
maxlen = 100

# fits the tokenizer transform
tokenizer = text.Tokenizer(nb_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))

# train data
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
train_array = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)

# test data
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
test_array = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

# split the data into training and test sets for modeling fitting
X_train, X_test, y_train, y_test = train_test_split(train_array, y, test_size=0.10,)
print('Training:', X_train.shape)
print('Testing:', X_test.shape)

# define the model type and shape, LSTM with maxlen input cols, 1D CNN and six classification output
def get_model():
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(input=inp, output=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = get_model()
model.summary()

batch_size = 500
epochs = 1
file_path = "models/weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
callbacks_list = [checkpoint, early]
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.1, callbacks=callbacks_list)
model.load_weights(file_path)

# predict the kaggle test data
test = pd.read_csv("input/test.csv")
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
test_array = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
y_test_preds = model.predict(test_array)
submission = pd.DataFrame(y_test_preds, columns=list_classes)
submission['id'] = test['id']
submission.to_csv('submission.csv', index=False)