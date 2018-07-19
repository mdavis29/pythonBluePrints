
from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras_preprocessing.text import Tokenizer

# set parameters:
num_words = 10
embedded_dims = 5
maxlen = 5
batch_size = 100
filters = 10
kernel_size = 2
hidden_dims = 2
epochs = 20

# create data that represents it taking longer to travel home from work, than to travel to
docs = ['home daycare work', 'home daycare work', 'work daycare home', 'work daycare home', 'home work', 'work home', 'home work', 'work home']
docs_eos = list(map(lambda x: '<go> ' + x, docs))
docs_eos = list(map(lambda x: x + ' <eos>', docs_eos))
y = [35, 34, 47, 46, 18,  23, 19, 26]

# set up the word tokenizer
# TO DO adda a stemmer some home
t = Tokenizer(num_words=10,  oov_token='<oov>')
t.fit_on_texts(docs_eos)

# extract a work sequence
x_seqs = t.texts_to_sequences(docs_eos)

# padding seqs
x_padded = sequence.pad_sequences(x_seqs, maxlen=maxlen)

# initialized the model
model = Sequential()

# our vocab indices into embedding_dims dimensions
model.add(Embedding(num_words,
                    embedded_dims,
                    input_length=maxlen))

# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])
model.fit(x_padded, y,
          epochs=epochs)
print('completed')


def predict(docs, maxlen=maxlen):
    docs_eos = list(map(lambda x: '<go> ' + x, docs))
    docs_eos = list(map(lambda x: x + ' <eos>', docs_eos))
    docs_seq = t.texts_to_sequences(docs_eos)
    x_padded = sequence.pad_sequences(docs_seq, maxlen=maxlen)
    return model.predict(x_padded)


test_data = ['home daycare work', 'work daycare home']

print(predict(test_data))