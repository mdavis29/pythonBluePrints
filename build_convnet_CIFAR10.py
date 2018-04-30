## Build CIFAR-10 with Convolutional Neural Net

from __future__ import print_function
import numpy as np
import keras
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
from tensorflow.python.client import device_lib
import os
import time
start = time.time()

# check to see if GPU is availible
device_lib.list_local_devices()

# For reproducibility
np.random.seed(1000)

# set log dir
log_dir = os.getcwd() + '//' + '.logs'
os.system('rm -R .logs')
os.system('mkdir .logs')

# setup tensorboard callback
tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

if __name__ == '__main__':
    # Load the dataset
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, decay=1e-6),
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train / 255.0, to_categorical(Y_train),
              batch_size=128,
              shuffle=True,
              epochs=5,
              validation_data=(X_test / 255.0, to_categorical(Y_test)),
              callbacks=[tensorboard, EarlyStopping(min_delta=0.001, patience=5)])

    # Evaluate the model
    scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))

    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])

print('build time in seconds : ', time.time()-start)

# save the model
model_file_name = 'models/convnet_v1.h5'
model.save(model_file_name)

# save only the weights
model_weights_name = 'models/convnet_v1_weights.h5'
model.save_weights(model_weights_name)


# load the model
model_loaded = keras.models.load_model(model_file_name)

# test prediction
preds = model_loaded.predict(X_test)

# get hidden layer output
from keras.models import Model
layer_name = 'dense_2'
final_dense_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
final_layer_output = final_dense_layer_model.predict(X_test)




