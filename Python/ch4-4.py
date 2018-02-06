
import os
import sys

try:
    base_directory = os.path.split(sys.executable)[0]
    os.environ['PATH'] += ';' + base_directory
    import cntk
    os.environ['KERAS_BACKEND'] = 'cntk'
except ImportError:
    print('CNTK not installed')

import keras
import keras.utils
import keras.datasets
import keras.models
import keras.layers
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)
# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


num_epochs = 7
batch_size = 32

# L1 and L2 regularization at the same time
keras.regularizers.l1_l2(l1=0.001, l2=0.001)

original_model = keras.models.Sequential()
original_model.add(keras.layers.Dense(16, activation='relu', input_shape=(10000,)))
original_model.add(keras.layers.Dense(16, activation='relu'))
original_model.add(keras.layers.Dense(1, activation='sigmoid'))
original_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

original_hist = original_model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))


dpt_model = keras.models.Sequential()
dpt_model.add(keras.layers.Dense(16, activation='relu', input_shape=(10000,)))
dpt_model.add(keras.layers.Dropout(0.5))
dpt_model.add(keras.layers.Dense(16, activation='relu'))
dpt_model.add(keras.layers.Dropout(0.5))
dpt_model.add(keras.layers.Dense(1, activation='sigmoid'))
dpt_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

dpt_model_hist = dpt_model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))


epochs = range(1, num_epochs+1)
original_val_loss = original_hist.history['val_loss']
dpt_model_val_loss = dpt_model_hist.history['val_loss']

plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, dpt_model_val_loss, 'bo', label='Dropout-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()