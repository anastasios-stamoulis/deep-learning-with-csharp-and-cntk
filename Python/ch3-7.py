
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


def build_model(input_dim):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()

mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std

x_test -= mean
x_test /= std

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(x_test.shape)

if True:
    x_train = np.ascontiguousarray(x_train, dtype=np.float32)
    y_train = np.ascontiguousarray(y_train, dtype=np.float32)
    x_train.tofile('x_train.bin')
    y_train.tofile('y_train.bin')
    x_test = np.ascontiguousarray(x_test, dtype=np.float32)
    y_test = np.ascontiguousarray(y_test, dtype=np.float32)
    x_test.tofile('x_test.bin')
    y_test.tofile('y_test.bin')
    quit()

k = 4
num_val_samples = len(x_train) // k
num_epochs = 20
all_scores = list()
all_mae_histories = list()
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate([x_train[:i * num_val_samples], x_train[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([y_train[:i * num_val_samples], y_train[(i + 1) * num_val_samples:]], axis=0)

    model = build_model(x_train.shape[1])
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, validation_data=(val_data, val_targets), batch_size=16)
    all_mae_histories.append(history.history['val_mean_absolute_error'])

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

