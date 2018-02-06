
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

# STEP 1: Load and Preprocess the Data
print('Keras Version:'+keras.__version__)
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = keras.utils.to_categorical(train_labels).astype('float32')
test_labels = keras.utils.to_categorical(test_labels).astype('float32')

# Save the pre-processed data to help with the C# port
train_images = np.ascontiguousarray(train_images)
train_labels = np.ascontiguousarray(train_labels)
test_images = np.ascontiguousarray(test_images)
test_labels = np.ascontiguousarray(test_labels)

train_images.tofile('train_images.bin')
train_labels.tofile('train_labels.bin')
test_images.tofile('test_images.bin')
test_labels.tofile('test_labels.bin')

# STEP 2: Create the Network
network = keras.models.Sequential()
network.add(keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(keras.layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 3: Train, and Evaluate the Network
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

