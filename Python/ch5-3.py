
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
import keras.applications
import keras.preprocessing.image
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np

base_dir = 'C:/Users/anastasios/Desktop/cats_and_dogs'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


def extract_features(directory, sample_count):
    conv_base = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    conv_base.summary()

    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    batch_size = 20

    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i % 4 == 0:
            print('{0}, processed {1} images'.format(directory, i*batch_size))
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels


def save_npy_files(features, labels, prefix):
    np.save(prefix+'_features.npy', features)
    np.save(prefix+'_labels', labels)


def load_npy_files(prefix):
    result = (np.load(prefix+'_features.npy'), np.load(prefix+'_labels.npy'))
    print('Loaded {0}_features.npy, {0}_labels.npy'.format(prefix))
    return result


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def train_with_extracted_features():
    if os.path.isfile('test_features.npy'):
        train_features, train_labels = load_npy_files('train')
        validation_features, validation_labels = load_npy_files('validation')
        test_features, test_labels = load_npy_files('test')
    else:
        train_features, train_labels = extract_features(train_dir, 2000)
        validation_features, validation_labels = extract_features(validation_dir, 1000)
        test_features, test_labels = extract_features(test_dir, 1000)

        train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
        validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
        test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

        save_npy_files(train_features, train_labels, 'train')
        save_npy_files(validation_features, validation_labels, 'validation')
        save_npy_files(test_features, test_labels, 'test')

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

    history = model.fit(train_features, train_labels, epochs=5, batch_size=20, validation_data=(validation_features, validation_labels))
    plot_history(history)


def train_with_augmentation(use_finetuning):
    conv_base = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

    model = keras.models.Sequential()
    model.add(conv_base)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))

    if use_finetuning:
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
    else:
        conv_base.trainable = False

    print('This is the number of trainable weights after freezing the conv base:', len(model.trainable_weights))
    model.summary()

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Note that the validation data should not be augmented!
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=2e-5), metrics=['acc'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50,
        verbose=2)

    plot_history(history)


if __name__ == '__main__':
    train_with_extracted_features()
    train_with_augmentation(use_finetuning=True)
