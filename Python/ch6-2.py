
import os
import sys
import numpy as np
import time
import datetime

use_cntk = True
if use_cntk:
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        base_directory = os.path.split(sys.executable)[0]
        os.environ['PATH'] += ';' + base_directory
        import cntk
        os.environ['KERAS_BACKEND'] = 'cntk'
    except ImportError:
        print('CNTK not installed')
else:
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import keras


def run_experiment(use_simple_rnn=True):
    import matplotlib.pyplot as plt

    max_features = 10000  # number of words to consider as features
    maxlen = 500  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32

    print('Loading data...')
    (input_train, y_train), (input_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
    print(len(input_train), 'train sequences')
    print(len(input_test), 'test sequences')

    print('Pad sequences (samples x time)')
    input_train = keras.preprocessing.sequence.pad_sequences(input_train, maxlen=maxlen)
    input_test = keras.preprocessing.sequence.pad_sequences(input_test, maxlen=maxlen)
    print('input_train shape:', input_train.shape)
    print('input_test shape:', input_test.shape)

    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(max_features, 32))
    if use_simple_rnn:
        model.add(keras.layers.SimpleRNN(32))
    else:
        model.add(keras.layers.LSTM(32))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

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


class Constants:
    maxlen = 500
    max_words = 10000  # We will only consider the top 10,000 words in the dataset
    embedding_dim = 32


def save_to_files(x_train, y_train, x_test, y_test):
    x_train = np.ascontiguousarray(x_train.astype(np.float32))
    y_train = np.ascontiguousarray(y_train.astype(np.float32))
    x_test = np.ascontiguousarray(x_test.astype(np.float32))
    y_test = np.ascontiguousarray(y_test.astype(np.float32))
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train.tofile('x_train_imdb.bin')
    y_train.tofile('y_train_imdb.bin')
    x_test.tofile('x_test_imdb.bin')
    y_test.tofile('y_test_imdb.bin')


def load_from_files(x_shape, y_shape):
    print('Loading .bin files')
    x_train = np.fromfile('x_train_imdb.bin', dtype=np.float32)
    y_train = np.fromfile('y_train_imdb.bin', dtype=np.float32)
    x_test = np.fromfile('x_test_imdb.bin', dtype=np.float32)
    y_test = np.fromfile('y_test_imdb.bin', dtype=np.float32)
    x_train = np.reshape(x_train, newshape=x_shape)
    y_train = np.reshape(y_train, newshape=y_shape)
    x_test = np.reshape(x_test, newshape=x_shape)
    y_test = np.reshape(y_test, newshape=y_shape)
    return x_train, y_train, x_test, y_test


def run_experiment_cntk():
    if os.path.isfile('x_train_imdb.bin'):
        print('Loading from .bin files')
        x_train, y_train, x_test, y_test = load_from_files(x_shape=(25000, 500), y_shape=(25000,))
    else:
        print('Loading data...')
        (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=Constants.max_words)
        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')

        print('Pad sequences (samples x time)')
        x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=Constants.maxlen)
        x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=Constants.maxlen)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        print('Saving to .bin files')
        save_to_files(x_train, y_train, x_test, y_test)

    x = cntk.sequence.input_variable(shape=(), dtype=np.float32)
    y = cntk.input_variable(shape=(), dtype=np.float32)
    x_placeholder = cntk.placeholder(shape=(), dynamic_axes=[cntk.Axis.default_batch_axis(), cntk.Axis.default_dynamic_axis()])

    model = cntk.one_hot(x_placeholder, num_classes=Constants.max_words, sparse_output=True)
    model = cntk.layers.Embedding(Constants.embedding_dim)(model)
    model = cntk.layers.Recurrence(cntk.layers.LSTM(32))(model)
    model = cntk.sequence.last(model)
    model = cntk.layers.Dense(1, activation=cntk.sigmoid)(model)
    model.save('ch6-2.cntk.model')
    model = None
    model = cntk.load_model('ch6-2.cntk.model')
    model.replace_placeholders({model.placeholders[0]: x})

    loss_function = cntk.binary_cross_entropy(model.output, y)
    round_predictions = cntk.round(model.output)
    equal_elements = cntk.equal(round_predictions, y)
    accuracy_function = cntk.reduce_mean(equal_elements, axis=cntk.Axis.all_static_axes())

    max_epochs = 10
    batch_size = 128
    learner = cntk.adam(model.parameters, cntk.learning_parameter_schedule_per_sample(0.01), cntk.learning_parameter_schedule_per_sample(0.9))
    progress_printer = cntk.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    trainer = cntk.Trainer(model, (loss_function, accuracy_function), [learner], progress_printer)
    evaluator = cntk.Evaluator(accuracy_function)

    cntk_train(x, y, x_train, y_train, max_epochs, batch_size, trainer, evaluator)


def cntk_train(x, y, x_train, y_train, max_epochs, batch_size, trainer, evaluator):
    N = len(x_train)
    y_train = np.expand_dims(y_train, axis=1)
    train_features = x_train[:int(N*0.8)]
    train_labels = y_train[:int(N*0.8)]
    validation_features = x_train[int(N*0.8):]
    validation_labels = y_train[int(N*0.8):]

    for current_epoch in range(max_epochs):
        epoch_start_time = time.time()
        train_indices = np.random.permutation(train_features.shape[0])
        pos = 0
        epoch_training_error = 0
        num_batches = 0
        while pos < len(train_indices):
            pos_end = min(pos + batch_size, len(train_indices))
            x_train_minibatch = train_features[train_indices[pos:pos_end]]
            y_train_minibatch = train_labels[train_indices[pos:pos_end]]
            trainer.train_minibatch({x: x_train_minibatch, y: y_train_minibatch})
            epoch_training_error += trainer.previous_minibatch_evaluation_average
            num_batches += 1
            pos = pos_end
        epoch_training_error /= num_batches

        epoch_validation_error = 0
        num_batches = 0
        pos = 0
        while pos < len(validation_features):
            pos_end = min(pos + batch_size, len(validation_features))
            x_train_minibatch = validation_features[pos:pos_end]
            y_train_minibatch = validation_labels[pos:pos_end]
            previous_minibatch_evaluation_average = evaluator.test_minibatch({x: x_train_minibatch, y: y_train_minibatch})
            epoch_validation_error += previous_minibatch_evaluation_average
            num_batches += 1
            pos = pos_end
        epoch_validation_error /= num_batches

        print('Epoch {0}/{1}, elapsed time: {2}, training_accuracy={3:.3f}, evaluation_accuracy={4:.3f}'.format(
            current_epoch+1, max_epochs,
            datetime.timedelta(seconds=time.time() - epoch_start_time),
            epoch_training_error, epoch_validation_error))



if __name__ == '__main__':
    # run_experiment(use_simple_rnn=False)
    run_experiment_cntk()