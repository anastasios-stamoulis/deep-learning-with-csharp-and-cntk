
import os
import sys


use_cntk = True
if use_cntk:
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        base_directory = os.path.split(sys.executable)[0]
        os.environ['PATH'] += ';' + base_directory
        import cntk
        os.environ['KERAS_BACKEND'] = 'cntk'
    except ImportError:
        print('CNTK not installed')
else:
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import numpy as np
import time
import datetime


def plot_results(history):
    import matplotlib.pyplot as plt

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


def load_data(max_features, max_len):
    if os.path.isfile('ch6-4_x_train_imdb.bin'):
        return load_from_files()

    import keras
    import keras.datasets
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
    print('x_train shape:', x_train.shape, ', y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape, ', y_test shape:', y_test.shape)
    save_to_files(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_test, y_test


def build_model(max_features, max_len):
    import keras
    import keras.layers
    import keras.models
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(max_features, 128, input_length=max_len))
    model.add(keras.layers.Conv1D(32, 7, activation='relu'))
    model.add(keras.layers.MaxPooling1D(5))
    model.add(keras.layers.Conv1D(32, 7, activation='relu'))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(1))
    model.summary()
    return model


def implementing_1d_convnet():
    import keras
    import keras.optimizers

    max_features = 10000  # number of words to consider as features
    max_len = 500  # cut texts after this number of words (among top max_features most common words)
    x_train, y_train, x_test, y_test = load_data(max_features, max_len)
    model = build_model(max_features, max_len)

    model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
    return history


def save_to_files(x_train, y_train, x_test, y_test):
    x_train = np.ascontiguousarray(x_train.astype(np.float32))
    y_train = np.ascontiguousarray(y_train.astype(np.float32))
    x_test = np.ascontiguousarray(x_test.astype(np.float32))
    y_test = np.ascontiguousarray(y_test.astype(np.float32))
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train.tofile('ch6-4_x_train_imdb.bin')
    y_train.tofile('ch6-4_y_train_imdb.bin')
    x_test.tofile('ch6-4_x_test_imdb.bin')
    y_test.tofile('ch6-4_y_test_imdb.bin')


def load_from_files(x_shape=(25000, 500), y_shape=(25000,)):
    print('Loading .bin files')
    x_train = np.fromfile('ch6-4_x_train_imdb.bin', dtype=np.float32)
    y_train = np.fromfile('ch6-4_y_train_imdb.bin', dtype=np.float32)
    x_test = np.fromfile('ch6-4_x_test_imdb.bin', dtype=np.float32)
    y_test = np.fromfile('ch6-4_y_test_imdb.bin', dtype=np.float32)
    x_train = np.reshape(x_train, newshape=x_shape)
    y_train = np.reshape(y_train, newshape=y_shape)
    x_test = np.reshape(x_test, newshape=x_shape)
    y_test = np.reshape(y_test, newshape=y_shape)
    return x_train, y_train, x_test, y_test


def build_model_cntk(max_features, max_len):
    x = cntk.placeholder(shape=(max_len,), name='x_placeholder')
    l_0 = cntk.one_hot(x, num_classes=max_features, sparse_output=True)
    l_1_0 = cntk.layers.Embedding(128)(l_0)
    l_1_1 = cntk.transpose(l_1_0, (1, 0))
    l_2 = cntk.layers.Convolution1D(filter_shape=7, num_filters=32, activation=cntk.relu)(l_1_1)
    l_3 = cntk.layers.MaxPooling(filter_shape=(5,), strides=5)(l_2)
    l_4 = cntk.layers.Convolution1D(filter_shape=7, num_filters=32, activation=cntk.relu)(l_3)
    l_5 = cntk.layers.GlobalMaxPooling()(l_4)
    model = cntk.layers.Dense(shape=1, activation=cntk.sigmoid)(l_5)
    return model


def implementing_1d_convnet_cntk():
    max_features = 10000  # number of words to consider as features
    max_len = 500  # cut texts after this number of words (among top max_features most common words)
    x_train, y_train, x_test, y_test = load_data(max_features, max_len)

    model = build_model_cntk(max_features, max_len)
    x = cntk.input_variable(shape=(max_len,), dtype=np.float32)
    y = cntk.input_variable(shape=(1,), dtype=np.float32)
    model.replace_placeholders({model.placeholders[0]: x})

    loss_function = cntk.binary_cross_entropy(model.output, y)
    round_predictions = cntk.round(model.output)
    equal_elements = cntk.equal(round_predictions, y)
    accuracy_function = cntk.reduce_mean(equal_elements, axis=0)

    max_epochs = 10
    batch_size = 32
    learner = cntk.adam(model.parameters, cntk.learning_parameter_schedule_per_sample(0.0001), cntk.learning_parameter_schedule_per_sample(0.99))
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

        print('Epoch Elapsed Time: {0}, training_accuracy={1:.3f}, evaluation_accuracy={2:.3f}'.format(
            datetime.timedelta(seconds=time.time() - epoch_start_time),
            epoch_training_error, epoch_validation_error))


if __name__ == '__main__':
    results = implementing_1d_convnet()
    plot_results(results)
    # implementing_1d_convnet_cntk()
