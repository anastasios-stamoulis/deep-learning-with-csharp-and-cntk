import time
import datetime
import os
import sys
import numpy as np

use_cntk = True
if use_cntk:
    try:
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


def learning_word_embeddings_with_the_embedding_layer():
    # Number of words to consider as features
    max_features = 10000
    # Cut texts after this number of words
    # (among top max_features most common words)
    maxlen = 20

    # Load the data as lists of integers.
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)

    # This turns our lists of integers
    # into a 2D integer tensor of shape `(samples, maxlen)`
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    model = keras.models.Sequential()
    # We specify the maximum input length to our Embedding layer
    # so we can later flatten the embedded inputs
    model.add(keras.layers.Embedding(max_features, 8, input_length=maxlen))
    # After the Embedding layer,
    # our activations have shape `(samples, maxlen, 8)`.

    # We flatten the 3D tensor of embeddings
    # into a 2D tensor of shape `(samples, maxlen * 8)`
    model.add(keras.layers.Flatten())

    # We add the classifier on top
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


def learning_word_embeddings_with_the_embedding_layer_cntk():
    x_train, y_train, x_test, y_test = load_from_files()

    max_features = 10000
    maxlen = 20
    embedding_dim = 8

    x = cntk.input_variable(shape=(maxlen,), dtype=np.float32)
    y = cntk.input_variable(shape=(1,), dtype=np.float32)
    model = cntk.one_hot(x, num_classes=max_features, sparse_output=True)
    model = cntk.layers.Embedding(embedding_dim)(model)
    model = cntk.layers.Dense(1, activation=cntk.sigmoid)(model)
    loss_function = cntk.binary_cross_entropy(model.output, y)
    round_predictions = cntk.round(model.output)
    equal_elements = cntk.equal(round_predictions, y)
    accuracy_function = cntk.reduce_mean(equal_elements, axis=0)

    max_epochs = 30
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


def load_from_files(x_shape=(25000, 20), y_shape=(25000,)):
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


class Constants:
    maxlen = 100  # We will cut reviews after 100 words
    training_samples = 200  # We will be training on 200 samples
    validation_samples = 10000  # We will be validating on 10000 samples
    max_words = 10000  # We will only consider the top 10,000 words in the dataset
    embedding_dim = 100
    imdb_dir = 'C:\\Users\\anastasios\\Downloads\\aclImdb'


def load_texts_labels(path):
    import tqdm
    labels = []
    texts = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(path, label_type)
        print('\nLoading ', dir_name, '\n', flush=True)
        for fname in tqdm.tqdm(os.listdir(dir_name)):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname), encoding='utf8')
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    return texts, labels


def tokenize_alImdb():
    import keras.preprocessing.text
    train_dir = os.path.join(Constants.imdb_dir, 'train')
    texts, labels = load_texts_labels(train_dir)

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=Constants.max_words)
    print('\n\nRunning tokenizer...', end='', flush=True)
    tokenizer.fit_on_texts(texts)
    return tokenizer, texts, labels


def from_raw_text_to_word_embeddings():
    import numpy as np

    import keras.preprocessing.sequence
    tokenizer, texts, labels = tokenize_alImdb()

    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=Constants.maxlen)

    data = np.asarray(data, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # Split the data into a training set and a validation set
    # But first, shuffle the data, since we started from data
    # where sample are ordered (all negative first, then all positive).
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data[:Constants.training_samples]
    y_train = labels[:Constants.training_samples]
    x_val = data[Constants.training_samples: Constants.training_samples + Constants.validation_samples]
    y_val = labels[Constants.training_samples: Constants.training_samples + Constants.validation_samples]
    return tokenizer, x_train, y_train, x_val, y_val


def preprocess_embeddings():
    import numpy as np
    import tqdm

    glove_dir = 'C:\\Users\\anastasios\\Downloads\\glove.6B'

    embeddings_index = {}
    glove_path = os.path.join(glove_dir, 'glove.6B.100d.txt')
    f = open(glove_path, encoding='utf8')
    print('Processing ', glove_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(Constants.max_words, Constants.embedding_dim, input_length=Constants.maxlen))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    return model


def use_glove_word_embeddings_cntk(preload_weights=False):
    tokenizer, x_train, y_train, x_val, y_val = from_raw_text_to_word_embeddings()

    x = cntk.input_variable(shape=(Constants.maxlen,), dtype=np.float32)
    y = cntk.input_variable(shape=(1,), dtype=np.float32)
    model = cntk.one_hot(x, num_classes=Constants.max_words, sparse_output=True)
    if preload_weights is True:
        embedding_matrix = compute_embedding_matrix(tokenizer)
        assert (Constants.embedding_dim == embedding_matrix.shape[0]) or (Constants.embedding_dim == embedding_matrix.shape[1])
        model = cntk.layers.Embedding(weights=embedding_matrix)(model)
    else:
        model = cntk.layers.Embedding(Constants.embedding_dim)(model)
    model = cntk.layers.Dense(32, activation=cntk.relu)(model)
    model = cntk.layers.Dense(1, activation=cntk.sigmoid)(model)
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


def compute_embedding_matrix(tokenizer):
    embeddings_index = preprocess_embeddings()
    embedding_matrix = np.zeros((Constants.max_words, Constants.embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < Constants.max_words:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


def use_glove_word_embeddings(preload_weights=True):
    tokenizer, x_train, y_train, x_val, y_val = from_raw_text_to_word_embeddings()

    model = build_model()

    if preload_weights:
        embedding_matrix = compute_embedding_matrix(tokenizer)
        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_val, y_val))
    model.save_weights('pre_trained_glove_model.h5')
    plot_results(history)


def plot_results(history):
    import matplotlib.pyplot as plt

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

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


def evaluate_on_test_data():
    import numpy as np
    test_dir = os.path.join(Constants.imdb_dir, 'test')
    tokenizer, _, _ = tokenize_alImdb()
    texts, labels = load_texts_labels(test_dir)

    sequences = tokenizer.texts_to_sequences(texts)
    x_test = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=Constants.maxlen)
    y_test = np.asarray(labels)
    model = build_model()
    model.load_weights('pre_trained_glove_model.h5')
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    print(model.evaluate(x_test, y_test))


if __name__ == '__main__':
    learning_word_embeddings_with_the_embedding_layer()
    # learning_word_embeddings_with_the_embedding_layer_cntk()
    use_glove_word_embeddings(preload_weights=True)
    # use_glove_word_embeddings_cntk(preload_weights=True)
