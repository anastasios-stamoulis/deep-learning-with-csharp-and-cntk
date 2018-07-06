
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

use_keras = True
if use_keras:
    use_cntk = False
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
else:
    import cntk


import numpy as np
import random
import time
import datetime

random.seed(2018)
np.random.seed(2018)

# Length of extracted character sequences
maxlen = 60


def get_text():
    filename = 'nietzsche.txt'
    if os.path.isfile(filename) is False:
        import keras
        path = keras.utils.get_file(filename, cache_subdir='.', cache_dir='.', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        assert os.path.basename(path)==filename
    else:
        path = filename
    text = open(path).read().lower()
    print('Corpus length:', len(text))
    return text


def get_data(one_hot_encode_features=True):
    text = get_text()
    # We sample a new sequence every `step` characters
    step = 3

    # This holds our extracted sequences
    sentences = []

    # This holds the targets (the follow-up characters)
    next_chars = []

    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('Number of sequences:', len(sentences))

    # List of unique characters in the corpus
    chars = sorted(list(set(text)))
    print('Unique characters:', len(chars))
    # Dictionary mapping unique characters to their index in `chars`
    char_indices = dict((char, chars.index(char)) for char in chars)

    # Next, one-hot encode the characters into binary arrays.
    print('Vectorization...')
    if one_hot_encode_features:
        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    else:
        x = np.zeros((len(sentences), maxlen), dtype=np.float32)
        y = np.zeros((len(sentences),), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            if one_hot_encode_features:
                x[i, t, char_indices[char]] = 1
            else:
                x[i, t] = char_indices[char]
        y[i] = char_indices[next_chars[i]]

    return text, chars, char_indices, x, y


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    new_predictions = np.random.multinomial(1, preds, 1)
    return np.argmax(new_predictions)


def build_model_cntk(alphabet_size):
    x_placeholder = cntk.placeholder(shape=(), dynamic_axes=[cntk.Axis.default_batch_axis(), cntk.Axis.default_dynamic_axis()])
    model = cntk.one_hot(x_placeholder, num_classes=alphabet_size, sparse_output=True)
    model = cntk.layers.Recurrence(cntk.layers.LSTM(128))(model)
    model = cntk.sequence.last(model)
    model = cntk.layers.Dense(alphabet_size)(model)
    return model


def build_model(alphabet_size):
    import keras
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(128, input_shape=(maxlen, alphabet_size)))
    model.add(keras.layers.Dense(alphabet_size, activation='softmax'))
    optimizer = keras.optimizers.RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


def run_cntk():
    text, chars, char_indices, x_train, y_train = get_data(one_hot_encode_features=False)
    alphabet_size = len(chars)
    print('alphabet_size=', alphabet_size)
    model = build_model_cntk(alphabet_size=alphabet_size)
    model_filename = 'ch8-1_cntk.model'
    model.save(model_filename)
    model = None
    model = cntk.load_model(model_filename)

    x = cntk.sequence.input_variable(shape=(), dtype=np.float32)
    y = cntk.input_variable(shape=(), dtype=np.float32)
    model.replace_placeholders({model.placeholders[0]: x})

    y_oneHot = cntk.one_hot(y, num_classes=alphabet_size)
    loss_function = cntk.cross_entropy_with_softmax(model.output, y_oneHot)
    learner = cntk.adam(model.parameters, cntk.learning_parameter_schedule_per_sample(0.001), cntk.learning_parameter_schedule_per_sample(0.9))
    trainer = cntk.Trainer(model, (loss_function, loss_function), [learner],)

    for epoch in range(1, 60):
        print('epoch', epoch)
        cntk_train(x, y, x_train, y_train, max_epochs=32, batch_size=128, trainer=trainer)
        model_filename = 'final_ch8-1_cntk.model'
        model.save(model_filename)
        generate_text_cntk(char_indices, chars, model, text)


def softmax(x):
    # https://stackoverflow.com/q/34968722
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def generate_text_cntk(char_indices, chars, model, text):
    # Select a text seed at random
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        # We generate 400 characters
        for i in range(400):
            sampled = np.zeros(shape=(1, maxlen), dtype=np.float32)
            for t, char in enumerate(generated_text):
                sampled[0, t] = char_indices[char]

            preds = model.eval({model.arguments[0]: sampled})[0]
            preds = softmax(preds)
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


def cntk_train(x, y, train_features, train_labels, max_epochs, batch_size, trainer):
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

        print('Epoch {0}/{1}, elapsed time: {2}, training_loss={3:.3f}'.format(
            current_epoch+1, max_epochs, datetime.timedelta(seconds=time.time() - epoch_start_time),
            epoch_training_error))


def run():
    text, chars, char_indices, x, y = get_data()
    model = build_model(alphabet_size=len(chars))

    for epoch in range(1, 60):
        print('epoch', epoch)
        model.fit(x, y, batch_size=128, epochs=1)
        generate_text(char_indices, chars, model, text)


def generate_text(char_indices, chars, model, text):
    # Select a text seed at random
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        # We generate 400 characters
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


if __name__ == '__main__':
    run()
    # run_cntk()
