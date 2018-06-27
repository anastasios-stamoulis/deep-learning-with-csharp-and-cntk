
import os
import sys

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
import cntk_util
import urllib
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime


def download_and_read_lines():
    # download https://www.bgc-jena.mpg.de/wetter/mpi_roof_2016b.zip
    url_format = 'https://www.bgc-jena.mpg.de/wetter/mpi_roof_{0}{1}.zip'
    csv_filenames = list()
    for year in range(2009, 2017):
        for c in ['a', 'b']:
            url = url_format.format(year, c)
            local_filename = url.split('/')[-1]
            local_filename = os.path.join('roof_data', local_filename)
            if os.path.isfile(local_filename) is False:
                print('Downloading {0} to {1}'.format(url, local_filename))
                urllib.request.urlretrieve(url, local_filename, reporthook=cntk_util.ShowProgress.show_progress_2)
            else:
                print('{0} exists'.format(local_filename))

            csv_filename = local_filename.replace('.zip', '.csv')
            csv_filenames.append(csv_filename)
            if os.path.isfile(csv_filename) is False:
                with zipfile.ZipFile(local_filename, 'r') as zip_ref:
                    print('Extracting {0}'.format(local_filename))
                    zip_ref.extractall()
            else:
                print('{0} exists'.format(csv_filename))

    all_lines = list()
    for csv_filename in csv_filenames:
        with open(csv_filename, 'r') as fd:
            file_lines = fd.readlines()[1:]
            for line in file_lines:
                comma_pos = line.index(',')
                all_lines.append(line[comma_pos+1:])
    return all_lines


def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6, reverse=False):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]), dtype=np.float32)
        targets = np.zeros((len(rows),), dtype=np.float32)
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        if reverse is True:
            yield samples[:, ::-1, :], targets
        else:
            yield samples, targets


def visualize_data(float_data):
    temp = float_data[:, 1]  # temperature (in degrees Celsius)
    plt.plot(range(len(temp)), temp)
    plt.show()
    plt.plot(range(1440), temp[:1440])
    plt.show()


def load_data():
    all_lines = download_and_read_lines()
    num_columns = len(all_lines[0].split(','))-1
    print('Num samples:', len(all_lines))
    float_data = np.zeros(shape=(len(all_lines), num_columns), dtype=np.float32)
    for i, line in enumerate(all_lines):
        values = [float(x) for x in line.split(',')[1:]]
        if len(values)<num_columns:
            print('over here')
        float_data[i, :] = values
    return float_data


def evaluate_naive_method(float_data):
    train_gen, val_gen, test_gen, val_steps, test_steps, lookback, step = create_generators(float_data)
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))


def create_generators(float_data):
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128

    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=200000,
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=300001,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)

    # This is how many steps to draw from `val_gen`
    # in order to see the whole validation set:
    val_steps = (300000 - 200001 - lookback) // batch_size

    # This is how many steps to draw from `test_gen`
    # in order to see the whole test set:
    test_steps = (len(float_data) - 300001 - lookback) // batch_size

    return train_gen, val_gen, test_gen, val_steps, test_steps, lookback, step


def plot_results(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def create_model(model_type, input_shape, dropout=0):
    model = keras.models.Sequential()
    if model_type == 0:
        model.add(keras.layers.Flatten(input_shape=input_shape))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(1))
    elif model_type == 1:
        model.add(keras.layers.GRU(32, dropout=dropout, recurrent_dropout=dropout, input_shape=input_shape))
        model.add(keras.layers.Dense(1))
        model.compile(optimizer=keras.optimizers.RMSprop(), loss='mae')
    elif model_type == 2:
        model.add(keras.layers.GRU(32, dropout=0.1, recurrent_dropout=0.5, return_sequences=True, input_shape=input_shape))
        model.add(keras.layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))
        model.add(keras.layers.Dense(1))
    else:
        raise NotImplementedError('model_type={0} not supported'.format(model_type))
    model.compile(optimizer=keras.optimizers.RMSprop(), loss='mae')
    return model


def create_model_cntk(model_type, input_shape, dropout=0, recurrent_dropout=0):
    x_placeholder = cntk.placeholder(shape=input_shape, dynamic_axes=[cntk.Axis.default_batch_axis(), cntk.Axis.default_dynamic_axis()], name='x_placeholder')
    model = x_placeholder
    if model_type == 0:
        model = cntk.layers.Dense(32, activation=cntk.relu)(model)
        model = cntk.layers.Dense(1)(model)
    elif model_type == 1:
        model = cntk.layers.Recurrence(cntk.layers.GRU(32))(model)
        model = cntk.sequence.last(model)
        if dropout > 0:
            model = cntk.layers.Dropout(dropout)(model)
        model = cntk.layers.Dense(1)(model)
    elif model_type == 2:
        model = cntk.layers.Recurrence(cntk.layers.GRU(32))(model)
        model = cntk.layers.Dropout(recurrent_dropout)(model)
        model = cntk.layers.Recurrence(cntk.layers.GRU(64, activation=cntk.relu))(model)
        model = cntk.sequence.last(model)
        model = cntk.layers.Dropout(dropout)(model)
        model = cntk.layers.Dense(1)(model)
    else:
        raise NotImplementedError('model_type={0} not supported'.format(model_type))
    return model


def basic_machine_learning_approach(float_data, epochs=20):
    train_gen, val_gen, test_gen, val_steps, test_steps, lookback, step = create_generators(float_data)
    model = create_model(model_type=0, input_shape=(lookback // step, float_data.shape[-1]))
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=epochs, validation_data=val_gen, validation_steps=val_steps)
    plot_results(history)


def basic_machine_learning_approach_cntk(float_data, epochs=20):
    train_gen, val_gen, test_gen, val_steps, test_steps, lookback, step = create_generators(float_data)

    input_shape = (lookback // step, float_data.shape[-1])
    model = create_model_cntk(model_type=0, input_shape=input_shape)

    x = cntk.input_variable(shape=input_shape, dtype=np.float32)
    model.replace_placeholders({model.placeholders[0]: x})

    y = cntk.input_variable(shape=(), dtype=np.float32)
    train_mse_cntk(x, y, model, train_gen, val_gen, epochs, val_steps)


def first_recurrent_baseline_cntk(float_data, dropout=0, epochs=20):
    train_gen, val_gen, test_gen, val_steps, test_steps, lookback, step = create_generators(float_data)

    input_shape = (float_data.shape[-1],)
    model = create_model_cntk(model_type=1, input_shape=input_shape, dropout=dropout)
    model_filename = 'ch6-3_model_type_1.model'
    model.save(model_filename)
    print('Saved ', model_filename)
    model = None
    model = cntk.load_model(model_filename)
    print('Loaded ', model_filename)

    x = cntk.input_variable(shape=input_shape, dtype=np.float32, dynamic_axes=[cntk.Axis.default_batch_axis(), cntk.Axis.default_dynamic_axis()])
    model.replace_placeholders({model.placeholders[0]: x})

    y = cntk.input_variable(shape=(), dtype=np.float32)
    train_mse_cntk(x, y, model, train_gen, val_gen, epochs, val_steps)


def stacking_recurrent_layers_cntk(float_data, epochs=20):
    train_gen, val_gen, test_gen, val_steps, test_steps, lookback, step = create_generators(float_data)

    input_shape = (float_data.shape[-1],)
    model = create_model_cntk(model_type=2, input_shape=input_shape, dropout=0.2, recurrent_dropout=0.5)
    model_filename = 'ch6-3_model_type_2.model'
    model.save(model_filename)
    print('Saved ', model_filename)
    model = None
    model = cntk.load_model(model_filename)
    print('Loaded ', model_filename)

    x = cntk.input_variable(shape=input_shape, dtype=np.float32, dynamic_axes=[cntk.Axis.default_batch_axis(), cntk.Axis.default_dynamic_axis()])
    model.replace_placeholders({model.placeholders[0]: x})

    y = cntk.input_variable(shape=(), dtype=np.float32)
    train_mse_cntk(x, y, model, train_gen, val_gen, epochs, val_steps)


def train_mse_cntk(x, y, model, train_gen, val_gen, epochs, val_steps):
    loss_function = cntk.squared_error(model, y)
    accuracy_function = loss_function
    learner = cntk.adam(model.parameters, cntk.learning_parameter_schedule_per_sample(0.001), cntk.learning_parameter_schedule_per_sample(0.9))
    trainer = cntk.Trainer(model, (loss_function, accuracy_function), [learner])
    evaluator = cntk.Evaluator(accuracy_function)

    history = fit_generator(x, y,
                            model=model,
                            trainer=trainer,
                            evaluator=evaluator,
                            train_gen=train_gen,
                            steps_per_epoch=500,
                            epochs=epochs,
                            val_gen=val_gen,
                            validation_steps=val_steps)

    plot_results(history)


def fit_generator(x, y, model, trainer, evaluator, train_gen, steps_per_epoch, epochs, val_gen, validation_steps):
    history = list()
    history.append(list())
    history.append(list())

    for current_epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_training_error = 0
        num_batches = 0
        for _ in range(steps_per_epoch):
            x_train_minibatch, y_train_minibatch = next(train_gen)
            trainer.train_minibatch({x: x_train_minibatch, y: y_train_minibatch})
            epoch_training_error += trainer.previous_minibatch_evaluation_average
            num_batches += 1
        epoch_training_error /= num_batches
        history[0].append(epoch_training_error)

        epoch_validation_error = 0
        num_batches = 0
        for _ in range(validation_steps):
            x_train_minibatch, y_train_minibatch = next(val_gen)
            previous_minibatch_evaluation_average = evaluator.test_minibatch({x: x_train_minibatch, y: y_train_minibatch})
            epoch_validation_error += previous_minibatch_evaluation_average
            num_batches += 1
        epoch_validation_error /= num_batches
        history[1].append(epoch_validation_error)

        print('Epoch {0}/{1}, elapsed time: {2}, training_accuracy={3:.3f}, evaluation_accuracy={4:.3f}'.format(
            current_epoch+1, epochs,
            datetime.timedelta(seconds=time.time() - epoch_start_time),
            epoch_training_error, epoch_validation_error))

    # https://stackoverflow.com/a/2827726
    h = type('', (), {})
    h.history = dict()
    h.history['loss'] = np.array(history[0])
    h.history['val_loss'] = np.array(history[1])
    return h


def first_recurrent_baseline(float_data, dropout=0, epochs=20):
    train_gen, val_gen, test_gen, val_steps, test_steps, lookback, step = create_generators(float_data)
    model = create_model(model_type=1, input_shape=(None, float_data.shape[-1]), dropout=dropout)
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=epochs, validation_data=val_gen, validation_steps=val_steps)
    plot_results(history)


def stacking_recurrent_layers(float_data, epochs=20):
    train_gen, val_gen, test_gen, val_steps, test_steps, lookback, step = create_generators(float_data)
    model = create_model(model_type=2, input_shape=(None, float_data.shape[-1]))
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=epochs, validation_data=val_gen, validation_steps=val_steps)
    plot_results(history)


def normalize_data(float_data):
    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std


def main():
    float_data = load_data()
    # visualize_data(float_data)
    normalize_data(float_data)

    epochs = 20
    # evaluate_naive_method(float_data)
    # basic_machine_learning_approach(float_data, epochs=epochs)
    # basic_machine_learning_approach_cntk(float_data, epochs=epochs)
    # first_recurrent_baseline(float_data, dropout=0.2, epochs=epochs)
    first_recurrent_baseline_cntk(float_data, dropout=0.2, epochs=epochs)
    # stacking_recurrent_layers(float_data, epochs=epochs)
    stacking_recurrent_layers_cntk(float_data, epochs=epochs)


if __name__ == '__main__':
    main()

