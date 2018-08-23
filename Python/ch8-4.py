
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import time

start_time = time.time()

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
    import keras
else:
    import cntk
    import cntk.ops.functions

import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(2018)
np.random.seed(2018)

img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 2  # Dimensionality of the latent space: a plane


def sampling(args):
    z_mean, z_log_var = args
    epsilon = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + keras.backend.exp(z_log_var) * epsilon


class CustomVariationalLayer(keras.layers.Layer):

    @staticmethod
    def vae_loss(x, z_decoded, z_mean, z_log_var):
        x = keras.backend.flatten(x)
        z_decoded = keras.backend.flatten(z_decoded)
        crossentropy_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        square_ = keras.backend.square(z_mean)
        exp_ = keras.backend.exp(z_log_var)
        diff_ = 1 + z_log_var - square_ - exp_
        kl_loss = -5e-4 * keras.backend.mean(diff_, axis=-1)
        return keras.backend.mean(crossentropy_loss + kl_loss)

    def call(self, inputs):
        x, z_decoded, z_mean, z_log_var = inputs
        loss = CustomVariationalLayer.vae_loss(x, z_decoded, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        return x


def run():
    input_img = keras.Input(shape=img_shape)

    # encoder
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    shape_before_flattening = keras.backend.int_shape(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(32, activation='relu')(x)

    # latent variables
    z_mean = keras.layers.Dense(latent_dim)(x)
    z_log_var = keras.layers.Dense(latent_dim)(x)
    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    # decoder
    decoder_input = keras.layers.Input(keras.backend.int_shape(z)[1:])
    x = keras.layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
    x = keras.layers.Reshape(shape_before_flattening[1:])(x)
    x = keras.layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    decoder = keras.models.Model(decoder_input, x, name='decoder')
    decoder.summary()

    # overall model
    z_decoded = decoder(z)
    y = CustomVariationalLayer()([input_img, z_decoded, z_mean, z_log_var])
    vae = keras.models.Model(input_img, y, name='vae_model')
    vae.compile(optimizer='rmsprop', loss=None)
    vae.summary()

    # Train the VAE on MNIST digits
    (x_train, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))

    vae.fit(x=x_train, y=None,
            shuffle=True,
            epochs=20,
            batch_size=batch_size,
            validation_data=(x_test, None))

    plot_results(decoder, batch_size)


def plot_results(decoder, batch_size):
    import scipy.stats

    # Display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # Linearly spaced coordinates on the unit square were transformed
    # through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z,
    # since the prior of the latent space is Gaussian
    grid_x = scipy.stats.norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = scipy.stats.norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
            x_decoded = decoder.predict(z_sample, batch_size=batch_size)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


if __name__=='__main__':
    run()