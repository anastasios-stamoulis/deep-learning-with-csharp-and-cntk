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

latent_dim = 32
height = 32
width = 32
channels = 3


def create_generator():
    generator_input = keras.Input(shape=(latent_dim,))

    # First, transform the input into a 16x16 128-channels feature map
    x = keras.layers.Dense(128 * 16 * 16)(generator_input)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Reshape((16, 16, 128))(x)

    # Then, add a convolution layer
    x = keras.layers.Conv2D(256, 5, padding='same')(x)
    x = keras.layers.LeakyReLU()(x)

    # Upsample to 32x32
    x = keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU()(x)

    # Few more conv layers
    x = keras.layers.Conv2D(256, 5, padding='same')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(256, 5, padding='same')(x)
    x = keras.layers.LeakyReLU()(x)

    # Produce a 32x32 1-channel feature map
    x = keras.layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
    generator = keras.models.Model(generator_input, x)
    generator.summary()
    return generator


def create_discriminator():
    discriminator_input = keras.layers.Input(shape=(height, width, channels))
    x = keras.layers.Conv2D(128, 3)(discriminator_input)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(128, 4, strides=2)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(128, 4, strides=2)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(128, 4, strides=2)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)

    # One dropout layer - important trick!
    x = keras.layers.Dropout(0.4)(x)

    # Classification layer
    x = keras.layers.Dense(1, activation='sigmoid')(x)

    discriminator = keras.models.Model(discriminator_input, x)
    discriminator.summary()

    # To stabilize training, we use learning rate decay
    # and gradient clipping (by value) in the optimizer.
    discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
    discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

    # Set discriminator weights to non-trainable# Set d
    # (will only apply to the `gan` model)
    discriminator.trainable = False
    return discriminator


def create_gan():
    generator = create_generator()
    discriminator = create_discriminator()

    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)

    gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
    gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
    return generator, discriminator, gan


def load_data():
    # Load CIFAR10 data
    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()

    # Select frog images (class 6)
    x_train = x_train[y_train.flatten() == 6]

    # Normalize data
    x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.
    x_channels_first = np.transpose(x_train, (0, 3, 1, 2))
    x_channels_first = np.ascontiguousarray(x_channels_first, dtype=np.float32)
    x_channels_first.tofile('x_channels_first_8_5.bin')
    print('Saved x_channels_first_8_5.bin')
    return x_train


def train():
    generator, discriminator, gan = create_gan()
    x_train = load_data()

    iterations = 10000
    batch_size = 20
    save_dir = '.'

    # Start training loop
    start = 0
    for step in range(iterations):
        # Sample random points in the latent space
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        # Decode them to fake images
        generated_images = generator.predict(random_latent_vectors)

        # Combine them with real images
        stop = start + batch_size
        real_images = x_train[start: stop]
        combined_images = np.concatenate([generated_images, real_images])

        # Assemble labels discriminating real from fake images
        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        # Add random noise to the labels - important trick!
        labels += 0.05 * np.random.random(labels.shape)

        # Train the discriminator
        d_loss = discriminator.train_on_batch(combined_images, labels)

        # sample random points in the latent space
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        # Assemble labels that say "all real images"
        misleading_targets = np.zeros((batch_size, 1))

        # Train the generator (via the gan model,
        # where the discriminator weights are frozen)
        a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

        start += batch_size
        if start > len(x_train) - batch_size:
            start = 0

        # Occasionally save / plot
        if step % 100 == 0:
            # Save model weights
            gan.save_weights('gan.h5')

            # Print metrics
            print('discriminator loss at step %s: %s' % (step, d_loss))
            print('adversarial loss at step %s: %s' % (step, a_loss))

            # Save one generated image
            img = keras.preprocessing.image.array_to_img(generated_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))

            # Save one real image, for comparison
            img = keras.preprocessing.image.array_to_img(real_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))

    print('Saving models...', end='')
    generator.save('generator_8_4.h5')
    discriminator.save('discriminator_8_4.h5')
    gan.save('gan_8_4.h5')
    print('done.')


def plot_results():
    import matplotlib.pyplot as plt
    generator = keras.models.load_model('generator_8_4.h5')

    # Sample random points in the latent space
    random_latent_vectors = np.random.normal(size=(10, latent_dim))

    # Decode them to fake images
    generated_images = generator.predict(random_latent_vectors)

    for i in range(generated_images.shape[0]):
        img = keras.preprocessing.image.array_to_img(generated_images[i] * 255., scale=False)
        plt.figure()
        plt.imshow(img)

    plt.show()


if __name__ == '__main__':
    train()
    plot_results()
