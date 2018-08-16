
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
import scipy
import matplotlib.pyplot as plt
import time
import datetime
import random

random.seed(2018)
np.random.seed(2018)


def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def preprocess_image(image_path, img_height, img_width):
    if use_keras is True:
        import keras.preprocessing
        import keras.applications
        import keras.backend
    else:
        import keras.backend
        keras.backend.set_image_data_format('channels_first')
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg19.preprocess_input(img)
    return img


def content_loss(base, combination):
    diff_ = combination - base
    square_ = keras.backend.square(diff_)
    sum_ = keras.backend.sum(square_)
    return sum_


def gram_matrix(x):
    # print('\n\nx=', keras.backend.int_shape(x))
    x_shape = keras.backend.int_shape(x)
    channels_first_ = keras.backend.permute_dimensions(x, (2, 0, 1))
    features = keras.backend.reshape(channels_first_, (x_shape[2], x_shape[0]*x_shape[1]))
    features_transposed = keras.backend.transpose(features)
    gram = keras.backend.dot(features, features_transposed)

    # import tensorflow as tf
    # channels_first_ = keras.backend.permute_dimensions(x, (2, 0, 1))
    # features = keras.backend.batch_flatten(channels_first_)
    # features = tf.Print(features, ['x=', tf.shape(x), 'channels_first_', tf.shape(channels_first_), 'features=', tf.shape(features)])
    # features_transposed = keras.backend.transpose(features)
    # gram = keras.backend.dot(features, features_transposed)

    return gram


def style_loss(style, combination, img_height, img_width):
    style_gram = gram_matrix(style)
    combination_gram = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    scaling_factor = (4. * (channels ** 2) * (size ** 2))
    square_ = keras.backend.square(style_gram - combination_gram)
    sum_ = keras.backend.sum(square_)
    result = sum_ / scaling_factor
    return result


def total_variation_loss(x, img_height, img_width):
    a_ = x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :]
    a = keras.backend.square(a_)

    b_ = x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :]
    b = keras.backend.square(b_)

    c = keras.backend.pow(a + b, 1.25)
    result = keras.backend.sum(c)
    return result


class Evaluator(object):

    def __init__(self, fetch_loss_and_grads, img_height, img_width):
        self.loss_value = None
        self.grad_values = None
        self.fetch_loss_and_grads = fetch_loss_and_grads
        self.img_height = img_height
        self.img_width = img_width

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, self.img_height, self.img_width, 3))
        outs = self.fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def plot_results(generated_image, target_image_path, style_reference_image_path, img_height, img_width):
    plt.imshow(keras.preprocessing.image.load_img(target_image_path, target_size=(img_height, img_width)))
    plt.figure()
    plt.imshow(keras.preprocessing.image.load_img(style_reference_image_path, target_size=(img_height, img_width)))
    plt.figure()
    plt.imshow(generated_image)
    plt.show()


def load_model(target_image_path, style_reference_image_path, img_height, img_width):
    target_image_ = preprocess_image(target_image_path, img_height, img_width)
    style_image_ = preprocess_image(style_reference_image_path, img_height, img_width)
    if use_keras:
        target_image = keras.backend.constant(target_image_)
        style_reference_image = keras.backend.constant(style_image_)
        combination_image = keras.backend.placeholder((1, img_height, img_width, 3))
        input_tensor = keras.backend.concatenate([target_image, style_reference_image, combination_image], axis=0)
        model = keras.applications.vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
        print('Model loaded.')
    else:
        target_image = cntk.constant(value=target_image_)
        style_reference_image = cntk.constant(value=style_image_)
        combination_image = cntk.placeholder(shape=(1, 3, img_height, img_width))
        input_tensor = cntk.ops.splice(target_image, style_reference_image, combination_image, axis=0)
        print(input_tensor.output.shape)
        quit()
    return model, combination_image


def create_loss_criterion(model, combination_image, img_height, img_width):
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    content_layer = 'block5_conv2'
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    total_variation_weight = 1e-4
    style_weight = 1.
    content_weight = 0.025

    loss = keras.backend.variable(0.)
    layer_features = outputs_dict[content_layer]
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(target_image_features, combination_features)
    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features, img_height, img_width)
        loss += (style_weight / len(style_layers)) * sl
    loss += total_variation_weight * total_variation_loss(combination_image, img_height, img_width)
    return loss


def style_transfer():
    training_start_time = time.time()
    target_image_path = '../DeepLearning/Ch_08_Neural_Style_Transfer/portrait.png'
    style_reference_image_path = '../DeepLearning/Ch_08_Neural_Style_Transfer/popova.png'
    img_height = 400
    img_width = 381
    model, combination_image = load_model(target_image_path, style_reference_image_path, img_height, img_width)
    img = run_keras(model, combination_image, target_image_path, img_height, img_width)
    print('Training Elapsed time: {0}'.format(datetime.timedelta(seconds=time.time() - training_start_time)))
    plot_results(img, target_image_path, style_reference_image_path, img_height, img_width)


def run_keras(model, combination_image, target_image_path, img_height, img_width):
    import scipy.optimize
    loss = create_loss_criterion(model, combination_image, img_height, img_width)
    grads = keras.backend.gradients(loss, combination_image)[0]
    fetch_loss_and_grads = keras.backend.function([combination_image], [loss, grads])
    evaluator = Evaluator(fetch_loss_and_grads, img_height, img_width)
    x = preprocess_image(target_image_path, img_height, img_width)
    x = x.flatten()

    for i in range(5):
        print('Start of iteration', i)
        iteration_start_time = time.time()
        x, min_val, info = scipy.optimize.fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        iteration_end_time = time.time()
        print('Iteration %d completed in %ds' % (i, iteration_end_time - iteration_start_time))
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    return img


if __name__ == '__main__':
    style_transfer()

