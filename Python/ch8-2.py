
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import time

try:
    import cntk_util
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))
    import cntk_util

start_time = time.time()

use_keras = False
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
import random
import scipy
import matplotlib.pyplot as plt
import time
import datetime

random.seed(2018)
np.random.seed(2018)


def get_response(msg):
    print('Python received: ', msg)
    return 'Hi From Python'


def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)


def preprocess_image(image_path):
    img = keras.preprocessing.image.load_img(image_path)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    if keras.backend.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def eval_loss_and_grads(x, fetch_loss_and_grads):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def gradient_ascent(fetch_loss_and_grads, x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x, fetch_loss_and_grads)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


def gradient_ascent_cntk(loss, x, iterations, step):
    for i in range(iterations):
        grad_values, loss_value = loss.grad({loss.arguments[0]: x}, outputs=(loss.output,))
        grad_values = grad_values[0]
        m = np.mean(np.abs(grad_values))
        loss_value = loss_value[0]/m
        grad_values /= m
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


def run_vgg16():
    model = keras.applications.VGG16(weights='imagenet', include_top=False)
    model.summary()

    layer_contributions = {
        'block1_pool': 0,
        'block2_pool': 0,
        'block3_pool': 0.,
        'block4_pool': 1,
    }

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # Define the loss.
    loss = keras.backend.variable(0.)
    for layer_name in layer_contributions:
        # Add the L2 norm of the features of a layer to the loss.
        coeff = layer_contributions[layer_name]
        activation = layer_dict[layer_name].output

        # We avoid border artifacts by only involving non-border pixels in the loss.
        scaling = keras.backend.prod(keras.backend.cast(keras.backend.shape(activation), 'float32'))
        loss += coeff * keras.backend.sum(keras.backend.square(activation[:, 2: -2, 2: -2, :])) / scaling

    # This holds our generated image
    dream = model.input

    # Compute the gradients of the dream with regard to the loss.
    grads = keras.backend.gradients(loss, dream)[0]

    # Normalize gradients.
    grads /= keras.backend.maximum(keras.backend.mean(keras.backend.abs(grads)), 1e-7)

    # Set up function to retrieve the value
    # of the loss and gradients given an input image.
    outputs = [loss, grads]
    fetch_loss_and_grads = keras.backend.function([dream], outputs)

    # Playing with these hyper-parameters will also allow you to achieve new effects
    step = 0.01  # Gradient ascent step size
    num_octave = 1  # Number of scales at which to run gradient ascent
    octave_scale = 1.4  # Size ratio between scales
    iterations = 30  # Number of ascent steps per scale

    # If our loss gets larger than 10,
    # we will interrupt the gradient ascent process, to avoid ugly artifacts
    max_loss = None

    # Fill this to the path to the image you want to use
    base_image_path = os.path.join('..', 'DeepLearning', 'Ch_05_Class_Activation_Heatmaps', 'creative_commons_elephant.jpg')

    # Load the image into a Numpy array
    img = preprocess_image(base_image_path)

    # We prepare a list of shape tuples
    # defining the different scales at which we will run gradient ascent
    original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)

    # Reverse list of shapes, so that they are in increasing order
    successive_shapes = successive_shapes[::-1]

    # Resize the Numpy array of the image to our smallest scale
    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0])

    for shape in successive_shapes:
        print('Processing image shape', shape)
        img = resize_img(img, shape)
        img = gradient_ascent(fetch_loss_and_grads,
                              img,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss)
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img

        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)
        save_img(img, fname='dream_at_scale_' + str(shape) + '.png')

    save_img(img, fname='final_dream.png')
    print('Elapsed time: {0}'.format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.imshow(deprocess_image(np.copy(img)))
    plt.show()


def run():
    # Build the InceptionV3 network as.backend.
    # The model will be loaded with pre-trained ImageNet weights.
    model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)
    model.summary()

    # Dict mapping layer names to a coefficient
    # quantifying how much the layer's activation
    # will contribute to the loss we will seek to maximize.
    # Note that these are layer names as they appear
    # in the built-in InceptionV3 application.
    # You can list all layer names using `model.summary()`.
    layer_contributions = {
        'mixed2': 0.2,
        'mixed3': 3.,
        'mixed4': 2.,
        'mixed5': 1.5,
    }

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # Define the loss.
    loss = keras.backend.variable(0.)
    for layer_name in layer_contributions:
        # Add the L2 norm of the features of a layer to the loss.
        coeff = layer_contributions[layer_name]
        activation = layer_dict[layer_name].output

        # We avoid border artifacts by only involving non-border pixels in the loss.
        scaling = keras.backend.prod(keras.backend.cast(keras.backend.shape(activation), 'float32'))
        loss += coeff * keras.backend.sum(keras.backend.square(activation[:, 2: -2, 2: -2, :])) / scaling

    # This holds our generated image
    dream = model.input

    # Compute the gradients of the dream with regard to the loss.
    grads = keras.backend.gradients(loss, dream)[0]

    # Normalize gradients.
    grads /= keras.backend.maximum(keras.backend.mean(keras.backend.abs(grads)), 1e-7)

    # Set up function to retrieve the value
    # of the loss and gradients given an input image.
    outputs = [loss, grads]
    fetch_loss_and_grads = keras.backend.function([dream], outputs)

    # Playing with these hyper-parameters will also allow you to achieve new effects
    step = 0.01  # Gradient ascent step size
    num_octave = 3  # Number of scales at which to run gradient ascent
    octave_scale = 1.4  # Size ratio between scales
    iterations = 20  # Number of ascent steps per scale

    # If our loss gets larger than 10,
    # we will interrupt the gradient ascent process, to avoid ugly artifacts
    max_loss = 10.

    # Fill this to the path to the image you want to use
    base_image_path = os.path.join('..', 'DeepLearning', 'Ch_05_Visualizing_Intermediate_Activations', 'cat.1700.jpg')

    # Load the image into a Numpy array
    img = preprocess_image(base_image_path)

    # We prepare a list of shape tuples
    # defining the different scales at which we will run gradient ascent
    original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)

    # Reverse list of shapes, so that they are in increasing order
    successive_shapes = successive_shapes[::-1]

    # Resize the Numpy array of the image to our smallest scale
    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0])

    for shape in successive_shapes:
        print('Processing image shape', shape)
        img = resize_img(img, shape)
        img = gradient_ascent(fetch_loss_and_grads,
                              img,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss)
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img

        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)
        save_img(img, fname='dream_at_scale_' + str(shape) + '.png')

    save_img(img, fname='final_dream.png')
    print('Elapsed time: {0}'.format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.imshow(deprocess_image(np.copy(img)))
    plt.show()


def run_cntk(image_path, model_path):
    import functools
    import cv2

    model = cntk.load_model(model_path)

    pool_nodes = list()
    for l in cntk.logging.depth_first_search(model, lambda x: True, depth=0):
        if type(l) is cntk.ops.functions.Function:
            description = str(l)
            if description.find('Pooling') >= 0:
                pool_nodes.append(l)
                print(l)
    print(pool_nodes)

    # node contributions to the loss metric
    layer_contributions = {
        pool_nodes[2]: 1,
        pool_nodes[3]: 3,
    }

    # Define the loss
    loss = None
    for layer in layer_contributions.keys():
        coeff = layer_contributions[layer]
        activation = layer.output
        scaling = functools.reduce(lambda x, y: x*y, activation.shape)
        sum_squares = cntk.reduce_sum(cntk.square(activation))
        scaled_sum_squares = (coeff/scaling) * sum_squares
        if loss is None:
            loss = scaled_sum_squares
        else:
            loss += scaled_sum_squares

    dream = cntk.input_variable(shape=model.arguments[0].shape, needs_gradient=True, name='features')
    model = cntk.ops.combine(loss).clone(cntk.ops.CloneMethod.freeze, substitutions={model.arguments[0]: dream})
    step = 0.1  # Gradient ascent step size
    iterations = 5  # Number of ascent steps per scale

    # Load the image into a Numpy array
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    # cv2.imshow('Original Image', img.copy())

    img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    img /= 127.5
    img -= 1
    img = gradient_ascent_cntk(model, img, iterations=iterations, step=step)
    img = np.transpose(img, (1, 2, 0))
    img /= 2.
    img += 0.5
    img *= 255.
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def demo_cntk():
    import cntk_util
    cntk_util.VGG16.get_model(features_node=None, include_top=True)
    image_path = os.path.join('..', 'DeepLearning', 'Ch_05_Class_Activation_Heatmaps', 'creative_commons_elephant.jpg')

    import cv2
    dream_image = run_cntk(image_path=image_path, model_path=cntk_util.VGG16.vgg16_filename)
    cv2.imshow('The Dream', dream_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    # run_vgg16()
    demo_cntk()
    print('Elapsed time: {0}'.format(datetime.timedelta(seconds=time.time() - start_time)))
