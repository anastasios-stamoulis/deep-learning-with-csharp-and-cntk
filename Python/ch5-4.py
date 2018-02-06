
import os
import sys

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
import keras.utils
import keras.datasets
import keras.models
import keras.layers
import keras.applications
import keras.preprocessing.image
import numpy as np
import matplotlib.pyplot as plt

base_dir = 'd:\\kaggle_cats_dogs'


def visualizing_intermediate_activations():
    model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    model.summary()
    img_path = os.path.join(base_dir, 'cat.1700.jpg')

    img = keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
    img_tensor = keras.preprocessing.image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.


    plt.imshow(img_tensor[0])
    plt.show()


    # Extracts the outputs of the top 8 layers:
    layer_outputs = [layer.output for layer in model.layers[1:8]]
    # Creates a model that will return these outputs, given the model input:
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)

    # This will return a list of 5 Numpy arrays:
    # one array per layer activation
    activations = activation_model.predict(img_tensor)

    first_layer_activation = activations[3]
    print(first_layer_activation.shape)
    print(model.layers[1:8][0].name)

    plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
    plt.show()

    # These are the names of the layers, so can have them as part of our plot
    layer_names = []
    for layer in model.layers[1:8]:
        layer_names.append(layer.name)

    images_per_row = 16

    # Now let's display our feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                row * size: (row + 1) * size] = channel_image

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()


def visualizing_convnet_filters():
    model = keras.applications.VGG16(weights='imagenet', include_top=False)
    model.summary()

    layer_name = 'block3_conv1'
    filter_index = 0

    layer_output = model.get_layer(layer_name).output
    loss = keras.backend.mean(layer_output[:, :, :, filter_index])

    # The call to `gradients` returns a list of tensors (of size 1 in this case)
    # hence we only keep the first element -- which is a tensor.
    grads = keras.backend.gradients(loss, model.input)[0]

    # We add 1e-5 before dividing so as to avoid accidentally dividing by 0.
    grads /= (keras.backend.sqrt(keras.backend.mean(keras.backend.square(grads))) + 1e-5)

    iterate = keras.backend.function([model.input], [loss, grads])

    # Let's test it:
    loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

    # We start from a gray image with some noise
    input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.  # this is the magnitude of each gradient update
    for i in range(40):
        # Compute the loss value and gradient value
        loss_value, grads_value = iterate([input_img_data])
        # Here we adjust the input image in the direction that maximizes the loss
        input_img_data += grads_value * step

    def generate_pattern(layer_name, filter_index, size=150):
        # Build a loss function that maximizes the activation
        # of the nth filter of the layer considered.
        layer_output = model.get_layer(layer_name).output
        loss = keras.backend.mean(layer_output[:, :, :, filter_index])

        # Compute the gradient of the input picture wrt this loss
        grads = keras.backend.gradients(loss, model.input)[0]

        # Normalization trick: we normalize the gradient
        grads /= (keras.backend.sqrt(keras.backend.mean(keras.backend.square(grads))) + 1e-5)

        # This function returns the loss and grads given the input picture
        def deprocess_image(x):
            # normalize tensor: center on 0., ensure std is 0.1
            x -= x.mean()
            x /= (x.std() + 1e-5)
            x *= 0.1

            # clip to [0, 1]
            x += 0.5
            x = np.clip(x, 0, 1)

            # convert to RGB array
            x *= 255
            x = np.clip(x, 0, 255).astype('uint8')
            return x

        iterate = keras.backend.function([model.input], [loss, grads])

        # We start from a gray image with some noise
        input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

        # Run gradient ascent for 40 steps
        step = 1.
        for i in range(40):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

        img = input_img_data[0]
        return deprocess_image(img)


    plt.imshow(generate_pattern('block3_conv1', 0))
    plt.show()



    for layer_name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']:
        size = 64
        margin = 5

        # This a empty (black) image where we will store our results.
        results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

        for i in range(8):  # iterate over the rows of our results grid
            for j in range(8):  # iterate over the columns of our results grid
                # Generate the pattern for filter `i + (j * 8)` in `layer_name`
                filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

                # Put the result in the square `(i, j)` of the results grid
                horizontal_start = i * size + i * margin
                horizontal_end = horizontal_start + size
                vertical_start = j * size + j * margin
                vertical_end = vertical_start + size
                results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

        # Display the results grid
        plt.figure(figsize=(20, 20))
        plt.imshow(results)
        plt.show()


def visualizing_heatmaps_of_class_activation():
    import cv2

    keras.backend.clear_session()

    # Note that we are including the densely-connected classifier on top;
    # all previous times, we were discarding it.
    model = keras.applications.VGG16(weights='imagenet')

    # The local path to our target image
    img_path = '..\\DeepLearning\\Ch_05_Class_Activation_Heatmaps\\creative_commons_elephant.jpg'

    # `img` is a PIL image of size 224x224
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

    # `x` is a float32 Numpy array of shape (224, 224, 3)
    x = keras.preprocessing.image.img_to_array(img)

    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)

    # Finally we preprocess the batch
    # (this does channel-wise color normalization)
    x = keras.applications.vgg16.preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', keras.applications.vgg16.decode_predictions(preds, top=3)[0])

    # This is the "african elephant" entry in the prediction vector
    african_elephant_output = model.output[:, 386]

    # The is the output feature map of the `block5_conv3` layer,
    # the last convolutional layer in VGG16
    last_conv_layer = model.get_layer('block5_conv3')

    # This is the gradient of the "african elephant" class with regard to
    # the output feature map of `block5_conv3`
    grads = keras.backend.gradients(african_elephant_output, last_conv_layer.output)[0]

    # This is a vector of shape (512,), where each entry
    # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = keras.backend.mean(grads, axis=(0, 1, 2))

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`,
    # given a sample image
    iterate = keras.backend.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the elephant class
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)


    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()


    # We use cv2 to load the original image
    img = cv2.imread(img_path)

    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # We convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)

    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + img

    # Save the image to disk
    cv2.imwrite('elephant_cam.jpg', superimposed_img)


if __name__ == '__main__':
    visualizing_intermediate_activations()
    visualizing_convnet_filters()
    visualizing_heatmaps_of_class_activation()
