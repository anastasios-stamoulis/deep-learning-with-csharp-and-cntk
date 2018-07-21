import os
import sys
import cntk
import urllib
import numpy as np

class ShowProgress:
    pbar = None
    previous = None

    @staticmethod
    def show_progress_2(block_num, block_size, total_size):
        import tqdm
        if ShowProgress.pbar is None:
            ShowProgress.pbar = tqdm.tqdm(total=100)
            ShowProgress.previous = 0
        downloaded = block_num * block_size
        if downloaded <= total_size:
            current = int(100 * downloaded/total_size)
            delta = current - ShowProgress.previous
            ShowProgress.pbar.update(delta)
            ShowProgress.previous = current
        else:
            ShowProgress.pbar.close()
            ShowProgress.pbar = None

    @staticmethod
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if downloaded < total_size:
            print('\rProgress: {0:.2f}%'.format(100*downloaded/total_size), end='')
        else:
            print('\r Done')


class VGG16:
    vgg16_filename = 'VGG16_ImageNet_Caffe.model'
    downlink = 'https://www.cntk.ai/Models/Caffe_Converted/VGG16_ImageNet_Caffe.model'

    @staticmethod
    def get_model(features_node, include_top=False, use_finetuning=False):
        if not os.path.isfile(VGG16.vgg16_filename):
            urllib.request.urlretrieve(VGG16.downlink, VGG16.vgg16_filename, reporthook=ShowProgress.show_progress_2)

        model = cntk.load_model(VGG16.vgg16_filename)
        if include_top:
            return model
        pool5_node = cntk.logging.graph.find_by_name(model, 'pool5')
        data_node = cntk.logging.graph.find_by_name(model, 'data')

        # cloned_model = cntk.ops.combine(pool5_node).clone(
        #     cntk.ops.CloneMethod.share,
        #     substitutions={data_node: cntk.placeholder(name='features')})

        assert use_finetuning is False
        cloned_model = cntk.ops.combine(pool5_node).clone(cntk.ops.CloneMethod.freeze, substitutions={data_node: features_node})
        return cloned_model


class ImagenetV3:
    downlink = 'https://www.cntk.ai/Models/Caffe_Converted/BNInception_ImageNet_Caffe.model'
    # downlink = 'https://www.cntk.ai/Models/CNTK_Pretrained/InceptionV3_ImageNet_CNTK.model'
    # model_filename = 'BNInception_ImageNet_Caffe.model'
    model_filename = downlink.split('/')[-1]

    @staticmethod
    def get_model(features_node=None, include_top=False, use_finetuning=False):
        if not os.path.isfile(ImagenetV3.model_filename):
            urllib.request.urlretrieve(ImagenetV3.downlink, ImagenetV3.model_filename, reporthook=ShowProgress.show_progress_2)

        model = cntk.load_model(ImagenetV3.model_filename)
        return model
        # if include_top:
        #     return model
        # pool5_node = cntk.logging.graph.find_by_name(model, 'pool5')
        # data_node = cntk.logging.graph.find_by_name(model, 'data')
        #
        # # cloned_model = cntk.ops.combine(pool5_node).clone(
        # #     cntk.ops.CloneMethod.share,
        # #     substitutions={data_node: cntk.placeholder(name='features')})
        #
        # assert use_finetuning is False
        # cloned_model = cntk.ops.combine(pool5_node).clone(cntk.ops.CloneMethod.freeze, substitutions={data_node: features_node})
        # return cloned_model


def predict_elephant(use_keras=True):
    import PIL
    import keras.applications.imagenet_utils

    use_vgg16 = False

    image_path = os.path.join('..', 'DeepLearning', 'Ch_05_Class_Activation_Heatmaps', 'creative_commons_elephant.jpg')
    image_size = (224, 224)
    if (use_vgg16 is False) and (use_keras is False):
        image_size = (299, 299)
    img = np.asarray(PIL.Image.open(image_path).resize(image_size), dtype=np.float32)

    if use_keras:
        img = np.expand_dims(img, axis=0)
        if use_vgg16:
            img = keras.applications.imagenet_utils.preprocess_input(img, mode='keras')
            model = keras.applications.VGG16(include_top=True)
        else:
            img = keras.applications.imagenet_utils.preprocess_input(img, mode='tf')
            model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=True)
        model.summary()
        predictions = model.predict(img)
    else:
        import cntk_util
        if use_vgg16:
            img = keras.applications.imagenet_utils.preprocess_input(img, mode='keras')
            model = cntk_util.VGG16.get_model(features_node=None, include_top=True, use_finetuning=False)
        else:
            img -= 128.0
            img /= 128.0
            model = cntk_util.ImagenetV3.get_model(features_node=None, include_top=True, use_finetuning=False)

        img = np.transpose(img, (2, 0, 1))
        predictions = model.eval({model.arguments[0]: img})

    predictions_decoded = keras.applications.imagenet_utils.decode_predictions(predictions)
    print(predictions_decoded)

    predictions = predictions[0]
    predictions_indices = np.argsort(predictions)
    print(predictions_indices[:3])


