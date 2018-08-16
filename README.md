    
# Deep Learning with C# and CNTK

[Keras](https://keras.io/) is a very popular Python Deep Learning library. 

Recently, the creator of Keras, [Francois Chollet](https://twitter.com/fchollet), published the excellent book [Deep Learning with Python](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438). 

This repository is a C# port of https://github.com/fchollet/deep-learning-with-python-notebooks using 
[CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/) as backend.


## Examples

[Ch_02_First_Look_At_A_Neural_Network](DeepLearning/Ch_02_First_Look_At_A_Neural_Network) introduces softmax and fully connected layers. MNIST data set. 
   
[Ch_03_Classifying_Movie_Reviews](DeepLearning/Ch_03_Classifying_Movie_Reviews) introduces binary cross-entropy. IMDB data set.
  
[Ch_03_Predicting_House_Prices](DeepLearning/Ch_03_Predicting_House_Prices) introduces MSE and KFold training on a regression problem. Housing Prices dataset. 

[Ch_04_Overfitting_and_Underfitting](DeepLearning/Ch_04_Overfitting_and_Underfitting) introduces Regularization and Dropout. Housing Prices dataset. 

[Ch_05_Introduction_to_Convnets](DeepLearning/Ch_05_Introduction_to_Convnets) introduces Conv2D filters. MNIST data set.

[Ch_05_Using_Convnets_With_Small_Datasets](DeepLearning/Ch_05_Using_Convnets_With_Small_Datasets) introduces on-the-fly data augmentation. Cats And Dogs dataset.

[Ch_05_Using_A_Pretrained_Convnet](DeepLearning/Ch_05_Using_A_Pretrained_Convnet) 
uses the VGG16 "bottleneck" features, with optional augmentation, and fine-tuning.

[Ch_05_Visualizing_Intermediate_Activations](DeepLearning/Ch_05_Visualizing_Intermediate_Activations) displays the
feature maps that are output by various layers. 

[Ch_05_Visualizing_Convnet_Filters](DeepLearning/Ch_05_Visualizing_Convnet_Filters)
finds the visual pattern that each filter responds to using gradient ascent in input space.

[Ch_05_Class_Activation_Heatmaps](DeepLearning/Ch_05_Class_Activation_Heatmaps) show which part of an image a convnet focused on.

[Ch_06_One_Hot_Encoding](DeepLearning/Ch_06_One_Hot_Encoding) discusses one-hot encoding of words.
   
[Ch_06_Using_Word_Embeddings](DeepLearning/Ch_06_Using_Word_Embeddings) introduces the Embedding layer, and pre-trained word embeddings ([GloVe](https://nlp.stanford.edu/projects/glove/)).

[Ch_06_Understanding_Recurrent_Neural_Networks](DeepLearning/Ch_06_Understanding_Recurrent_Neural_Networks) introduces LSTMs.

[Ch_06_Advanced_Usage_Of_Recurrent_Neural_Networks](DeepLearning/Ch_06_Advanced_Usage_Of_Recurrent_Neural_Networks) 
does temperature forecasting with (stacked) GRUs. 

[Ch_06_Sequence_Processing_With_Convnets](DeepLearning/Ch_06_Sequence_Processing_With_Convnets) introduces Conv1D filters.

[Ch_08_Text_Generation_With_LSTM](DeepLearning/Ch_08_Text_Generation_With_LSTM) implements character-level LSTM text generation.  

[Ch_08_Deep_Dream](DeepLearning/Ch_08_Deep_Dream) shows how neural networks hallucinate.

[Ch_08_Neural_Style_Transfer](DeepLearning/Ch_08_Neural_Style_Transfer) applies the style of an image to another.


## Running the examples on Windows 10

The folder [`DeepLearning`](DeepLearning) contains the Visual Studio 2015 solution. 

No need to install CNTK, as it will be installed automatically by NuGet. 

If you have an NVIDIA graphics card, you will need to have CUDA + cuDNN installed. 

The project is self-contained. No need to install anything else. 

Note that apart from CNTK, the following NuGet packages are used:

* [Newtonsoft.JSON](https://www.newtonsoft.com/json) for parsing the ImageNet .json files.
* [OpenCV For Windows (pre-built)](https://www.nuget.org/packages/opencv.win.native/310.3.0) for
some image manipulation.
* [OxyPlot](http://www.oxyplot.org/) as a replacement of Matplotlib.
* [SciColorMaps](https://github.com/ar1st0crat/SciColorMaps), which are custom .NET color maps for
scientific visualization.


## Python Code

If you would like to experiment with the original Python code, the folder [`Python`](Python) contains the python code, as extracted from the [notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks). 

To run the Python code on Windows 10, just get anaconda3, install Keras (with `pip install keras`), and also install CNKT, if you would like
to use CNTK as Keras backend. 

For each Python script in the `Python` folder, there is a corresponding C# project, with a `README.md` file that explains how the port was made. 

