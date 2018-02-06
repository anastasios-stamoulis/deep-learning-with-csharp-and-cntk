
# MNIST Example 

The original Python code can be found in [ch2-1.py](../../Python/ch2-1.py)

We'll need to do 4 things:

1. Load the MNIST data
2. Create the Network
3. Train the Network
4. Evaluate the Network

### Loading the MNIST Data with C#

The Python code that loads, preprocesses, and saves the data is: 

```
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = keras.utils.to_categorical(train_labels).astype('float32')
test_labels = keras.utils.to_categorical(test_labels).astype('float32')

train_images = np.ascontiguousarray(train_images)
train_labels = np.ascontiguousarray(train_labels)
test_images = np.ascontiguousarray(test_images)
test_labels = np.ascontiguousarray(test_labels)

train_images.tofile('train_images.bin')
train_labels.tofile('train_labels.bin')
test_images.tofile('test_images.bin')
test_labels.tofile('test_labels.bin')
```

In C#, as MNIST is a very small dataset, we'll store the data in plain-vanilla 2D arrays
 
```
float[][] train_images;
float[][] test_images;
float[][] train_labels;
float[][] test_labels;
```

To read the binary files that were created by the Python script, we'll use the helper method `load_binary_file`. 
This will read a numpy array that was saved in a flat memory layout, and create a `float[][]`. 

```
public static float[][] load_binary_file(string filepath, int numRows, int numColumns) {
  Console.WriteLine("Loading " + filepath);
  var buffer = new byte[sizeof(float) * numRows * numColumns];
  using (var reader = new System.IO.BinaryReader(System.IO.File.OpenRead(filepath))) {
    reader.Read(buffer, 0, buffer.Length);
  }
  var dst = new float[numRows][];
  for (int row = 0; row < dst.Length; row++) {
    dst[row] = new float[numColumns];
    System.Buffer.BlockCopy(buffer, row * numColumns, dst[row], 0, numColumns);
  }
  return dst;
}
```

The initialization of the arrays is done in the method `load_data`. 

```
void load_data() {
  if ( !System.IO.File.Exists("train_images.bin")) {
    System.IO.Compression.ZipFile.ExtractToDirectory("mnist_data.zip", ".");
  }
  train_images = Util.load_binary_file("train_images.bin", 60000, 28 * 28);
  test_images = Util.load_binary_file("test_images.bin", 10000, 28 * 28);
  train_labels = Util.load_binary_file("train_labels.bin", 60000, 10);
  test_labels = Util.load_binary_file("test_labels.bin", 60000, 10);
}
```



### Creating the Network in C# 

The Keras code that creates the network is: 

```
network = keras.models.Sequential()
network.add(keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(keras.layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

In CNTK, we'll need to be a little bit more explicit.

```
void create_network() {
  computeDevice = Util.get_compute_device();
  Console.WriteLine("Compute Device: " + computeDevice.AsString());

  image_tensor = CNTK.Variable.InputVariable(CNTK.NDShape.CreateNDShape(new int[] { 28, 28 }), CNTK.DataType.Float);
  label_tensor = CNTK.Variable.InputVariable(CNTK.NDShape.CreateNDShape(new int[] { 10 }), CNTK.DataType.Float);

  network = Util.Dense(image_tensor, 512, computeDevice);
  network = CNTK.CNTKLib.ReLU(network);
  network = Util.Dense(network, 10, computeDevice);

  loss_function = CNTK.CNTKLib.CrossEntropyWithSoftmax(network.Output, label_tensor);
  accuracy_function = CNTK.CNTKLib.ClassificationError(network.Output, label_tensor);

  var parameterVector = new CNTK.ParameterVector((System.Collections.ICollection)network.Parameters());
  var learner = CNTK.CNTKLib.RMSPropLearner(parameterVector, new CNTK.TrainingParameterScheduleDouble(0.99), 0.95, 2.0, 0.5, 2.0, 0.5);
  trainer = CNTK.CNTKLib.CreateTrainer(network, loss_function, accuracy_function, new CNTK.LearnerVector() { learner });
}
```

### Training the Network

With Keras, with this very small dataset, the training happens in a single line: 

```
network.fit(train_images, train_labels, epochs=5, batch_size=128)
```

With CNTK, we'll need to write a small loop that does the following:

* For the specified number of epochs

	* Randomly permute the training data

	* Group the training data into "mini-batches" 

	* Feed each mini-batch into the network

In C#, we'll have:
```
void train_network() {
  int epochs = 5;
  int batch_size = 128;

  for (int current_epoch = 0; current_epoch < epochs; current_epoch++) {
    Console.WriteLine(string.Format("Epoch {0}/{1}", current_epoch+1, epochs));
    var train_indices = Util.shuffled_indices(60000);
    var pos = 0;
    while (pos < train_indices.Length) {
      var pos_end = Math.Min(pos + batch_size, train_indices.Length);
      var minibatch_images = Util.get_tensors(image_tensor.Shape, train_images, train_indices, pos, pos_end, computeDevice);
      var minibatch_labels = Util.get_tensors(label_tensor.Shape, train_labels, train_indices, pos, pos_end, computeDevice);
      var feed_dictionary = new feed_t() { { image_tensor, minibatch_images }, { label_tensor, minibatch_labels } };
      trainer.TrainMinibatch(feed_dictionary, false, computeDevice);
      pos = pos_end;
    }
  }
}
```

### Evaluating the Network

The Keras code used for evaluating the network is: 

```
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
```

In C#, again we'll write a small loop to go over all test images: 
```
void evaluate_network() {
  var batch_size = 128;
  var pos = 0;
  var accuracy = 0.0;
  var num_batches = 0;
  var evaluator = CNTK.CNTKLib.CreateEvaluator(accuracy_function);
  while (pos < test_images.Length) {
    var pos_end = Math.Min(pos + batch_size, test_images.Length);
    var minibatch_images = Util.get_tensors(image_tensor.Shape, test_images, pos, pos_end, computeDevice);
    var minibatch_labels = Util.get_tensors(label_tensor.Shape, test_labels, pos, pos_end, computeDevice);
    var feed_dictionary = new CNTK.UnorderedMapVariableValuePtr() { { image_tensor, minibatch_images }, { label_tensor, minibatch_labels } };
    var minibatch_accuracy = evaluator.TestMinibatch(feed_dictionary, computeDevice);
    accuracy += minibatch_accuracy;
    pos = pos_end;
    num_batches++;
  }
  accuracy /= num_batches;
  Console.WriteLine(string.Format("Accuracy:{0:F3}", accuracy));
}
```

