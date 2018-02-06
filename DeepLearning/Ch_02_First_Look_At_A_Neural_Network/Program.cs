using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;
using feed_t = System.Collections.Generic.Dictionary<CNTK.Variable, CNTK.Value>;

namespace Ch_02_First_Look_At_A_Neural_Network {

  class Program {
    static void Main(string[] args) {
      new Program().run();
    }

    void load_data() {
      if ( !System.IO.File.Exists("train_images.bin")) {
        System.IO.Compression.ZipFile.ExtractToDirectory("mnist_data.zip", ".");
      }
      train_images = Util.load_binary_file("train_images.bin", 60000, 28 * 28);
      test_images = Util.load_binary_file("test_images.bin", 10000, 28 * 28);
      train_labels = Util.load_binary_file("train_labels.bin", 60000, 10);
      test_labels = Util.load_binary_file("test_labels.bin", 60000, 10);
      Console.WriteLine("Done with loading data\n");
    }


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

    void run() {
      var ver = System.Reflection.Assembly.GetAssembly(typeof(CNTK.Trainer)).FullName;
      Console.WriteLine(ver);
      load_data();
      create_network();
      train_network();
      evaluate_network();
    }

    CNTK.Function network;
    CNTK.Function loss_function;
    CNTK.Function accuracy_function;
    CNTK.Trainer trainer;

    CNTK.Variable image_tensor;
    CNTK.Variable label_tensor;
    CNTK.DeviceDescriptor computeDevice;

    float[][] train_images;
    float[][] test_images;
    float[][] train_labels;
    float[][] test_labels;
  }
}
