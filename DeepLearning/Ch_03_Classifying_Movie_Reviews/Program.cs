using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;
using feed_t = System.Collections.Generic.Dictionary<CNTK.Variable, CNTK.Value>;
using test_feed_t = CNTK.UnorderedMapVariableValuePtr;

namespace Ch_03_Classifying_Movie_Reviews {
  
  class PlotWindow : System.Windows.Window {

    public PlotWindow(List<List<double>> results) {
      var plotModel = new OxyPlot.PlotModel();
      plotModel.Title = "Training and Validation Accuracy";

      plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis() { Position = OxyPlot.Axes.AxisPosition.Left, Title = "Accuracy", Minimum=0, Maximum=1 });
      plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis() { Position = OxyPlot.Axes.AxisPosition.Bottom, Title = "Epochs" });

      var labels = new string[] { "Training", "Validation" };
      var colors = new OxyPlot.OxyColor[] { OxyPlot.OxyColors.Blue, OxyPlot.OxyColors.Green };
      for (int row = 0; row < results.Count; row++) {
        var lineSeries = new OxyPlot.Series.LineSeries();
        lineSeries.ItemsSource = results[row].Select((value, index) => new OxyPlot.DataPoint(index, value));
        lineSeries.Title = labels[row];
        lineSeries.Color = colors[row];
        plotModel.Series.Add(lineSeries);
      }

      var plotView = new OxyPlot.Wpf.PlotView();
      plotView.Model = plotModel;

      Title = "Chart";
      Content = plotView;

    }
  }

  class Program {
    [STAThread]
    static void Main(string[] args) {
      new Program().run();
    }

    void run() {
      load_data();
      create_network();
      var results = train_network();
      var wpfApp = new System.Windows.Application();
      wpfApp.Run(new PlotWindow(results));
    }

    void create_network() {
      computeDevice = Util.get_compute_device();
      Console.WriteLine("Compute Device: " + computeDevice.AsString());

      x_tensor = CNTK.Variable.InputVariable(CNTK.NDShape.CreateNDShape(new int[] { 10000 }), CNTK.DataType.Float);
      y_tensor = CNTK.Variable.InputVariable(CNTK.NDShape.CreateNDShape(new int[] { 1 }), CNTK.DataType.Float);

      network = CNTK.CNTKLib.ReLU(Util.Dense(x_tensor, 16, computeDevice));
      network = CNTK.CNTKLib.ReLU(Util.Dense(network, 16, computeDevice));
      network = CNTK.CNTKLib.Sigmoid(Util.Dense(network, 1, computeDevice));

      loss_function = CNTK.CNTKLib.BinaryCrossEntropy(network.Output, y_tensor);
      accuracy_function = Util.BinaryAccuracy(network.Output, y_tensor);

      var parameterVector = new CNTK.ParameterVector((System.Collections.ICollection)network.Parameters());
      var learner = CNTK.CNTKLib.AdamLearner(parameterVector, new CNTK.TrainingParameterScheduleDouble(0.001, 1), new CNTK.TrainingParameterScheduleDouble(0.9, 1), true);
      trainer = CNTK.CNTKLib.CreateTrainer(network, loss_function, accuracy_function, new CNTK.LearnerVector() { learner });
      evaluator = CNTK.CNTKLib.CreateEvaluator(accuracy_function);
    }

    double train_phase() {
      var train_indices = Util.shuffled_indices(x_train.Length - offset);
      var pos = 0;
      var num_batches = 0;
      var epoch_training_accuracy = 0.0;
      while (pos < train_indices.Length) {
        var pos_end = Math.Min(pos + batch_size, train_indices.Length);
        var minibatch_x = Util.get_tensors(x_tensor.Shape, x_train, train_indices, pos, pos_end, computeDevice);
        var minibatch_y = Util.get_tensors(y_tensor.Shape, y_train, train_indices, pos, pos_end, computeDevice);
        var feed_dictionary = new feed_t() { { x_tensor, minibatch_x }, { y_tensor, minibatch_y } };
        trainer.TrainMinibatch(feed_dictionary, true, computeDevice);
        var minibatch_accuracy = trainer.PreviousMinibatchEvaluationAverage();
        epoch_training_accuracy += minibatch_accuracy;
        pos = pos_end;
        num_batches++;
      }
      epoch_training_accuracy /= num_batches;
      return epoch_training_accuracy;
    }

    double evaluation_phase() {
      var pos = offset;
      var num_batches = 0;
      var epoch_evaluation_accuracy = 0.0;
      while (pos < x_test.Length) {
        var pos_end = Math.Min(pos + batch_size, x_test.Length);
        var minibatch_x = Util.get_tensors(x_tensor.Shape, x_test, pos, pos_end, computeDevice);
        var minibatch_y = Util.get_tensors(y_tensor.Shape, y_test, pos, pos_end, computeDevice);
        var feed_dictionary = new test_feed_t() { { x_tensor, minibatch_x }, { y_tensor, minibatch_y } };
        var minibatch_accuracy = evaluator.TestMinibatch(feed_dictionary, computeDevice);
        epoch_evaluation_accuracy += minibatch_accuracy;
        num_batches++;
        pos = pos_end;
      }
      epoch_evaluation_accuracy /= num_batches;
      return epoch_evaluation_accuracy;
    }

    List<List<double>> train_network() {
      var training_accuracy_results = new List<double>();
      var evaluation_accuracy_results = new List<double>();

      for (int current_epoch = 0; current_epoch < epochs; current_epoch++) {
        training_accuracy_results.Add(train_phase());
        evaluation_accuracy_results.Add(evaluation_phase());

        Console.WriteLine(string.Format("Epoch {0}/{1}, training_accuracy={2:F3}, evaluation_accuracy={3:F3}", 
          current_epoch + 1, 
          epochs, 
          training_accuracy_results[current_epoch], 
          evaluation_accuracy_results[current_epoch]));
        }

        return new List<List<double>>() { training_accuracy_results, evaluation_accuracy_results };
    }

    void load_data() {
      if (!System.IO.File.Exists("x_train.bin")) {
        System.IO.Compression.ZipFile.ExtractToDirectory("imdb_data.zip", ".");
      }
      x_train = Util.load_binary_file("x_train.bin", 25000, 10000);
      y_train = Util.load_binary_file("y_train.bin", 25000);
      x_test = Util.load_binary_file("x_test.bin", 25000, 10000);
      y_test = Util.load_binary_file("y_test.bin", 25000);

      Console.WriteLine("Done with loading data\n");
    }

    CNTK.Function network;
    CNTK.Function loss_function;
    CNTK.Function accuracy_function;
    CNTK.Trainer trainer;
    CNTK.Evaluator evaluator;

    CNTK.Variable x_tensor;
    CNTK.Variable y_tensor;
    CNTK.DeviceDescriptor computeDevice;

    float[][] x_train;
    float[] y_train;
    float[][] x_test;
    float[] y_test;

    readonly int epochs = 7;
    readonly int batch_size = 32;
    readonly int offset = 10000;
  }
}
