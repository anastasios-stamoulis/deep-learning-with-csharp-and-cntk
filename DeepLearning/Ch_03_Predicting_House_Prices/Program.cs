using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;
using feed_t = System.Collections.Generic.Dictionary<CNTK.Variable, CNTK.Value>;
using test_feed_t = CNTK.UnorderedMapVariableValuePtr;


namespace Ch_03_Predicting_House_Prices {

  class PlotWindow : System.Windows.Window {

    public PlotWindow(List<List<double>> results) {
      var plotModel = new OxyPlot.PlotModel();
      plotModel.Title = "Mean Absolute Validation Error Per Fold";

      plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis() { Position = OxyPlot.Axes.AxisPosition.Left, Title = "Error" });
      plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis() { Position = OxyPlot.Axes.AxisPosition.Bottom, Title = "Epochs" });

      var colors = new OxyPlot.OxyColor[] { OxyPlot.OxyColors.Blue, OxyPlot.OxyColors.Green, OxyPlot.OxyColors.Red, OxyPlot.OxyColors.Black };
      for (int row = 0; row < results.Count; row++) {
        var lineSeries = new OxyPlot.Series.LineSeries();
        lineSeries.ItemsSource = results[row].Select((value, index) => new OxyPlot.DataPoint(index, value));
        lineSeries.Title = string.Format("Fold {0}/{1}", row + 1, results.Count);
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
      var results = train_network();
      var wpfApp = new System.Windows.Application();
      wpfApp.Run(new PlotWindow(results));
    }

    void create_network() {
      computeDevice = Util.get_compute_device();
      Console.WriteLine("Compute Device: " + computeDevice.AsString());

      x_tensor = CNTK.Variable.InputVariable(CNTK.NDShape.CreateNDShape(new int[] { 13 }), CNTK.DataType.Float);
      y_tensor = CNTK.Variable.InputVariable(CNTK.NDShape.CreateNDShape(new int[] { 1 }), CNTK.DataType.Float);

      network = CNTK.CNTKLib.ReLU(Util.Dense(x_tensor, 64, computeDevice));
      network = CNTK.CNTKLib.ReLU(Util.Dense(network, 64, computeDevice));
      network = Util.Dense(network, 1, computeDevice);

      loss_function = Util.MeanSquaredError(network.Output, y_tensor);
      accuracy_function = Util.MeanAbsoluteError(network.Output, y_tensor);

      var parameterVector = new CNTK.ParameterVector((System.Collections.ICollection)network.Parameters());
      var learner = CNTK.CNTKLib.AdamLearner(parameterVector, new CNTK.TrainingParameterScheduleDouble(0.001, 1), new CNTK.TrainingParameterScheduleDouble(0.9, 1), true);
      trainer = CNTK.CNTKLib.CreateTrainer(network, loss_function, accuracy_function, new CNTK.LearnerVector() { learner });
      evaluator = CNTK.CNTKLib.CreateEvaluator(accuracy_function);
    }

    double train_phase(int[] train_indices) {
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

    double evaluation_phase(int[] validation_indices) {
      var pos = 0;
      var num_batches = 0;
      var epoch_evaluation_accuracy = 0.0;
      while (pos < validation_indices.Length) {
        var pos_end = Math.Min(pos + batch_size, validation_indices.Length);
        var minibatch_x = Util.get_tensors(x_tensor.Shape, x_train, validation_indices, pos, pos_end, computeDevice);
        var minibatch_y = Util.get_tensors(y_tensor.Shape, y_train, validation_indices, pos, pos_end, computeDevice);
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
      var result = new List<List<double>>();

      int num_folds = 4;
      var num_val_samples = x_train.Length / num_folds;
      for (int fold_index = 0; fold_index < num_folds; fold_index++) {
        create_network();
        Console.WriteLine(string.Format("Fold {0}/{1}", fold_index + 1, num_folds));

        var val_indices = Enumerable.Range(fold_index * num_val_samples, num_val_samples).ToArray();
        var training_indices_part_one = Enumerable.Range(0, fold_index * num_val_samples).ToList();
        var training_indices_part_two = Enumerable.Range((fold_index + 1) * num_val_samples, x_train.Length - (fold_index + 1) * num_val_samples).ToList();
        var training_indices = training_indices_part_one.Concat(training_indices_part_two).ToArray();
        Util.shuffle(training_indices);

        var evaluation_accuracy_results = new List<double>();

        for (int current_epoch = 0; current_epoch < epochs; current_epoch++) {
          var training_accuracy = train_phase(training_indices);
          var evaluation_accuracy = evaluation_phase(val_indices);

          if (current_epoch % 10 == 9) {
            Console.WriteLine(string.Format("\tEpoch {0}/{1}, training_accuracy={2:F3}, evaluation_accuracy={3:F3}",
              current_epoch + 1,
              epochs,
              training_accuracy,
              evaluation_accuracy));
          }

          evaluation_accuracy_results.Add(evaluation_accuracy);
        }
        result.Add(evaluation_accuracy_results);
      }

      return result;
    }

    void load_data() {
      if (!System.IO.File.Exists("x_train.bin")) {
        System.IO.Compression.ZipFile.ExtractToDirectory("house_prices.zip", ".");
      }
      x_train = Util.load_binary_file("x_train.bin", 404, 13);
      y_train = Util.load_binary_file("y_train.bin", 404);
      x_test = Util.load_binary_file("x_test.bin", 102, 13);
      y_test = Util.load_binary_file("y_test.bin", 102);

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

    readonly int epochs = 50;
    readonly int batch_size = 16;
  }
}
