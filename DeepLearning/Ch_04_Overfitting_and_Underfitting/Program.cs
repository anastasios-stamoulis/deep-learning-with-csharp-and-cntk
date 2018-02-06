using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;
using feed_t = System.Collections.Generic.Dictionary<CNTK.Variable, CNTK.Value>;
using test_feed_t = CNTK.UnorderedMapVariableValuePtr;


namespace Ch_04_Overfitting_and_Underfitting {
  class PlotWindow : System.Windows.Window {

    public PlotWindow(List<List<double>> results) {
      var plotModel = new OxyPlot.PlotModel();
      plotModel.Title = "Overfitting and Underfitting";

      plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis() {
        Position = OxyPlot.Axes.AxisPosition.Left,
        Title = "Validation Loss",
        Minimum = 0,
        Maximum = 1,
      });
      plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis() {
        Position = OxyPlot.Axes.AxisPosition.Bottom,
        Minimum = 0,
        Maximum = results[0].Count+1,
        Title = "Epochs" });

      var labels = new string[] { "Original Model", "Dropout-regularized model" };
      var markerTypes = new OxyPlot.MarkerType[] { OxyPlot.MarkerType.Plus, OxyPlot.MarkerType.Circle };
      for (int row = 0; row < results.Count; row++) {
        var scatterSeries = new OxyPlot.Series.ScatterSeries() {
          MarkerType = markerTypes[row],
          MarkerStroke = OxyPlot.OxyColors.Blue,
          MarkerFill = OxyPlot.OxyColors.Blue
        };
        scatterSeries.ItemsSource = results[row].Select((value, index) => new OxyPlot.Series.ScatterPoint(index+1, value));
        scatterSeries.Title = labels[row];
        plotModel.Series.Add(scatterSeries);
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

    void create_network(bool regularize=false, bool add_dropout=false) {
      computeDevice = Util.get_compute_device();
      Console.WriteLine("Compute Device: " + computeDevice.AsString());

      x_tensor = CNTK.Variable.InputVariable(CNTK.NDShape.CreateNDShape(new int[] { 10000 }), CNTK.DataType.Float);
      y_tensor = CNTK.Variable.InputVariable(CNTK.NDShape.CreateNDShape(new int[] { 1 }), CNTK.DataType.Float);

      network = CNTK.CNTKLib.ReLU(Util.Dense(x_tensor, 16, computeDevice));
      if ( add_dropout ) {
        network = CNTK.CNTKLib.Dropout(network, 0.5);
      }
      network = CNTK.CNTKLib.ReLU(Util.Dense(network, 16, computeDevice));
      if (add_dropout) {
        network = CNTK.CNTKLib.Dropout(network, 0.5);
      }
      network = CNTK.CNTKLib.Sigmoid(Util.Dense(network, 1, computeDevice));

      loss_function = CNTK.CNTKLib.BinaryCrossEntropy(network.Output, y_tensor);
      accuracy_function = loss_function;

      var learningOptions = new CNTK.AdditionalLearningOptions() {
        l1RegularizationWeight = regularize ? 0.001 : 0,
        l2RegularizationWeight = regularize ? 0.001 : 0
      };

      var learner = CNTK.CNTKLib.AdamLearner(
        parameters: new CNTK.ParameterVector((System.Collections.ICollection)network.Parameters()),
        learningRateSchedule: new CNTK.TrainingParameterScheduleDouble(0.001, 1), 
        momentumSchedule: new CNTK.TrainingParameterScheduleDouble(0.9, 1), 
        unitGain: true, 
        varianceMomentumSchedule: new CNTK.TrainingParameterScheduleDouble(0.9999986111120757, 1),
        epsilon: 1e-8,
        adamax: false,
        additionalOptions: learningOptions);

      trainer = CNTK.CNTKLib.CreateTrainer(
        network, 
        loss_function, 
        accuracy_function, 
        new CNTK.LearnerVector() { learner });

      evaluator = CNTK.CNTKLib.CreateEvaluator(accuracy_function);
    }

    double train_phase() {
      var train_indices = Util.shuffled_indices(x_train.Length);
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
      var num_batches = 0;
      var epoch_evaluation_accuracy = 0.0;
      var pos = 0;
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
      Console.WriteLine("Training Original Model");
      var original_model_validation_loss = train_network(regularize: false, add_dropout: false);

      Console.WriteLine("\nTraining Dropout-Regularized Model");
      var dropout_regularized_validation_loss = train_network(regularize: true, add_dropout: true);

      return new List<List<double>> { original_model_validation_loss, dropout_regularized_validation_loss };
    }


    List<double> train_network(bool regularize, bool add_dropout) {
      create_network(regularize, add_dropout);

      var training_loss = new List<double>();
      var evaluation_loss = new List<double>();

      for (int current_epoch = 0; current_epoch < epochs; current_epoch++) {
        training_loss.Add(train_phase());
        evaluation_loss.Add(evaluation_phase());

        Console.WriteLine(string.Format("Epoch {0}/{1}, training_loss={2:F3}, validation_loss={3:F3}",
          current_epoch + 1,
          epochs,
          training_loss[current_epoch],
          evaluation_loss[current_epoch]));
      }

      return evaluation_loss;
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
  }
}
