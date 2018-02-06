using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;
using batch_t = System.Collections.Generic.Dictionary<CNTK.Variable, CNTK.MinibatchData>;
using test_feed_t = CNTK.UnorderedMapVariableMinibatchData;

namespace Ch_05_Using_Convnets_With_Small_Datasets {
  class PlotWindow : System.Windows.Window {

    public PlotWindow(List<List<double>> results) {
      var plotModel = new OxyPlot.PlotModel();
      plotModel.Title = "Cats And Dogs";
      plotModel.LegendPosition = OxyPlot.LegendPosition.BottomRight;

      plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis() {
        Position = OxyPlot.Axes.AxisPosition.Left,
        Title = "Accuracy",
        Minimum = 0,
        Maximum = 1,
      });
      plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis() {
        Position = OxyPlot.Axes.AxisPosition.Bottom,
        Minimum = 0,
        Maximum = results[0].Count + 1,
        Title = "Epochs"
      });

      var labels = new string[] { "Train Set", "Validation Set" };
      var markerTypes = new OxyPlot.MarkerType[] { OxyPlot.MarkerType.Plus, OxyPlot.MarkerType.Circle };
      for (int row = 0; row < results.Count; row++) {
        var scatterSeries = new OxyPlot.Series.ScatterSeries() {
          MarkerType = markerTypes[row],
          MarkerStroke = OxyPlot.OxyColors.Blue,
          MarkerFill = OxyPlot.OxyColors.Black
        };
        scatterSeries.ItemsSource = results[row].Select((value, index) => new OxyPlot.Series.ScatterPoint(index + 1, value));
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
    
    const int training_set_size = 1000;
    const int validation_set_size = 500;
    const int test_set_size = 500;
    const int image_width = 150;
    const int image_height = 150;
    const int num_channels = 3;
    const int max_epochs = 100;
    const int epoch_size = 2 * training_set_size;
    const int minibatch_size = 32;

    readonly string DATASET_PATH = @"D:\kaggle_cats_dogs";

    void load_data() {
      if (!System.IO.Directory.Exists(DATASET_PATH)) {
        Console.WriteLine("You need to download the data set first");
        System.Environment.Exit(1);
      }
      create_map_txt_files();
    }

    void create_map_txt_files() {
      var filenames = new string[] { "train_map.txt", "validation_map.txt", "test_map.txt" };
      var num_entries = new int[] { training_set_size, validation_set_size, test_set_size };
      var counter = 0;
      for (int j = 0; j < filenames.Length; j++) {
        var filename = filenames[j];
        using (var dstFile = new System.IO.StreamWriter(filename)) {
          for (int i = 0; i < num_entries[j]; i++) {
            var cat_path = System.IO.Path.Combine(DATASET_PATH, $"cat.{counter}.jpg");
            var dog_path = System.IO.Path.Combine(DATASET_PATH, $"dog.{counter}.jpg");
            counter++;
            dstFile.WriteLine($"{cat_path}\t0");
            dstFile.WriteLine($"{dog_path}\t1");
          }
        }
        Console.WriteLine("Wrote " + System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), filename));
      }
    }

    CNTK.MinibatchSource create_minibatchSource(string map_file, int num_classes, bool train) {
      var transforms = new List<CNTK.CNTKDictionary>();
      if (true) {
        var randomSideTransform = CNTK.CNTKLib.ReaderCrop("RandomSide",
          new Tuple<int, int>(0, 0),
          new Tuple<float, float>(0.8f, 1.0f),
          new Tuple<float, float>(0.0f, 0.0f),
          new Tuple<float, float>(1.0f, 1.0f),
          "uniRatio");
        transforms.Add(randomSideTransform);
      }

      var scaleTransform = CNTK.CNTKLib.ReaderScale(image_width, image_height, num_channels);
      transforms.Add(scaleTransform);

      var imageDeserializer = CNTK.CNTKLib.ImageDeserializer(map_file, "labels", (uint)num_classes, "features", transforms);
      var minibatchSourceConfig = new CNTK.MinibatchSourceConfig(new CNTK.DictionaryVector() { imageDeserializer });
      return CNTK.CNTKLib.CreateCompositeMinibatchSource(minibatchSourceConfig);
    }

    void create_network() {
      computeDevice = Util.get_compute_device();
      features_tensor = CNTK.Variable.InputVariable(new int[] { image_height, image_width, num_channels }, CNTK.DataType.Float);
      label_tensor = CNTK.Variable.InputVariable(new int[] { 2 }, CNTK.DataType.Float);

      var scalar_factor = CNTK.Constant.Scalar<float>((float)(1.0 / 255.0), computeDevice);
      network = CNTK.CNTKLib.ElementTimes(scalar_factor, features_tensor);

      network = Util.Convolution2DWithReLU(network, 32, new int[] { 3, 3 }, computeDevice);
      network = CNTK.CNTKLib.Pooling(network, CNTK.PoolingType.Max, new int[] { 2, 2 }, new int[] { 2 });
      network = Util.Convolution2DWithReLU(network, 64, new int[] { 3, 3 }, computeDevice);
      network = CNTK.CNTKLib.Pooling(network, CNTK.PoolingType.Max, new int[] { 2, 2 }, new int[] { 2 });
      network = Util.Convolution2DWithReLU(network, 128, new int[] { 3, 3 }, computeDevice);
      network = CNTK.CNTKLib.Pooling(network, CNTK.PoolingType.Max, new int[] { 2, 2 }, new int[] { 2 });
      network = Util.Convolution2DWithReLU(network, 128, new int[] { 3, 3 }, computeDevice);
      network = CNTK.CNTKLib.Pooling(network, CNTK.PoolingType.Max, new int[] { 2, 2 }, new int[] { 2 });
      network = CNTK.CNTKLib.Dropout(network, 0.5);
      network = CNTK.CNTKLib.ReLU(Util.Dense(network, 512, computeDevice));
      network = Util.Dense(network, 2, computeDevice);

      loss_function = CNTK.CNTKLib.CrossEntropyWithSoftmax(network.Output, label_tensor);
      accuracy_function = CNTK.CNTKLib.ClassificationError(network.Output, label_tensor);
      var parameterVector = new CNTK.ParameterVector((System.Collections.ICollection)network.Parameters());
      var learner = CNTK.CNTKLib.AdamLearner(parameterVector, new CNTK.TrainingParameterScheduleDouble(0.0001, 1), new CNTK.TrainingParameterScheduleDouble(0.99, 1));
      trainer = CNTK.CNTKLib.CreateTrainer(network, loss_function, accuracy_function, new CNTK.LearnerVector() { learner });
      evaluator = CNTK.CNTKLib.CreateEvaluator(accuracy_function);
    }

    double train_phase(CNTK.MinibatchSource reader) {
      var featuresStreamInfo = reader.StreamInfo("features");
      var labelsStreamInfo = reader.StreamInfo("labels");
      var num_samples = 0;
      var num_minibatches = 0;
      var score = 0.0;
      while (num_samples < epoch_size) {
        num_minibatches++;
        var minibatchData = reader.GetNextMinibatch(minibatch_size, computeDevice);
        var arguments = new batch_t() { { features_tensor, minibatchData[featuresStreamInfo] }, { label_tensor, minibatchData[labelsStreamInfo] } };
        num_samples += (int)(minibatchData[featuresStreamInfo].numberOfSamples);
        trainer.TrainMinibatch(arguments, computeDevice);
        score += trainer.PreviousMinibatchEvaluationAverage();
      }
      var result = 1.0 - (score / num_minibatches);      
      return result;
    }

    double validation_phase(CNTK.MinibatchSource reader) {
      var featuresStreamInfo = reader.StreamInfo("features");
      var labelsStreamInfo = reader.StreamInfo("labels");
      var num_samples = 0;
      var num_minibatches = 0;
      var score = 0.0;
      while (num_samples < 2*validation_set_size) {
        num_minibatches++;
        var minibatchData = reader.GetNextMinibatch(minibatch_size, computeDevice);
        var arguments = new test_feed_t { { features_tensor, minibatchData[featuresStreamInfo] }, { label_tensor, minibatchData[labelsStreamInfo] } };
        num_samples += (int)(minibatchData[featuresStreamInfo].numberOfSamples);
        evaluator.TestMinibatch(arguments, computeDevice);
        score += trainer.PreviousMinibatchEvaluationAverage();
      }
      var result = 1.0 - (score / num_minibatches);
      return result;
    }

    List<List<double>> train_network() {
      var result = new List<List<double>>() { new List<double>(), new List<double>() };
      
      var train_minibatchSource = create_minibatchSource("train_map.txt", num_classes: 2, train: true);
      var validation_minibatchSource = create_minibatchSource("validation_map.txt", num_classes: 2, train: false);

      for (int epoch = 0; epoch < max_epochs; epoch++) {
        Console.Write($"Epoch {epoch + 1:D3}/{max_epochs}");
        var train_accuracy = train_phase(train_minibatchSource);
        Console.Write($", training accuracy: {train_accuracy:F3}");
        var validation_accuracy = validation_phase(validation_minibatchSource);
        Console.WriteLine($", validation accuracy: {validation_accuracy:F3}");

        result[0].Add(train_accuracy);
        result[1].Add(validation_accuracy);
      }
      return result;
    }

    void run() {
      load_data();
      create_network();
      var results = train_network();
      var wpfApp = new System.Windows.Application();
      wpfApp.Run(new PlotWindow(results));
    }

    CNTK.Function network;
    CNTK.Function loss_function;
    CNTK.Function accuracy_function;
    CNTK.Trainer trainer;
    CNTK.Evaluator evaluator;

    CNTK.Variable features_tensor;
    CNTK.Variable label_tensor;
    CNTK.DeviceDescriptor computeDevice;

  }
}
