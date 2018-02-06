using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;
using feed_t = System.Collections.Generic.Dictionary<CNTK.Variable, CNTK.Value>;
using batch_t = System.Collections.Generic.Dictionary<CNTK.Variable, CNTK.MinibatchData>;
using test_feed_t = CNTK.UnorderedMapVariableValuePtr;

namespace Ch_05_Using_A_Pretrained_Convnet {

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
    string create_map_txt_file_if_needed(int start_index, int sample_count, string prefix) {
      var map_filename = prefix + "_map.txt";
      if (System.IO.File.Exists(map_filename)) { return map_filename; }
      using (var dstFile = new System.IO.StreamWriter(map_filename)) {
        for (int i = start_index; i < (start_index + sample_count); i++) {
          var cat_path = System.IO.Path.Combine(DATASET_PATH, $"cat.{i}.jpg");
          var dog_path = System.IO.Path.Combine(DATASET_PATH, $"dog.{i}.jpg");
          dstFile.WriteLine($"{cat_path}\t0");
          dstFile.WriteLine($"{dog_path}\t1");
        }
      }
      Console.WriteLine("Wrote " + System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), map_filename));
      return map_filename;
    }


    CNTK.MinibatchSource create_minibatch_source(CNTK.NDShape shape, int start_index, int sample_count, string prefix, bool is_training=false, bool use_augmentations=false) {
      var map_filename = create_map_txt_file_if_needed(start_index, sample_count, prefix);

      var transforms = new List<CNTK.CNTKDictionary>();
      if (use_augmentations) {
        var randomSideTransform = CNTK.CNTKLib.ReaderCrop("RandomSide",
          new Tuple<int, int>(0, 0),
          new Tuple<float, float>(0.8f, 1.0f),
          new Tuple<float, float>(0.0f, 0.0f),
          new Tuple<float, float>(1.0f, 1.0f),
          "uniRatio");
        transforms.Add(randomSideTransform);
      }
      
      var scaleTransform = CNTK.CNTKLib.ReaderScale(width: shape[1], height: shape[0], channels: shape[2]);
      transforms.Add(scaleTransform);

      var imageDeserializer = CNTK.CNTKLib.ImageDeserializer(map_filename, "labels", 2, "features", transforms);
      var minibatchSourceConfig = new CNTK.MinibatchSourceConfig(new CNTK.DictionaryVector() { imageDeserializer });
      if ( !is_training ) {
        minibatchSourceConfig.randomizationWindowInChunks = 0;
        minibatchSourceConfig.randomizationWindowInSamples = 0;
      }
      return CNTK.CNTKLib.CreateCompositeMinibatchSource(minibatchSourceConfig);
    }

    Tuple<float[][], float[][]> extract_features(int start_index, int sample_count, string prefix) {
      var extracted_features = new float[2*sample_count][];
      var extracted_labels = new float[extracted_features.Length][];
      
      var labels = CNTK.Variable.InputVariable(new int[] { 2 }, CNTK.DataType.Float, "labels");
      var features = CNTK.Variable.InputVariable(new int[] { 150, 150, 3 }, CNTK.DataType.Float, "features");
      var scalar_factor = CNTK.Constant.Scalar<float>((float)(1.0 / 255.0), computeDevice);
      var scaled_features = CNTK.CNTKLib.ElementTimes(scalar_factor, features);

      var conv_base = VGG16.get_model(scaled_features, computeDevice);
      //Util.PredorderTraverse(conv_base);

      var minibatch_source = create_minibatch_source(features.Shape, start_index, sample_count, prefix);
      var features_stream_info = minibatch_source.StreamInfo("features");
      var labels_stream_info = minibatch_source.StreamInfo("labels");
      var pos = 0;
      while (pos < extracted_features.Length) {
        var pos_end = Math.Min(pos + batch_size, extracted_features.Length);
        var data = minibatch_source.GetNextMinibatch((uint)(pos_end - pos), computeDevice);

        var input_d = new Dictionary<CNTK.Variable, CNTK.Value>() { { features, data[features_stream_info].data } };
        var output_d = new Dictionary<CNTK.Variable, CNTK.Value>() { { conv_base.Output, null } };
        conv_base.Evaluate(input_d, output_d, computeDevice);

        var minibatch_extracted_features = output_d[conv_base.Output].GetDenseData<float>(conv_base.Output);
        for (int i = 0; i < data[features_stream_info].numberOfSamples; i++) {
          extracted_features[pos + i] = minibatch_extracted_features[i].ToArray();
          extracted_labels[pos+i] = new float[2];
          extracted_labels[pos + i][i % 2] = 1;
        }
        pos = pos_end;
      }
      
      return Tuple.Create(extracted_features, extracted_labels);
    }

    void for_debugging() {
#if false
        var shape = data[features_stream_info].data.Shape;
        var numElements = shape.TotalSize;
        var buffer = new float[numElements];
        var buffer_cpu = new CNTK.NDArrayView(shape, buffer, CNTK.DeviceDescriptor.CPUDevice);
        buffer_cpu.CopyFrom(data[features_stream_info].data.Data);
        var firstImage = new float[3 * 150 * 150];
        System.Array.Copy(buffer, firstImage.Length, firstImage, 0, firstImage.Length);
        var wpfApp = new System.Windows.Application();
        wpfApp.Run(new PlotWindow(firstImage));
        var mel = new float[40];
        var nd_cpu = new CNTK.NDArrayView(CNTK.NDShape.CreateNDShape(new int[] { 2, 1, (int)data[labels_stream_info].numberOfSamples }), mel, CNTK.DeviceDescriptor.CPUDevice);
        nd_cpu.CopyFrom(data[labels_stream_info].data.Data);
#endif
    }

    void compute_features_and_labels(
      ref float[][] train_features, 
      ref float[][] validation_features, 
      ref float[][] test_features,
      ref float[][] train_labels, 
      ref float[][] validation_labels, 
      ref float[][] test_labels) {

      if (System.IO.File.Exists("train_features.bin")) {
        train_features = Util.load_binary_file("train_features.bin", 2000, extracted_feature_length);
        train_labels = Util.load_binary_file("train_labels.bin", 2000, 2);

        validation_features = Util.load_binary_file("validation_features.bin", 1000, extracted_feature_length);
        validation_labels = Util.load_binary_file("validation_labels.bin", 1000, 2);

        test_features = Util.load_binary_file("test_features.bin", 1000, extracted_feature_length);
        test_labels = Util.load_binary_file("test_labels.bin", 1000, 2);
      }
      else {
        var extracted_info = extract_features(0, 2000, "all");
        var all_features = extracted_info.Item1;
        var all_labels = extracted_info.Item2;

        train_features = new float[2000][];
        train_labels = new float[2000][];
        Array.Copy(all_features, train_features, 2000);
        Array.Copy(all_labels, train_labels, 2000);
        Util.save_binary_file(train_features, "train_features.bin");
        Util.save_binary_file(train_labels, "train_labels.bin");

        validation_features = new float[1000][];
        validation_labels = new float[1000][];
        Array.Copy(all_features, 2000, validation_features, 0, 1000);
        Array.Copy(all_labels, 2000, validation_labels, 0, 1000);
        Util.save_binary_file(validation_features, "validation_features.bin");
        Util.save_binary_file(validation_labels, "validation_labels.bin");

        test_features = new float[1000][];
        test_labels = new float[1000][];
        Array.Copy(all_features, 3000, test_features, 0, 1000);
        Array.Copy(all_labels, 3000, test_labels, 0, 1000);
        Util.save_binary_file(test_features, "test_features.bin");
        Util.save_binary_file(test_labels, "test_labels.bin");
      }
    }

    List<List<double>> train_with_extracted_features() {
      float[][] train_features = null, validation_features = null, test_features = null;
      float[][] train_labels = null, validation_labels = null, test_labels = null;

      compute_features_and_labels(ref train_features, ref validation_features, ref test_features, ref train_labels, ref validation_labels, ref test_labels);

      var labels = CNTK.Variable.InputVariable(new int[] { 2, }, CNTK.DataType.Float, "labels_var");
      var features = CNTK.Variable.InputVariable(new int[] { extracted_feature_length, }, CNTK.DataType.Float, "features_var");

      var model = CNTK.CNTKLib.ReLU(Util.Dense(features, 256, computeDevice));
      model = CNTK.CNTKLib.Dropout(model, 0.5);
      model = Util.Dense(model, 2, computeDevice);

      var loss_function = CNTK.CNTKLib.CrossEntropyWithSoftmax(model.Output, labels);
      var accuracy_function = CNTK.CNTKLib.ClassificationError(model.Output, labels);

      var pv = new CNTK.ParameterVector((System.Collections.ICollection)model.Parameters());
      var learner = CNTK.CNTKLib.AdamLearner(pv, new CNTK.TrainingParameterScheduleDouble(0.0001, 1), new CNTK.TrainingParameterScheduleDouble(0.99, 1));
      var trainer = CNTK.Trainer.CreateTrainer(model, loss_function, accuracy_function, new CNTK.Learner[] { learner });
      var evaluator = CNTK.CNTKLib.CreateEvaluator(accuracy_function);

      var training_accuracy = new List<double>();
      var validation_accuracy = new List<double>();
      for (int epoch = 0; epoch < max_epochs; epoch++) {

        // training phase
        var epoch_training_error = 0.0;
        var train_indices = Util.shuffled_indices(train_features.Length);
        var pos = 0;
        var num_batches = 0;
        while (pos < train_indices.Length) {
          var pos_end = Math.Min(pos + batch_size, train_indices.Length);
          var minibatch_images = Util.get_tensors(features.Shape, train_features, train_indices, pos, pos_end, computeDevice);
          var minibatch_labels = Util.get_tensors(labels.Shape, train_labels, train_indices, pos, pos_end, computeDevice);
          var feed_dictionary = new feed_t() { { features, minibatch_images }, { labels, minibatch_labels } };
          trainer.TrainMinibatch(feed_dictionary, false, computeDevice);
          epoch_training_error += trainer.PreviousMinibatchEvaluationAverage();
          num_batches++;
          pos = pos_end;
        }
        epoch_training_error /= num_batches;
        training_accuracy.Add(1.0-epoch_training_error);

        // evaluation phase
        var epoch_validation_error = 0.0;
        num_batches = 0;
        pos = 0;
        while (pos < validation_features.Length) {
          var pos_end = Math.Min(pos + batch_size, validation_features.Length);
          var minibatch_images = Util.get_tensors(features.Shape, validation_features, pos, pos_end, computeDevice);
          var minibatch_labels = Util.get_tensors(labels.Shape, validation_labels, pos, pos_end, computeDevice);
          var feed_dictionary = new test_feed_t() { { features, minibatch_images }, { labels, minibatch_labels } };
          epoch_validation_error += evaluator.TestMinibatch(feed_dictionary, computeDevice);
          pos = pos_end;
          num_batches++;
        }
        epoch_validation_error /= num_batches;
        validation_accuracy.Add(1.0-epoch_validation_error);

        Console.WriteLine($"Epoch {epoch + 1:D2}/{max_epochs}, training_accuracy={1.0-epoch_training_error:F3}, validation accuracy:{1-epoch_validation_error:F3}");

        if ( epoch_training_error<0.001 ) { break; }
      }

      return new List<List<double>>() { training_accuracy, validation_accuracy };
    }

    List<List<double>> train_with_augmentation(bool use_finetuning) {
      var labels = CNTK.Variable.InputVariable(new int[] { 2 }, CNTK.DataType.Float, "labels");
      var features = CNTK.Variable.InputVariable(new int[] { 150, 150, 3 }, CNTK.DataType.Float, "features");
      var scalar_factor = CNTK.Constant.Scalar<float>((float)(1.0 / 255.0), computeDevice);
      var scaled_features = CNTK.CNTKLib.ElementTimes(scalar_factor, features);

      var conv_base = VGG16.get_model(scaled_features, computeDevice, use_finetuning);
      var model = Util.Dense(conv_base, 256, computeDevice);
      model = CNTK.CNTKLib.ReLU(model);
      model = CNTK.CNTKLib.Dropout(model, 0.5);
      model = Util.Dense(model, 2, computeDevice);

      var loss_function = CNTK.CNTKLib.CrossEntropyWithSoftmax(model.Output, labels);
      var accuracy_function = CNTK.CNTKLib.ClassificationError(model.Output, labels);

      var pv = new CNTK.ParameterVector((System.Collections.ICollection)model.Parameters());
      var learner = CNTK.CNTKLib.AdamLearner(pv, new CNTK.TrainingParameterScheduleDouble(0.0001, 1), new CNTK.TrainingParameterScheduleDouble(0.99, 1));
      var trainer = CNTK.Trainer.CreateTrainer(model, loss_function, accuracy_function, new CNTK.Learner[] { learner });
      var evaluator = CNTK.CNTKLib.CreateEvaluator(accuracy_function);

      var train_minibatch_source = create_minibatch_source(features.Shape, 0, 1000, "train", is_training: true, use_augmentations: true);
      var validation_minibatch_source = create_minibatch_source(features.Shape, 1000, 500, "validation", is_training: false, use_augmentations: false);

      var train_featuresStreamInformation = train_minibatch_source.StreamInfo("features");
      var train_labelsStreamInformation = train_minibatch_source.StreamInfo("labels");
      var validation_featuresStreamInformation = validation_minibatch_source.StreamInfo("features");
      var validation_labelsStreamInformation = validation_minibatch_source.StreamInfo("labels");


      var training_accuracy = new List<double>();
      var validation_accuracy = new List<double>();
      for (int epoch = 0; epoch < max_epochs; epoch++) {
        var startTime = DateTime.Now;

        // training phase
        var epoch_training_error = 0.0;
        var pos = 0;
        var num_batches = 0;
        while (pos < 2000) {
          var pos_end = Math.Min(pos + batch_size, 2000);
          var minibatch_data = train_minibatch_source.GetNextMinibatch((uint)(pos_end - pos), computeDevice);
          var feed_dictionary = new batch_t() {
            { features, minibatch_data[train_featuresStreamInformation] },
            { labels, minibatch_data[train_labelsStreamInformation]}
          };
          trainer.TrainMinibatch(feed_dictionary, computeDevice);
          epoch_training_error += trainer.PreviousMinibatchEvaluationAverage();
          num_batches++;
          pos = pos_end;
        }
        epoch_training_error /= num_batches;
        training_accuracy.Add(1.0 - epoch_training_error);

        // evaluation phase
        var epoch_validation_error = 0.0;
        num_batches = 0;
        pos = 0;
        while (pos < 1000) {
          var pos_end = Math.Min(pos + batch_size, 1000);
          var minibatch_data = validation_minibatch_source.GetNextMinibatch((uint)(pos_end - pos), computeDevice);
          var feed_dictionary = new CNTK.UnorderedMapVariableMinibatchData() {
            { features, minibatch_data[validation_featuresStreamInformation] },
            { labels, minibatch_data[validation_labelsStreamInformation]}
          };
          epoch_validation_error += evaluator.TestMinibatch(feed_dictionary);
          pos = pos_end;
          num_batches++;
        }
        epoch_validation_error /= num_batches;
        validation_accuracy.Add(1.0 - epoch_validation_error);

        var elapsedTime = DateTime.Now.Subtract(startTime);
        Console.WriteLine($"Epoch {epoch + 1:D2}/{max_epochs}, training_accuracy={1.0 - epoch_training_error:F3}, validation accuracy:{1 - epoch_validation_error:F3}, elapsed time={elapsedTime.TotalSeconds:F1} seconds");

        if (epoch_training_error < 0.001) { break; }
      }

      return new List<List<double>>() { training_accuracy, validation_accuracy };
    }

    void run() {
      Console.Title = "Ch_05_Using_A_Pretrained_Convnet";
      computeDevice = Util.get_compute_device();
      List<List<double>> results = null;

      var trainingMode = TrainingMode.AugmentationWithFinetuning;
      Console.WriteLine($"Training mode: {trainingMode}\n");

      switch ( trainingMode) {
        case TrainingMode.UseExtractedFeatures: results = train_with_extracted_features(); break;
        case TrainingMode.Augmentation: results = train_with_augmentation(use_finetuning: false); break;
        case TrainingMode.AugmentationWithFinetuning: results = train_with_augmentation(use_finetuning: true); break;
      }

      var wpfApp = new System.Windows.Application();
      var window = new PlotWindow(results);
      wpfApp.Run(window);
    }

    [STAThread]
    static void Main(string[] args) {
      new Program().run();
    }

    enum TrainingMode { UseExtractedFeatures, Augmentation, AugmentationWithFinetuning};

    readonly string DATASET_PATH = @"D:\kaggle_cats_dogs";
    readonly int batch_size = 20;
    readonly int max_epochs = 19;
    readonly int extracted_feature_length = 5 * 5 * 512;

    CNTK.DeviceDescriptor computeDevice;
  }
}
