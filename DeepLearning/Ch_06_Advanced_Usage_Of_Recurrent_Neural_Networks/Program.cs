using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;

namespace Ch_06_Advanced_Usage_Of_Recurrent_Neural_Networks {

  class PlotWindow : System.Windows.Window {

    public PlotWindow(List<List<double>> results) {
      var plotModel = new OxyPlot.PlotModel();
      plotModel.Title = "Training and Validation MSE";

      plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis() { Position = OxyPlot.Axes.AxisPosition.Left, Title = "MSE" });
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
      Content = plotView;
    }
  }

  class TemperaturePlotWindow : System.Windows.Window {
    public TemperaturePlotWindow(float[] temperature) {
      var plotModel = new OxyPlot.PlotModel();
      plotModel.Title = "Temperature";

      plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis() { Position = OxyPlot.Axes.AxisPosition.Left, Title = "Celcius" });
      //plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis() { Position = OxyPlot.Axes.AxisPosition.Bottom, Title = "Epochs" });

      var labels = new string[] { "Temperature" };
      var colors = new OxyPlot.OxyColor[] { OxyPlot.OxyColors.Blue };
      var lineSeries = new OxyPlot.Series.LineSeries();
      lineSeries.ItemsSource = temperature.Select((value, index) => new OxyPlot.DataPoint(index, value));
      lineSeries.Title = labels[0];
      lineSeries.Color = colors[0];
      plotModel.Series.Add(lineSeries);

      var plotView = new OxyPlot.Wpf.PlotView();
      plotView.Model = plotModel;

      Content = plotView;
    }
  }

  class Program {
    [STAThread]
    static void Main(string[] args) {
      new Program().run();
    }

    void run() {
      var wpfApp = new System.Windows.Application();
      var all_lines = download_and_read_lines();
      var float_data = load_data(all_lines);
      //visualize_data(float_data);
      normalize_data(float_data);
      //evaluate_naive_method(float_data);

      var computeDevice = Util.get_compute_device();
      //basic_machine_learning_approach_cntk(float_data, epochs: 20, computeDevice: computeDevice);
      //first_recurrent_baseline_cntk(float_data, epochs: 20, computeDevice: computeDevice);
      stacking_recurrent_layers_cntk(float_data, epochs: 20, computeDevice: computeDevice);
      wpfApp.Run();
    }

    void plot_results(List<List<double>> history, string title = "") {
      var pt = new PlotWindow(history) { Title = title };
      pt.Show();
    }

    List<List<double>> train_mse_cntk(bool sequence_mode, CNTK.Variable x, CNTK.Variable y, CNTK.Function model, GeneratorsInfo gi, int epochs, int steps_per_epoch, CNTK.DeviceDescriptor computeDevice) {

      var loss_function = CNTK.CNTKLib.SquaredError(model, y);
      var accuracy_function = loss_function;

      var lr = 0.001;
      var parameterVector = new CNTK.ParameterVector((System.Collections.ICollection)model.Parameters());
      var learner = CNTK.CNTKLib.AdamLearner(parameterVector,
        new CNTK.TrainingParameterScheduleDouble(lr /*, (uint)batch_size*/),
        new CNTK.TrainingParameterScheduleDouble(0.9 /*, (uint)batch_size*/),
        unitGain: false);
      var trainer = CNTK.CNTKLib.CreateTrainer(model, loss_function, accuracy_function, new CNTK.LearnerVector() { learner });
      var evaluator = CNTK.CNTKLib.CreateEvaluator(accuracy_function);
      var history = fit_generator(sequence_mode, x, y, model, trainer, evaluator, gi, epochs, steps_per_epoch, computeDevice);
      return history;
    }

    CNTK.Value create_x_minibatch(bool sequence_mode, CNTK.Variable x, GeneratorsInfo gi, SamplesTargets st, CNTK.DeviceDescriptor computeDevice) {
      if ( sequence_mode == false ) {
        return CNTK.Value.CreateBatch(x.Shape, st.samples, computeDevice);
      }
      var sequence_length = gi.lookback / gi.step;
      var minibatch_size = st.samples.Length / sequence_length / gi.num_records;
      var x_shape = CNTK.NDShape.CreateNDShape(new int[] { gi.num_records, sequence_length, minibatch_size });
      var ndArrayView = new CNTK.NDArrayView(x_shape, st.samples, computeDevice, readOnly: true);
      return new CNTK.Value(ndArrayView);
    }

    List<List<double>> fit_generator(bool sequence_mode, CNTK.Variable x, CNTK.Variable y, CNTK.Function model, CNTK.Trainer trainer, CNTK.Evaluator evaluator, GeneratorsInfo gi, int epochs, int steps_per_epoch, CNTK.DeviceDescriptor computeDevice) {
      var history = new List<List<double>>() { new List<double>(), new List<double>() };

      var train_enumerator = gi.train_gen.GetEnumerator();
      var val_enumerator = gi.val_gen.GetEnumerator();

      var x_minibatch_dims = new List<int>(x.Shape.Dimensions);
      if ( sequence_mode==false ) {
        x_minibatch_dims.Add(gi.batch_size);
      }

      for (int current_epoch = 0; current_epoch < epochs; current_epoch++) {
        var epoch_start_time = DateTime.Now;

        var epoch_training_error = 0.0;
        {
          var num_total_samples = 0;
          for (int s = 0; s < steps_per_epoch; s++) {
            train_enumerator.MoveNext();
            var st = train_enumerator.Current;
            var x_minibatch = create_x_minibatch(sequence_mode, x, gi, st, computeDevice);
            var y_minibatch = CNTK.Value.CreateBatch(y.Shape, st.targets, computeDevice);

            var feed_dictionary = new Dictionary<CNTK.Variable, CNTK.Value> { { x, x_minibatch }, { y, y_minibatch } };
            bool isSweepEndInArguments = (s == (steps_per_epoch - 1));
            trainer.TrainMinibatch(feed_dictionary, isSweepEndInArguments, computeDevice);
            var minibatch_metric = trainer.PreviousMinibatchEvaluationAverage();
            epoch_training_error += minibatch_metric * st.targets.Length;
            num_total_samples += st.targets.Length;
            x_minibatch.Erase();
            y_minibatch.Erase();
          }
          epoch_training_error /= num_total_samples;
        }
        history[0].Add(epoch_training_error);

        var epoch_validation_error = 0.0;
        {
          var num_total_samples = 0;
          for (int s = 0; s < gi.val_steps; s++) {
            val_enumerator.MoveNext();
            var st = val_enumerator.Current;
            var x_minibatch = create_x_minibatch(sequence_mode, x, gi, st, computeDevice);
            var y_minibatch = CNTK.Value.CreateBatch(y.Shape, st.targets, computeDevice);
            var feed_dictionary = new CNTK.UnorderedMapVariableValuePtr() { { x, x_minibatch }, { y, y_minibatch } };
            var minibatch_metric = evaluator.TestMinibatch(feed_dictionary, computeDevice);
            epoch_validation_error += minibatch_metric * st.targets.Length;
            num_total_samples += st.targets.Length;
            x_minibatch.Erase();
            y_minibatch.Erase();
          }
          epoch_validation_error /= num_total_samples;
        }
        history[1].Add(epoch_validation_error);

        var elapsedTime = DateTime.Now.Subtract(epoch_start_time);
        Console.WriteLine($"Epoch {current_epoch + 1:D2}/{epochs}, Elapsed time: {elapsedTime.TotalSeconds:F3} seconds. " +
          $"Training Error: {epoch_training_error:F3}. Validation Error: {epoch_validation_error:F3}.");
      }

      return history;
    }

    CNTK.Function create_model_cntk(int model_type, int[] input_shape, CNTK.DeviceDescriptor computeDevice) {
      CNTK.Function model = null;
      if (model_type == 0) {
        var dynamic_axes = new List<CNTK.Axis>() { CNTK.Axis.DefaultDynamicAxis(), CNTK.Axis.DefaultBatchAxis() };
        var x_placeholder = CNTK.Variable.PlaceholderVariable(input_shape, dynamic_axes);
        model = Util.Dense(x_placeholder, 32, computeDevice);
        model = CNTK.CNTKLib.ReLU(model);
        model = Util.Dense(model, 1, computeDevice);
      }
      else if  ( (model_type==1) || (model_type == 2) ) {
        var filename = $"ch6-3_model_type_{model_type}.model";
        model = CNTK.Function.Load(filename, computeDevice);
        Console.WriteLine("Loaded " + filename);
      }
      return model;
    }

    void stacking_recurrent_layers_cntk(float[][] float_data, int epochs, CNTK.DeviceDescriptor computeDevice) {
      rnn_cntk(2, float_data, epochs, computeDevice, "Stacking Recurrent Layers");
    }

    void first_recurrent_baseline_cntk(float[][] float_data, int epochs, CNTK.DeviceDescriptor computeDevice) {
      rnn_cntk(1, float_data, epochs, computeDevice, "First Recurrent Baseline");
    }

    void rnn_cntk(int model_type, float[][] float_data, int epochs, CNTK.DeviceDescriptor computeDevice, string title) { 
      var gi = create_generators(float_data);
      var input_shape = new int[] { float_data[0].Length };
      var model = create_model_cntk(model_type, input_shape, computeDevice);

      var x = CNTK.Variable.InputVariable(input_shape, CNTK.DataType.Float, name: "x");
      model.ReplacePlaceholders(new CNTK.UnorderedMapVariableVariable() { { model.Placeholders()[0], x } });

      var y = CNTK.Variable.InputVariable(new CNTK.NDShape(0), CNTK.DataType.Float, name: "y", dynamicAxes: new List<CNTK.Axis>() { CNTK.Axis.DefaultBatchAxis() });
      var history = train_mse_cntk(true, x, y, model, gi, epochs, steps_per_epoch: 500, computeDevice: computeDevice);
      plot_results(history, title);
    }

    void basic_machine_learning_approach_cntk(float[][] float_data, int epochs, CNTK.DeviceDescriptor computeDevice) {
      var gi = create_generators(float_data);
      var input_shape = new int[] { gi.lookback / gi.step, float_data[0].Length };
      var model = create_model_cntk(0, input_shape, computeDevice);

      var x = CNTK.Variable.InputVariable(input_shape, CNTK.DataType.Float, name: "x");
      model.ReplacePlaceholders(new CNTK.UnorderedMapVariableVariable() { { model.Placeholders()[0], x } });

      var y = CNTK.Variable.InputVariable(new CNTK.NDShape(0), CNTK.DataType.Float, name: "y");
      var history = train_mse_cntk(false, x, y, model, gi, epochs, steps_per_epoch: 500, computeDevice: computeDevice);
      plot_results(history, title: "Basic Machine Learning Approach");
    }

    class SamplesTargets {
      public float[] samples;
      public float[] targets;
      public int num_strides;
    }

    class GeneratorsInfo {
      public IEnumerable<SamplesTargets> train_gen;
      public IEnumerable<SamplesTargets> val_gen;
      public IEnumerable<SamplesTargets> test_gen;
      public int val_steps;
      public int test_steps;
      public int lookback;
      public int step;
      public int batch_size;
      public int delay;
      public int num_records;
    }

    GeneratorsInfo create_generators(float[][] float_data) {
      var rtrn = new GeneratorsInfo();

      rtrn.lookback = 1440;
      rtrn.step = 6;
      rtrn.delay = 144;
      rtrn.batch_size = 128;
      rtrn.num_records = float_data[0].Length;

      rtrn.train_gen = generator(float_data,
                      lookback: rtrn.lookback,
                      delay: rtrn.delay,
                      min_index: 0,
                      max_index: 200000,
                      shuffle: true,
                      step: rtrn.step,
                      batch_size: rtrn.batch_size);

      rtrn.val_gen = generator(float_data,
                          lookback: rtrn.lookback,
                          delay: rtrn.delay,
                          min_index: 200001,
                          max_index: 300000,
                          step: rtrn.step,
                          batch_size: rtrn.batch_size);

      rtrn.test_gen = generator(float_data,
                           lookback: rtrn.lookback,
                           delay: rtrn.delay,
                           min_index: 300001,
                           max_index: -1,
                           step: rtrn.step,
                           batch_size: rtrn.batch_size);

      // This is how many steps to draw from `val_gen`
      // in order to see the whole validation set:
      rtrn.val_steps = (300000 - 200001 - rtrn.lookback) / rtrn.batch_size;


      // This is how many steps to draw from `test_gen`
      // in order to see the whole test set:
      rtrn.test_steps = (float_data.Length - 300001 - rtrn.lookback) / rtrn.batch_size;

      return rtrn;
    }

    IEnumerable<SamplesTargets> generator(float[][] data, int lookback, int delay, int min_index, int max_index, bool shuffle = false, int batch_size = 128, int step = 6, bool reverse = false) {
      if (reverse) { throw new NotImplementedException(); }
      var random = new Random();
      if (max_index < 0) {
        max_index = data.Length - delay - 1;
      }
      var i = min_index + lookback;
      while (true) {
        int[] rows = null;
        if (shuffle) {
          rows = new int[batch_size];
          var range = max_index - (min_index + lookback);
          for (int k = 0; k < batch_size; k++) { rows[k] = random.Next(range) + (min_index + lookback); }
        }
        else {
          if (i + batch_size >= max_index) {
            i = min_index + lookback;
          }
          var num_rows = Math.Min(i + batch_size, max_index) - i;
          rows = new int[num_rows];
          for (int k = 0; k < num_rows; k++) { rows[k] = k + i; }
          i += num_rows;
        }

        var st = new SamplesTargets();
        st.num_strides = (int)(lookback / step);
        st.samples = new float[rows.Length * st.num_strides * data[0].Length];
        st.targets = new float[rows.Length];

        var samples_index = 0;
        var num_bytes_in_block_row = data[0].Length * sizeof(float);
        for (int j = 0; j < rows.Length; j++) {
          for (int s = rows[j] - lookback; s < rows[j]; s += step) {
            Buffer.BlockCopy(data[s], 0, st.samples, samples_index * num_bytes_in_block_row, num_bytes_in_block_row);
            samples_index++;
          }
          st.targets[j] = data[rows[j] + delay][1];
        }

        yield return st;
      }
      /**
   
        if reverse is True:
            yield samples[:, ::-1, :], targets
       */
    }

    void evaluate_naive_method(float[][] float_data) {
      var gi = create_generators(float_data);
      var batch_maes = new List<float>();
      var val_enumerator = gi.val_gen.GetEnumerator();

      for (int i = 0; i < gi.val_steps; i++) {
        if (val_enumerator.MoveNext() == false) { break; }
        var st = val_enumerator.Current;
        var num_samples_in_batch = st.num_strides * float_data[0].Length;
        var offset = (st.num_strides - 1) * float_data[0].Length + 1;
        var preds = Enumerable.Range(0, gi.batch_size).Select(x => st.samples[x * num_samples_in_batch + offset]);
        var mae = preds.Zip(st.targets, (l, n) => Math.Abs(l - n)).Average();
        batch_maes.Add(mae);
      }
      var mean_batch_maes = batch_maes.Average();
      Console.WriteLine(mean_batch_maes);
    }

    void normalize_data(float[][] float_data) {
      var N = 200000;
      var num_columns = float_data[0].Length;
      for (int column = 0; column < num_columns; column++) {
        var mean = float_data.Take(N).Select(v => v[column]).Average();
        for (int i = 0; i < float_data.Length; i++) { float_data[i][column] -= mean; }
        var sum_squares = float_data.Take(N).Sum((v) => Math.Pow(v[column], 2));
        var std = (float)Math.Sqrt(sum_squares / (N - 1));
        for (int i = 0; i < float_data.Length; i++) { float_data[i][column] /= std; }
      }
    }

    void visualize_data(float[][] float_data) {
      var temperature = float_data.Select((x) => x[1]).ToArray();
      var window_0 = new TemperaturePlotWindow(temperature) { Title = "Figure 0" };
      window_0.Show();
      var temperature_range = temperature.Take(1440).ToArray();
      var window_1 = new TemperaturePlotWindow(temperature_range) { Title = "Figure 1" };
      window_1.Show();
    }

    float[][] load_data(List<string> all_lines) {
      var float_data = new float[all_lines.Count][];
      for (int i = 0; i < all_lines.Count; i++) {
        var current_line = all_lines[i];
        float_data[i] = current_line.Split(',').Skip(1).Select((x) => float.Parse(x)).ToArray();
      }
      return float_data;
    }

    List<string> download_and_read_lines() {
      System.Net.ServicePointManager.SecurityProtocol = System.Net.SecurityProtocolType.Tls12;
      var url_format = "https://www.bgc-jena.mpg.de/wetter/mpi_roof_{0}{1}.zip";
      var csv_filepaths = new List<string>();

      // step 1: download .zip files, and extract them to .csv files
      for (int year = 2009; year < 2017; year++) {
        foreach (var c in new char[] { 'a', 'b' }) {
          var url = String.Format(url_format, year, c);
          var zip_filepath = url.Split(new char[] { '/' }).Last();
          zip_filepath = Util.fullpathForDownloadedFile("roof_data", zip_filepath);
          var csv_filepath = zip_filepath.Replace(".zip", ".csv");
          if (System.IO.File.Exists(csv_filepath)) {
            csv_filepaths.Add(csv_filepath);
            continue;
          }
          if (System.IO.File.Exists(zip_filepath) == false) {
            var success = FromStackOverflow.FileDownloader.DownloadFile(url, zip_filepath, timeoutInMilliSec: 360000);
            if (!success) {
              Console.WriteLine("Could not download " + url);
              continue;
            }
          }
          var basepath = System.IO.Path.GetDirectoryName(zip_filepath);
          System.IO.Compression.ZipFile.ExtractToDirectory(zip_filepath, basepath);
          csv_filepaths.Add(csv_filepath);
        }
      }

      // step 2: read all .csv files, skipping the first line
      var all_lines = new List<string>();
      foreach (var csv_filepath in csv_filepaths) {
        var file_lines = System.IO.File.ReadAllLines(csv_filepath);
        for (int i = 1; i < file_lines.Length; i++) {
          var comma_pos = file_lines[i].IndexOf(',');
          all_lines.Add(file_lines[i].Substring(comma_pos + 1));
        }
      }
      Console.WriteLine(all_lines.Count);
      return all_lines;
    }
  }
}
