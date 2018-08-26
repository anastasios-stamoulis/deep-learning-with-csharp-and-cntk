using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;

namespace Ch_08_Generating_Images_With_VAEs {
  class Program {
    static void Main(string[] args) {
      new Program().run();
    }

    static readonly int[] img_shape = { 28, 28 };
    static readonly int latent_dim = 2;
    static readonly string model_filename = "trained_model.cntk";

    public Program() {
      Console.Title = "Generating Images With VAEs";
      computeDevice = Util.get_compute_device();
    }

    void plotImages(IList<IList<float>> src_f) {
      var h = img_shape[0];
      var w = img_shape[1];
      System.Diagnostics.Debug.Assert(w * h == src_f[0].Count);
      var matBuffer = new byte[src_f.Count * w * h];
      var gridLength = (int)Math.Sqrt(src_f.Count);
      for (int imageIndex = 0; imageIndex < src_f.Count; imageIndex++) {
        var img_f = src_f[imageIndex].ToArray();
        System.Diagnostics.Debug.Assert(img_f.Length == w * h);

        var blockRow = imageIndex / gridLength;
        var blockColumn = imageIndex % gridLength;        
        for (int i = 0; i < img_f.Length; i++) {
          var v = (byte)(Math.Max(0, Math.Min(img_f[i], 1)) * 255);
          var digit_row = i / w;
          var digit_col = i % w;

          var row = blockRow * h + digit_row;
          var col = blockColumn * w + digit_col;

          var pos = row * gridLength * w + col;
          matBuffer[pos] = v;
        }
      }
    
      var mat = new OpenCvSharp.Mat(gridLength*h, gridLength*w, OpenCvSharp.MatType.CV_8UC1, matBuffer, gridLength*w);
      OpenCvSharp.Cv2.ImShow("Generated Images", mat);
      OpenCvSharp.Cv2.WaitKey(0);
    }

    void run() {
      train();
      generateImages();
      Console.WriteLine("All Done");
    }

    void generateImages() {
      if ( model==null ) {
        System.Diagnostics.Debug.Assert(System.IO.File.Exists(model_filename));
        model = CNTK.Function.Load(model_filename, computeDevice);
      }

      var node_z = model.FindByName("input_z");
      var decoder_start = Util.find_function_with_input(model, node_z);
      var z_sample_var = CNTK.Variable.InputVariable(node_z.Output.Shape, CNTK.DataType.Float, "z_sample");
      var replacements = new Dictionary<CNTK.Variable, CNTK.Variable>() { {node_z, z_sample_var } };
      var decoder = model.Clone(CNTK.ParameterCloningMethod.Freeze, replacements);

      var n = 15; // figure with 15x15 digits
      var xy_buffer = new float[n * n * 2];
      var sample_start = -2f;
      var sample_interval_width = 4f;
      for (int i=0, pos=0; i<n; i++) {
        for (int j=0; j<n; j++) {
          xy_buffer[pos++] = sample_start + (sample_interval_width / (n-1)) * i;
          xy_buffer[pos++] = sample_start + (sample_interval_width/ (n - 1)) * j;
        }
      }

      var ndArrayView = new CNTK.NDArrayView(new int[] {2, 1, xy_buffer.Length/2}, xy_buffer, computeDevice, true);
      var value = new CNTK.Value(ndArrayView);
      var inputs_dir = new Dictionary<CNTK.Variable, CNTK.Value>() { { z_sample_var, value} };
      var outputs_dir = new Dictionary<CNTK.Variable, CNTK.Value>() { { decoder.Output, null } };
      decoder.Evaluate(inputs_dir, outputs_dir, computeDevice);
      var values = outputs_dir[decoder.Output].GetDenseData<float>(decoder.Output);
      plotImages(values);
    }

    void load_data() {
      if (!System.IO.File.Exists("train_images.bin")) {
        System.IO.Compression.ZipFile.ExtractToDirectory("mnist_data.zip", ".");
      }
      x_train = Util.load_binary_file("train_images.bin", 60000, 28 * 28);
      x_test = Util.load_binary_file("test_images.bin", 10000, 28 * 28);
      y_train = Util.load_binary_file("train_labels.bin", 60000, 10);
      y_test = Util.load_binary_file("test_labels.bin", 60000, 10);
      Console.WriteLine("Done with loading data\n");
    }

    void create_model() {
      input_img = CNTK.Variable.InputVariable(new int[] { img_shape[0], img_shape[1], 1 }, CNTK.DataType.Float);

      // encoder part
      var x = Util.Convolution2DWithReLU(input_img, 32, new int[] { 3, 3 }, computeDevice, true, outputName: "dense_0");
      x = Util.Convolution2DWithReLU(x, 64, new int[] { 3, 3 }, computeDevice, true, strides: new int[] { 2, 2 }, outputName: "dense_1");
      x = Util.Convolution2DWithReLU(x, 64, new int[] { 3, 3 }, computeDevice, true, outputName: "dense_2");
      x = Util.Convolution2DWithReLU(x, 64, new int[] { 3, 3 }, computeDevice, true, outputName: "dense_3");
      var shape_before_dense = x.Output.Shape;
      x = CNTK.CNTKLib.ReLU(Util.Dense(x, 32, computeDevice));
     
      // latent variables
      var z_mean = Util.Dense(x, latent_dim, computeDevice);
      var z_log_var = Util.Dense(x, latent_dim, computeDevice);
      var epsilon = CNTK.CNTKLib.NormalRandom(new int[] { latent_dim }, CNTK.DataType.Float);
      var z = CNTK.CNTKLib.Plus(z_mean, CNTK.CNTKLib.ElementTimes(CNTK.CNTKLib.Exp(z_log_var), epsilon), "input_z");

      // decoder
      x = CNTK.CNTKLib.ReLU(Util.Dense(z, shape_before_dense.TotalSize, computeDevice), name: "decoder");
      x = CNTK.CNTKLib.Reshape(x, shape_before_dense);
      x = Util.ConvolutionTranspose(x, 
        computeDevice, 
        filter_shape: new int[] { 3, 3 }, 
        num_filters: 32, 
        use_padding: true,
        activation: CNTK.CNTKLib.ReLU,
        strides: new int[] { 2, 2 },
        output_shape: new int[] {28, 28});
      model = Util.Convolution2D(x, 1, new int[] { 3, 3, }, computeDevice, use_padding: true, activation: CNTK.CNTKLib.Sigmoid);

      // regularization metric
      var square_ = CNTK.CNTKLib.Square(z_mean);
      var exp_ = CNTK.CNTKLib.Exp(z_log_var);
      var constant_1 = CNTK.Constant.Scalar(CNTK.DataType.Float, 1.0);
      var diff_ = CNTK.CNTKLib.Plus(constant_1, z_log_var);
      diff_ = CNTK.CNTKLib.Minus(diff_, square_);
      diff_ = CNTK.CNTKLib.Minus(diff_, exp_);
      var constant_2 = CNTK.Constant.Scalar(CNTK.DataType.Float, -5e-4);
      var regularization_metric = CNTK.CNTKLib.ElementTimes(constant_2, CNTK.CNTKLib.ReduceMean(diff_, CNTK.Axis.AllStaticAxes()));

      // overall loss function
      var crossentropy_loss = CNTK.CNTKLib.BinaryCrossEntropy(model, input_img);
      crossentropy_loss = CNTK.CNTKLib.ReduceMean(crossentropy_loss, CNTK.Axis.AllStaticAxes());
      loss = CNTK.CNTKLib.Plus(crossentropy_loss, regularization_metric);

      Util.log_number_of_parameters(model);
    }

    void train() {
      if ( System.IO.File.Exists(model_filename)) {
        System.Console.WriteLine("Model exists: " + model_filename);
        return;
      }
      create_model();
      load_data();

      var pv = new CNTK.ParameterVector((System.Collections.ICollection)model.Parameters());
      var learner = CNTK.CNTKLib.AdamLearner(pv, new CNTK.TrainingParameterScheduleDouble(0.001), new CNTK.TrainingParameterScheduleDouble(0.9));
      var trainer = CNTK.CNTKLib.CreateTrainer(model, loss, loss, new CNTK.LearnerVector() { learner });
      var evaluator = CNTK.CNTKLib.CreateEvaluator(loss);
      
      var batch_size = 16;
      var epochs = 10;
      Console.WriteLine("Training on: " + computeDevice.AsString());
      for (int current_epoch = 0; current_epoch < epochs; current_epoch++) {
        var epoch_start_time = DateTime.Now;
        var epoch_training_loss = 0.0;
        {
          var train_indices = Util.shuffled_indices(x_train.Length);
          var pos = 0;
          while (pos < train_indices.Length) {
            var pos_end = Math.Min(pos + batch_size, train_indices.Length);
            var x_minibatch = Util.get_tensors(input_img.Shape, x_train, train_indices, pos, pos_end, computeDevice);
            var feed_dictionary = new Dictionary<CNTK.Variable, CNTK.Value> { { input_img, x_minibatch } };
            trainer.TrainMinibatch(feed_dictionary, false, computeDevice);
            epoch_training_loss += (pos_end - pos) * trainer.PreviousMinibatchLossAverage();
            pos = pos_end;
            x_minibatch.Erase(); x_minibatch.Dispose(); x_minibatch = null;
          }
          epoch_training_loss /= x_train.Length;
        }

        var epoch_validation_loss = 0.0;
        {
          var pos = 0;
          while (pos < x_test.Length) {
            var pos_end = Math.Min(pos + batch_size, x_test.Length);
            var x_minibatch = Util.get_tensors(input_img.Shape, x_test, pos, pos_end, computeDevice);
            var feed_dictionary = new CNTK.UnorderedMapVariableValuePtr() { { input_img, x_minibatch } };
            epoch_validation_loss += (pos_end - pos) * evaluator.TestMinibatch(feed_dictionary, computeDevice);
            pos = pos_end;
            x_minibatch.Erase(); x_minibatch.Dispose(); x_minibatch = null;
          }
          epoch_validation_loss /= x_test.Length;
        }

        var elapsedTime = DateTime.Now.Subtract(epoch_start_time);
        Console.Write($"Epoch {current_epoch + 1:D2}/{epochs}, Elapsed time: {elapsedTime.TotalSeconds:F3} seconds. ");
        Console.WriteLine($"Training loss: {epoch_training_loss:F3}. Validation Loss: {epoch_validation_loss:F3}.");
      }
      model.Save(model_filename);
    }

    float[][] x_train;
    float[][] y_train;
    float[][] x_test;
    float[][] y_test;
    CNTK.Function model;
    CNTK.Function loss;
    CNTK.Variable input_img;
    CNTK.DeviceDescriptor computeDevice;
  }
}
