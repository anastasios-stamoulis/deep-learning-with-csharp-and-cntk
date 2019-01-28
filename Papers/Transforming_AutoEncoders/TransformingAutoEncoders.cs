using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;
using CC = CNTK.CNTKLib;
using C = CNTK;

namespace Transforming_AutoEncoders {
  class TransformingAutoEncoders {
    static void Main(string[] args) {
      new TransformingAutoEncoders().run();
    }

    C.Function create_capsule(string name, int[] input_shape, int[] extra_input_shape, int recognizer_dim, int generator_dim) {
      var input_dim = Util.np_prod(input_shape);
      var x = Util.placeholderVariable(input_shape, "x");
      var extra_input = Util.placeholderVariable(extra_input_shape, "extra_input");

      var x_flat = CC.Flatten(x);
      var recognition = Layers.Dense(x_flat, recognizer_dim, computeDevice, "recognition_layer", CC.Sigmoid);
      var probability = Layers.Dense(recognition, 1, computeDevice, "probability", CC.Sigmoid);
      var learnt_transformation = Layers.Dense(recognition, 2, computeDevice, "xy_prediction");
      var learnt_transformation_extended = CC.Plus(learnt_transformation, extra_input, "learnt_transformation_extended");

      var generation = Layers.Dense(learnt_transformation_extended, generator_dim, computeDevice, "generator_layer", CC.Sigmoid);
      var out_flat = Layers.Dense(generation, input_dim, computeDevice, "output");
      out_flat = CC.ElementTimes(out_flat, probability);
      var output = CC.Reshape(out_flat, input_shape);
      return output;
    }

    C.Function create_transforming_autoencoder(
      int num_capsules,
      int[] input_shape,
      int[] extra_input_shape,
      int recognizer_dim,
      int generator_dim) {

      var capsules_outputs = new List<C.Function>();

      for (int i = 0; i < num_capsules; i++) {
        var capsule = create_capsule($"capsule_{i}", input_shape, extra_input_shape, recognizer_dim, generator_dim);
        var capsule_output = Model.invoke_model(capsule, new C.Variable[] { imageVariable, transformationVariable });
        capsules_outputs.Add(capsule_output);
      }

      var inference = capsules_outputs[0];
      for (int i = 1; i < capsules_outputs.Count; i++) {
        inference = CC.Plus(inference, capsules_outputs[i]);
      }
      inference = CC.Sigmoid(inference, "autoencoder_inference");
      return inference;
    }

    void create_network() {
      imageVariable = Util.inputVariable(input_shape, "image");
      transformationVariable = Util.inputVariable(extra_input_shape, "transformation");
      transformedImageVariable = Util.inputVariable(input_shape, "transformed_image");
      network = create_transforming_autoencoder(num_capsules, input_shape, extra_input_shape, recognizer_dim, generator_dim);
      Logging.log_number_of_parameters(network, show_filters: false);

      var mse_normalizing_factor = C.Constant.Scalar(C.DataType.Float, 1.0 / network.Output.Shape.TotalSize, computeDevice);
      var squared_error = CC.SquaredError(network.Output, transformedImageVariable);
      var mse = CC.ElementTimes(squared_error, mse_normalizing_factor);
      loss_function = mse;
      eval_function = mse;

      learner = CC.AdamLearner(
        new C.ParameterVector(network.Parameters().ToArray()),
        new C.TrainingParameterScheduleDouble(learning_rate * batch_size, (uint)batch_size),
        new C.TrainingParameterScheduleDouble(0.9),
        true,
        new C.TrainingParameterScheduleDouble(0.99));

      trainer = CC.CreateTrainer(network, loss_function, new C.LearnerVector(new C.Learner[] { learner }));

      evaluator = CC.CreateEvaluator(eval_function);
    }

    OpenCvSharp.Mat convertFloatRowTo2D(float[] img, int rows, int cols) {
      var srcImg = new byte[img.Length];
      for (int i = 0; i < srcImg.Length; i++) {
        srcImg[i] = (byte)(255 * img[i]);
      }
      var mat = new OpenCvSharp.MatOfByte(rows, cols, srcImg);
      return mat;
    }

    double extractLoss(string filename) {
      var prefix = "val_metric_";
      var pos = filename.LastIndexOf(prefix);
      if ( pos<0 ) { return 0; }
      var dotPos = filename.LastIndexOf(".model");
      if ( dotPos<0 ) { return 0; }
      var startPos = pos + prefix.Length;
      var lossMetricAsString = filename.Substring(startPos, dotPos - startPos);
      var lossMetric = double.Parse(lossMetricAsString);
      return lossMetric;
    }

    C.Function load_best_network() {
      //var allFiles = System.IO.Directory.GetFiles(".", "*.model");
      //var lossMetrics = allFiles.Select((v) => -extractLoss(v)).ToArray();
      //var negative_max_index = Util.argmax(lossMetrics);
      //var min_loss = lossMetrics[negative_max_index];
      //var model_filename = allFiles[negative_max_index];
      var model_filename = "D:\\OneDrive\\Cloud\\backup\\cntk_0152_train_metric_0.0164_0.0164_val_metric_0.0174.model";

      var model = C.Function.Load(model_filename, computeDevice);
      Console.WriteLine($"Loaded {model_filename}");
      return model;
    }

    bool show_translations(bool use_network) {
      if ( use_network ) {
        network = load_best_network();
      }
      var mnist = new Datasets.MNIST();
      var image_size = new int[] { input_shape[0], input_shape[1] };
      var allData = new Data(mnist.train_images, image_size).All();

      for (int index = 0; index < 1000; index++) {
        var img = allData[0][index];
        var T = allData[1][index];
        var img_t = allData[2][index];

        var label = mnist.train_labels[index];
        var digit = Util.argmax(label);
        var mat = convertFloatRowTo2D(img, 28, 28);

        var window_name = $"mnist_{index}_Digit_{digit}";
        // allow the window to be resized
        OpenCvSharp.Cv2.NamedWindow(window_name, OpenCvSharp.WindowMode.Normal);
        OpenCvSharp.Cv2.ImShow(window_name, mat);

        var mat_t = convertFloatRowTo2D(img_t, 28, 28);
        var window_name_T = $"{T[0]}_{T[1]}";
        OpenCvSharp.Cv2.NamedWindow(window_name_T, OpenCvSharp.WindowMode.Normal);
        OpenCvSharp.Cv2.ImShow(window_name_T, mat_t);

        var predicted_mat_t = new OpenCvSharp.Mat();
        if (network != null) {
          var prediction = Model.predict(network, new float[][] { img, T }, computeDevice);
          var predicted_img_t = prediction[0].ToArray();
          predicted_mat_t = convertFloatRowTo2D(predicted_img_t, 28, 28);
          var window_name_predicted_T = $"predicted_{T[0]}_{T[1]}";
          OpenCvSharp.Cv2.NamedWindow(window_name_predicted_T, OpenCvSharp.WindowMode.Normal);
          OpenCvSharp.Cv2.ImShow(window_name_predicted_T, predicted_mat_t);
        }

        OpenCvSharp.Cv2.WaitKey();
        OpenCvSharp.Cv2.DestroyAllWindows();
        //OpenCvSharp.Cv2.DestroyWindow(window_name);
        //OpenCvSharp.Cv2.DestroyWindow(window_name_T);

        GC.KeepAlive(mat);
        GC.KeepAlive(mat_t);
        GC.KeepAlive(predicted_mat_t);
      }
      return false;
    }

    void run() {
      //if ( show_translations(use_network: true)==false ) { return; }

      Console.Title = "Transforming AutoEncoders";
      Console.WriteLine("Using: " + computeDevice.AsString());

      create_network();

      var mnist = new Datasets.MNIST();
      var image_size = new int[] { input_shape[0], input_shape[1] };
      var train_data = new Data(mnist.train_images, image_size);
      var test_data = new Data(mnist.test_images, image_size);

      train_network(train_data, test_data);
    }

    void train_network(Data train_data, Data test_data) {
      // For a discussion on the MNIST training/validation sets, 
      // see https://github.com/keras-team/keras/issues/1753
      var trainSteps = train_data.numSteps(batch_size);
      var validationSteps = test_data.numSteps(batch_size);

      var inner_trainGenerator = new InMemoryMiniBatchGenerator(
        train_data.All(), allInputVariables(), 
        batch_size, true, false, "training");
      var trainGenerator = new MultiThreadedGenerator(4, inner_trainGenerator);      
      
      var inner_validationGenerator = new InMemoryMiniBatchGenerator(
        test_data.All(), allInputVariables(), 
        batch_size, true, false, "validation");
      var validationGenerator = new MultiThreadedGenerator(4, inner_validationGenerator);

      Model.fit_generator(network, learner, trainer, evaluator, batch_size, epochs,
        trainGenerator, trainSteps, validationGenerator, validationSteps, computeDevice, prefix: "ta_");

      trainGenerator.Dispose(); trainGenerator = null;
      validationGenerator.Dispose(); validationGenerator = null;
    }

    class Data {
      readonly float[][] images;
      readonly float[][] transformations;
      readonly float[][] transformed_images;

      public float[][][] All() { return new float[][][] { images, transformations, transformed_images }; }
      public int numSteps(int batch_size) { return images.Length / batch_size; }

      public Data(float[][] images, int[] image_size) {
        this.images = images;
        transformations = new float[images.Length][];
        transformed_images = new float[images.Length][];
        int max_translation = 5;
        for (int i = 0; i < images.Length; i++) {
          var translation_x = rng.Next(-max_translation, max_translation);
          var translation_y = rng.Next(-max_translation, max_translation);
          transformations[i] = new float[2] { translation_x, translation_y };
          transformed_images[i] = Augmentations.translate(images[i], image_size[0], image_size[1], translation_x, translation_y);
        }
      }
    }

    C.Variable[] allInputVariables() {
      return new C.Variable[] { imageVariable, transformationVariable, transformedImageVariable };
    }
    
    readonly int[] input_shape = new int[] { 28, 28, 1 };
    readonly int[] extra_input_shape = new int[] { 2 };
    readonly int recognizer_dim = 30;
    readonly int generator_dim = 30;
    readonly int num_capsules = 10;
    readonly int batch_size = 64;
    readonly int epochs = 10000;
    readonly double learning_rate = 1e-3;
    
    readonly C.DeviceDescriptor computeDevice = Util.get_compute_device();
    static readonly Random rng = new Random(Seed: 2019);

    C.Variable imageVariable;
    C.Variable transformationVariable;
    C.Variable transformedImageVariable;

    C.Function network;
    C.Function loss_function;
    C.Function eval_function;
    C.Learner learner;
    C.Trainer trainer;
    C.Evaluator evaluator;
  }
}
