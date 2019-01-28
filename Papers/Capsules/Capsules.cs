using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;
using CC = CNTK.CNTKLib;
using C = CNTK;

namespace Capsules {
  class MNISTExample {
    C.Function network;
    C.Function loss_function;
    C.Function eval_function;
    C.Learner learner;
    C.Trainer trainer;
    C.Evaluator evaluator;

    C.Variable imageVariable;
    C.Variable categoricalVariable;
    C.DeviceDescriptor computeDevice = Util.get_compute_device();

    public void run() {
      create_network();
      train_network();
    }

    const int batch_size = 64;

    void create_network() {
      Console.WriteLine("Compute Device: " + computeDevice.AsString());
      imageVariable = Util.inputVariable(new int[] { 28, 28, 1 }, "image_tensor");
      categoricalVariable = Util.inputVariable(new int[] { 10 }, "label_tensor");

      network = imageVariable;
      network = Layers.Convolution2D(network, 32, new int[] { 3, 3 }, computeDevice, CC.ReLU);
      network = CC.Pooling(network, C.PoolingType.Max, new int[] { 2, 2 }, new int[] { 2 });
      network = Layers.Convolution2D(network, 64, new int[] { 3, 3 }, computeDevice, CC.ReLU);
      network = CC.Pooling(network, C.PoolingType.Max, new int[] { 2, 2 }, new int[] { 2 });
      network = Layers.Convolution2D(network, 64, new int[] { 3, 3 }, computeDevice, CC.ReLU);
      network = Layers.Dense(network, 64, computeDevice, activation: CC.ReLU);
      network = Layers.Dense(network, 10, computeDevice);

      Logging.detailed_summary(network);
      Logging.log_number_of_parameters(network);

      loss_function = CC.CrossEntropyWithSoftmax(network, categoricalVariable);
      eval_function = CC.ClassificationError(network, categoricalVariable);

      learner = CC.AdamLearner(
        new C.ParameterVector(network.Parameters().ToArray()),
        new C.TrainingParameterScheduleDouble(0.001 * batch_size, (uint)batch_size),
        new C.TrainingParameterScheduleDouble(0.9),
        true,
        new C.TrainingParameterScheduleDouble(0.99));

      trainer = CC.CreateTrainer(network, loss_function, eval_function, new C.LearnerVector(new C.Learner[] { learner }));
      evaluator = CC.CreateEvaluator(eval_function);
    }

    void train_network() {
      var mnist = new Datasets.MNIST();
      var trainSteps = mnist.train_images.Length / batch_size;
      var validationSteps = mnist.test_images.Length / batch_size;

      var wrapped_trainGenerator = new InMemoryMiniBatchGenerator(
        new float[][][] { mnist.train_images, mnist.train_labels },
        new C.Variable[] { imageVariable, categoricalVariable },
        batch_size, true, false, "training");
      var trainGenerator = new MultiThreadedGenerator(4, wrapped_trainGenerator);
      
      var wrapped_testGenerator = new InMemoryMiniBatchGenerator(
        new float[][][] { mnist.test_images, mnist.test_labels },
        new C.Variable[] { imageVariable, categoricalVariable },
        batch_size, true, false, "testing");
      var validationGenerator = new MultiThreadedGenerator(4, wrapped_testGenerator);
      
      var epochs = 6;
      Model.fit_generator(
        network,
        learner,
        trainer,
        evaluator,
        batch_size,
        epochs,
        trainGenerator, trainSteps,
        validationGenerator, validationSteps,
        computeDevice, "mnist_");

      trainGenerator.Dispose(); trainGenerator = null;
      validationGenerator.Dispose(); validationGenerator = null;
    }
  }

  class Capsules {
    static void Main(string[] args) {
      //new MNISTExample().run();
      new Capsules().run();
    }

    void run() {
      Console.Title = "Capsules";
      Console.WriteLine($"Using {computeDevice.AsString()}");
      create_network();
      train_network();
    }

    void train_network() {
      var mnist = new Datasets.MNIST();
      var trainSteps = mnist.train_images.Length / batch_size;
      var validationSteps = mnist.test_images.Length / batch_size;

      var trainGenerator = new InMemoryMiniBatchGenerator(
        new float[][][] { mnist.train_images, mnist.train_labels },
        new C.Variable[] { imageVariable, categoricalLabel },
        batch_size, shuffle: true, only_once: false, name: "train");
      var mtTrainGenerator = new MultiThreadedGenerator(workers: 4, generator: trainGenerator);
      
      var validationGenerator = new InMemoryMiniBatchGenerator(
        new float[][][] { mnist.test_images, mnist.test_labels },
        new C.Variable[] { imageVariable, categoricalLabel },
        batch_size, shuffle: false, only_once: false, name: "validation");
      var mtValidationGenerator = new MultiThreadedGenerator(workers: 4, generator: validationGenerator);
          
      var epochs = 6;
      Model.fit_generator(
        network,
        learner,
        trainer,
        evaluator,
        batch_size,
        epochs,
        mtTrainGenerator, trainSteps,
        mtValidationGenerator, validationSteps,
        computeDevice, 
        prefix: "capsules_",
        trainingLossMetricName: "Training Loss",
        trainingEvaluationMetricName: "Training Error",
        validationMetricName: "Validation Error");

      mtTrainGenerator.Dispose(); 
      mtValidationGenerator.Dispose();
    }


    C.Function create_primary_cap(C.Function inputs, int dim_capsule, int n_channels, int[] kernel_size, int[] strides, bool pad) {
      var output = Layers.Convolution2D(
        inputs,
        dim_capsule * n_channels,
        kernel_size,
        computeDevice,
        strides: strides,
        use_padding: pad,
        name: "primarycap_conv2d");
      var outputShape = output.Output.Shape.Dimensions;
      System.Diagnostics.Debug.Assert((outputShape[2] == 256) && (outputShape[1] == 6) && (outputShape[0] == 6));

      var num_rows = (int)(Util.np_prod(outputShape.ToArray()) / dim_capsule);
      var target_shape = new int[] { num_rows, dim_capsule };
      var outputs = CC.Reshape(output, target_shape, name: "primarycap_reshape");
      var rtrn = squash(outputs, name: "primarycap_squash", axis: 1);
      return rtrn;
    }

    C.Function squash(C.Function vectors, string name, int axis) {
      var squared_values = CC.Square(vectors);
      var s_squared_sum = CC.ReduceSum(squared_values, new C.AxisVector(new C.Axis[] { new C.Axis(axis) }), keepDims: true);
      var epsilon = C.Constant.Scalar(C.DataType.Float, 1e-7, computeDevice);
      var one = C.Constant.Scalar(C.DataType.Float, 1.0, computeDevice);
      var normalize_factor = CC.Plus(CC.Sqrt(s_squared_sum), epsilon);
      var one_plus_s_squared_sum = CC.Plus(s_squared_sum, one);
      var scale = CC.ElementDivide(s_squared_sum, one_plus_s_squared_sum);
      scale = CC.ElementDivide(scale, normalize_factor);
      var result = CC.ElementTimes(scale, vectors, name);
      return result;
    }

    C.Function create_capsule_layer(C.Function inputs, int num_capsule, int dim_capsule, int routings, string name) {
      var inputs_shape = inputs.Output.Shape.Dimensions;
      var input_num_capsule = inputs_shape[0];
      var input_dim_capsule = inputs_shape[1];
      var W = new C.Parameter(
        new int[] { num_capsule, dim_capsule, input_num_capsule, input_dim_capsule },
        C.DataType.Float,
        CC.GlorotUniformInitializer(),
        computeDevice,
        name: "W");

      inputs = CC.Reshape(inputs, new int[] { 1, 1, input_num_capsule, input_dim_capsule }); // [1, 1, 1152, 8])
      var inputs_hat = CC.ElementTimes(W, inputs);
      inputs_hat = CC.ReduceSum(inputs_hat, new C.Axis(3));
      inputs_hat = CC.Squeeze(inputs_hat);

      C.Function outputs = null;
      var zeros = new C.Constant(new int[] { num_capsule, 1, input_num_capsule }, C.DataType.Float, 0, computeDevice);
      var b = CC.Combine(new C.VariableVector() { zeros });
      for (int i = 0; i < routings; i++) {
        var c = CC.Softmax(b, new C.Axis(0));
        var batch_dot_result = CC.ElementTimes(c, inputs_hat);
        batch_dot_result = CC.ReduceSum(batch_dot_result, new C.Axis(2));
        batch_dot_result = CC.Squeeze(batch_dot_result);
        outputs = squash(batch_dot_result, name: $"squashed_{i}", axis: 1);
        if (i < (routings - 1)) {
          outputs = CC.Reshape(outputs, new int[] { num_capsule, dim_capsule, 1 });
          batch_dot_result = CC.ElementTimes(outputs, inputs_hat);
          batch_dot_result = CC.ReduceSum(batch_dot_result, new C.Axis(1));
          b = CC.Plus(b, batch_dot_result);
        }
      }
      outputs = CC.Combine(new C.VariableVector() { outputs }, name);
      return outputs;
    }

    C.Function get_length_and_remove_last_dimension(C.Function x, string name) {
      var number_dimensions = x.Output.Shape.Dimensions.Count;
      x = CC.Square(x);
      var sum_entries = CC.ReduceSum(x, new C.Axis(number_dimensions - 1));
      var epsilon = C.Constant.Scalar(C.DataType.Float, 1e-7, computeDevice);
      x = CC.Sqrt(CC.Plus(sum_entries, epsilon));
      x = CC.Squeeze(x);
      return x;
    }

    C.Function get_mask_and_infer_from_last_dimension(C.Function inputs, C.Function mask) {
      if (mask == null) {
        var inputs_shape = inputs.Output.Shape.Dimensions.ToArray();
        var ndims = inputs_shape.Length - 1;
        var x = CC.Sqrt(CC.ReduceSum(CC.Square(inputs), new C.Axis(ndims - 1)));
        x = CC.Squeeze(x);
        System.Diagnostics.Debug.Assert(x.Output.Shape.Dimensions.Count == 1);
        x = CC.Argmax(x, new C.Axis(0));
        mask = CC.OneHotOp(x, numClass: (uint)inputs_shape[0], outputSparse: false, axis: new C.Axis(0));
      }
      mask = CC.Reshape(mask, mask.Output.Shape.AppendShape(new int[] { 1 }));
      var masked = CC.ElementTimes(inputs, mask);
      masked = CC.Flatten(masked);
      masked = CC.Squeeze(masked);
      return masked;
    }

    C.Function create_decoder(int[] digits_capsules_output_shape) {
      var decoder_input = Util.inputVariable(digits_capsules_output_shape);
      var decoder = Layers.Dense(decoder_input, 512, computeDevice, activation: CC.ReLU);
      decoder = Layers.Dense(decoder, 1024, computeDevice, activation: CC.ReLU);
      decoder = Layers.Dense(decoder, Util.np_prod(input_shape), computeDevice, activation: CC.Sigmoid);
      decoder = CC.Reshape(decoder, input_shape, name: "out_recon");
      return decoder;
    }

    void create_network() {

      imageVariable = Util.inputVariable(input_shape, "image");
      var conv1 = Layers.Convolution2D(
        imageVariable, 256, new int[] { 9, 9 }, computeDevice, 
        use_padding: false, activation: CC.ReLU, name: "conv1");

      var primarycaps = create_primary_cap(
        conv1, dim_capsule: 8, n_channels: 32, 
        kernel_size: new int[] { 9, 9 }, strides: new int[] { 2, 2 }, pad: false);

      var digitcaps = create_capsule_layer(
        primarycaps, num_capsule: 10, dim_capsule: 16, 
        routings: routings, name: "digitcaps");

      var out_caps = get_length_and_remove_last_dimension(digitcaps, name: "capsnet");

      categoricalLabel = Util.inputVariable(new int[] { 10 }, "label");
      var masked_by_y = get_mask_and_infer_from_last_dimension(digitcaps, CC.Combine(new C.VariableVector() { categoricalLabel }));
      var masked = get_mask_and_infer_from_last_dimension(digitcaps, null);

      var decoder = create_decoder(masked.Output.Shape.Dimensions.ToArray());
      var decoder_output_training = Model.invoke_model(decoder, new C.Variable[] { masked_by_y });
      var decoder_output_evaluation = Model.invoke_model(decoder, new C.Variable[] { masked });

      network = CC.Combine(new C.VariableVector() { out_caps, decoder_output_training }, "overall_training_network");
      Logging.log_number_of_parameters(network);

      // first component of the loss
      var y_true = categoricalLabel;
      var y_pred = out_caps;
      var digit_loss = CC.Plus(
        CC.ElementTimes(y_true, CC.Square(CC.ElementMax(DC(0), CC.Minus(DC(0.9), y_pred), ""))),
        CC.ElementTimes(DC(0.5), 
        CC.ElementTimes(CC.Minus(DC(1), y_true), CC.Square(CC.ElementMax(DC(0), CC.Minus(y_pred, DC(0.1)), "")))));
      digit_loss = CC.ReduceSum(digit_loss, C.Axis.AllStaticAxes());

      // second component of the loss
      var num_pixels_at_output = Util.np_prod(decoder_output_training.Output.Shape.Dimensions.ToArray());
      var squared_error = CC.SquaredError(decoder_output_training, imageVariable);
      var image_mse = CC.ElementDivide(squared_error, DC(num_pixels_at_output));

      loss_function = CC.Plus(digit_loss, CC.ElementTimes(DC(0.35), image_mse));
      eval_function = CC.ClassificationError(y_pred, y_true);

      learner = CC.AdamLearner(
        new C.ParameterVector(network.Parameters().ToArray()),
        new C.TrainingParameterScheduleDouble(0.001 * batch_size, (uint)batch_size),
        new C.TrainingParameterScheduleDouble(0.9),
        true,
        new C.TrainingParameterScheduleDouble(0.99));

      trainer = CC.CreateTrainer(network, loss_function, eval_function, new C.LearnerVector(new C.Learner[] { learner }));
      evaluator = CC.CreateEvaluator(eval_function);
    }

    C.Constant DC(double value) {
      // DC: device constant
      return C.Constant.Scalar(C.DataType.Float, value, computeDevice);
    }

    readonly int[] input_shape = new int[] { 28, 28, 1 };
    readonly int routings = 3;
    readonly int batch_size = 64;
    readonly C.DeviceDescriptor computeDevice = Util.get_compute_device();

    C.Variable imageVariable;
    C.Variable categoricalLabel;

    C.Function network;
    C.Function loss_function;
    C.Function eval_function;
    C.Learner learner;
    C.Trainer trainer;
    C.Evaluator evaluator;
  }
}
