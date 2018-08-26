using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using DeepLearningWithCNTK;

namespace Ch_08_Neural_Style_Transfer {
  class Program {
    static void Main(string[] args) {
      new Program().style_transfer();
    }

    static readonly float[] offsets = new float[] { 103.939f, 116.779f, 123.68f };
    static readonly string target_image_path = "portrait.png";
    static readonly string style_reference_image_path = "popova.png";
    static readonly int img_height = 400;
    static readonly int img_width = 381;

    static float[] convert_to_channel_first(Mat mat, float[] offsets) {
      var num_pixels = mat.Size().Height * mat.Size().Width;
      float[] result = new float[num_pixels * 3];
      MatOfByte3 mat3 = new MatOfByte3(mat);
      var indexer = mat3.GetIndexer();
      var pos = 0;
      for (int y = 0; y < mat.Height; y++) {
        for (int x = 0; x < mat.Width; x++) {
          var color = indexer[y, x];
          result[pos] = color.Item0 - offsets[0];
          result[pos + num_pixels] = color.Item1 - offsets[1];
          result[pos + 2 * num_pixels] = color.Item2 - offsets[2];
          pos++;
        }
      }
      mat3.Dispose(); mat3 = null;
      return result;
    }

    static float[] preprocess_image(string image_path, int img_height, int img_width) {
      var mat = Cv2.ImRead(image_path);
      var mat2 = new Mat(img_height, img_width, mat.Type());
      Cv2.Resize(mat, mat2, new Size(img_width, img_height));
      var img = convert_to_channel_first(mat2, offsets);
      mat2.Dispose(); mat2 = null;
      mat.Dispose(); mat = null;      
      return img;
    }

    List<CNTK.Variable> traverse_content_and_styles_nodes(CNTK.Function model) {
      var nodes = new List<CNTK.Variable>();
      var node_names = new string[] { "conv5_2", "conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1" };
      foreach (var node_name in node_names) {
        var node = model.FindByName(node_name);
        nodes.Add(node);
      }
      return nodes;
    }

    CNTK.Function create_base_content_and_styles_model(int img_height, int img_width) {
      var combination_image = CNTK.Variable.InputVariable(new int[] { img_width, img_height, 3 }, CNTK.DataType.Float);
      var model = VGG19.get_model(combination_image, computeDevice, freeze: true, include_top: false);
      var nodes = traverse_content_and_styles_nodes(model);
      var content_and_styles_model = CNTK.Function.Combine(nodes, "content_and_styles").Clone(CNTK.ParameterCloningMethod.Freeze);
      return content_and_styles_model;
    }

    float[][] compute_labels(CNTK.Function model, float[] target_image, float[] style_reference_image) {
      var input_shape = model.Arguments[0].Shape.Dimensions.ToArray();
      System.Diagnostics.Debug.Assert(input_shape[0] * input_shape[1] * input_shape[2] == target_image.Length);
      System.Diagnostics.Debug.Assert(target_image.Length == style_reference_image.Length);

#if false
      var cpuDevice = CNTK.DeviceDescriptor.CPUDevice;
      var target_image_nd = new CNTK.NDArrayView(input_shape, target_image, cpuDevice, readOnly: true);
      var style_reference_image_nd = new CNTK.NDArrayView(input_shape, style_reference_image, cpuDevice, readOnly: true);
      var batch_nd = new CNTK.NDArrayView[] { target_image_nd, style_reference_image_nd };
      var batch = CNTK.Value.Create(input_shape, batch_nd, computeDevice, readOnly: true);
#else
      var batch_buffer = new float[2 * target_image.Length];
      Array.Copy(target_image, 0, batch_buffer, 0, target_image.Length);
      Array.Copy(style_reference_image, 0, batch_buffer, target_image.Length, target_image.Length);
      var batch_nd = new CNTK.NDArrayView(new int[] { model.Arguments[0].Shape[0], model.Arguments[0].Shape[1], model.Arguments[0].Shape[2], 1, 2 }, batch_buffer, computeDevice);
      var batch = new CNTK.Value(batch_nd);
#endif
      var inputs = new Dictionary<CNTK.Variable, CNTK.Value>() { { model.Arguments[0], batch } };
      var outputs = new Dictionary<CNTK.Variable, CNTK.Value>();
      foreach (var output in model.Outputs) {
        outputs.Add(output, null);
      }
      model.Evaluate(inputs, outputs, computeDevice);

      float[][] labels = new float[model.Outputs.Count][];
      labels[0] = outputs[model.Outputs[0]].GetDenseData<float>(model.Outputs[0])[0].ToArray();
      for (int i = 1; i < labels.Length; i++) {
        labels[i] = outputs[model.Outputs[i]].GetDenseData<float>(model.Outputs[i])[1].ToArray();
      }

      return labels;
    }

    CNTK.Function content_loss(CNTK.Variable x, CNTK.Function y) {
      var diff_ = CNTK.CNTKLib.Minus(x, y, name: "content_loss_diff_");
      var square_ = CNTK.CNTKLib.Square(diff_, name: "content_loss_square_");
      var sum_ = CNTK.CNTKLib.ReduceSum(square_, CNTK.Axis.AllStaticAxes(), name: "content_loss_sum_");      
      var scaling = CNTK.Constant.Scalar((float)(1.0/x.Shape.TotalSize), computeDevice);
      sum_ = CNTK.CNTKLib.ElementTimes(sum_, scaling, name: "content_loss_");
      return sum_;
    }

    CNTK.Function gram_matrix(CNTK.Variable x) {
      var x_shape = x.Shape.Dimensions.ToArray();
      var features = CNTK.CNTKLib.Reshape(x, new int[] { x_shape[0] * x_shape[1], x_shape[2] }, name: "gram_matrix_reshape_");
      var gram = CNTK.CNTKLib.TransposeTimes(features, features, name: "gram_matrix_transpose_times_");      
      return gram;
    }

    CNTK.Function style_loss(CNTK.Variable style, CNTK.Variable combination) {
      var style_gram = gram_matrix(style);
      var combination_gram = gram_matrix(combination);
      var diff_ = CNTK.CNTKLib.Minus(style_gram, combination_gram, name: "style_loss_diff_");
      var square_ = CNTK.CNTKLib.Square(diff_, name: "style_loss_square_");
      var sum_ = CNTK.CNTKLib.ReduceSum(square_, CNTK.Axis.AllStaticAxes(), name: "style_loss_reduce_sum_");
      var max_ = CNTK.CNTKLib.ReduceMax(style_gram, CNTK.Axis.AllStaticAxes(), name: "style_loss_reduce_max");
      var style_gram_total_size = style_gram.Output.Shape.Dimensions[0] * style_gram.Output.Shape.Dimensions[1];
      var scaling_factor = CNTK.Constant.Scalar((float)style_gram_total_size, computeDevice);
      var result = CNTK.CNTKLib.ElementDivide(sum_, scaling_factor, name: "style_loss_result_");
      result = CNTK.CNTKLib.ElementDivide(result, max_, name: "style_loss_");
      return result;
    }

    public Program() {
      computeDevice = Util.get_compute_device();      
    }

    void create_model(ref CNTK.Function model, ref float[][] labels) {
      var target_image = preprocess_image(target_image_path, img_height, img_width);
      var style_reference_image = preprocess_image(style_reference_image_path, img_height, img_width);
      var base_model = create_base_content_and_styles_model(img_height, img_width);
      labels = compute_labels(base_model, target_image, style_reference_image);

      var dream_weights_init = new CNTK.NDArrayView(new int[] { img_width, img_height, 3 }, target_image, computeDevice);
      var dream_weights = new CNTK.Parameter(dream_weights_init, "the_dream");
      var dummy_features = CNTK.Variable.InputVariable(new int[] { 1 }, CNTK.DataType.Float, "dummy_features");
      var dream_layer = CNTK.CNTKLib.ElementTimes(dream_weights, dummy_features, "the_dream_layler");

      var replacements = new Dictionary<CNTK.Variable, CNTK.Variable>() { { base_model.Arguments[0], dream_layer.Output } };
      model = base_model.Clone(CNTK.ParameterCloningMethod.Freeze, replacements);

      var all_outputs = new List<CNTK.Variable>() { dream_layer };
      all_outputs.AddRange(model.Outputs);
      model = CNTK.Function.Combine(all_outputs, name: "overall_model");
    }

    CNTK.Function create_loss_function(CNTK.Function model, IList<CNTK.Variable> outputs, IList<CNTK.Variable> labels) {
      var loss_function = content_loss(outputs[0], labels[0]);
      for (int i = 1; i < outputs.Count; i++) {
        var sl = style_loss(outputs[i], labels[i]);
        loss_function = CNTK.CNTKLib.Plus(loss_function, sl);
      }
      Util.summary(loss_function);
      Util.log_number_of_parameters(model);
      return loss_function;
    }

    Dictionary<CNTK.Variable, CNTK.Value> create_batch(CNTK.Function model, float[][] labels) {
      var dict_inputs = new Dictionary<CNTK.Variable, CNTK.Value>();
      for (int i = 0; i < model.Arguments.Count; i++) {
        var loss_input_variable = model.Arguments[i];
        if (loss_input_variable.Name == "dummy_features") {
          var dummy_scalar_buffer = new float[] { 1 };
          var dummy_scalar_nd = new CNTK.NDArrayView(new int[] { 1 }, dummy_scalar_buffer, computeDevice, readOnly: true);
          dict_inputs[loss_input_variable] = new CNTK.Value(dummy_scalar_nd);
        }
        else { 
          var cs_index = Int32.Parse(loss_input_variable.Name.Substring("content_and_style_".Length));
          var nd = new CNTK.NDArrayView(loss_input_variable.Shape, labels[cs_index], computeDevice, readOnly: true);
          dict_inputs[loss_input_variable] = new CNTK.Value(nd);
        }
      }
      return dict_inputs;
    }

    void train(CNTK.Function model, float[][] labels) {
      var content_and_style_outputs = traverse_content_and_styles_nodes(model);
      var label_variables = new List<CNTK.Variable>();
      for (int i = 0; i < labels.Length; i++) {
        var shape = content_and_style_outputs[i].Shape;
        var input_variable = CNTK.Variable.InputVariable(shape, CNTK.DataType.Float, "content_and_style_" + i);
        label_variables.Add(input_variable);
      }

      var loss_function = create_loss_function(model, content_and_style_outputs, label_variables);
      var pv = new CNTK.ParameterVector((System.Collections.ICollection)model.Parameters());
      var learner = CNTK.CNTKLib.AdamLearner(pv, new CNTK.TrainingParameterScheduleDouble(10), new CNTK.TrainingParameterScheduleDouble(0.95));
      var trainer = CNTK.CNTKLib.CreateTrainer(model, loss_function, loss_function, new CNTK.LearnerVector() { learner });

      var batch = create_batch(loss_function, labels);
      Console.WriteLine("Training on a "+computeDevice.AsString());
      var startTime = DateTime.Now;
      for (int i = 0; i < 301; i++) {
        trainer.TrainMinibatch(batch, true, computeDevice);
        if (i % 100 == 0) {
          Console.WriteLine($"epoch {i}, loss={trainer.PreviousMinibatchLossAverage():F3}");
        }
      }
      var elapsedTime = DateTime.Now.Subtract(startTime);
      Console.WriteLine($"Done in {elapsedTime.TotalSeconds:F1} seconds");
    }

    byte[] inference(CNTK.Function model, float[][] labels) {
      var batch = create_batch(model, labels);
      var outputs = new Dictionary<CNTK.Variable, CNTK.Value>() { { model.Outputs[0], null } };
      model.Evaluate(batch, outputs, computeDevice);
      var img = outputs[model.Outputs[0]].GetDenseData<float>(model.Outputs[0])[0].ToArray();
      var img_data = Util.convert_from_channels_first(img, offsets);
      return img_data;
    }

    void style_transfer() {
      CNTK.Function model = null;
      float[][] labels = null;
      create_model(ref model, ref labels);
      train(model, labels);
      var img_data = inference(model, labels);
      display_images(img_data);
    }

    void display_images(byte[] img_data) { 
      var target_image_mat = Cv2.ImRead(target_image_path);
      var target_image_mat2 = new Mat(img_height, img_width, target_image_mat.Type());
      Cv2.Resize(target_image_mat, target_image_mat2, new Size(img_width, img_height));
      Cv2.ImShow("Original Image", target_image_mat2);

      var style_reference_image_mat = Cv2.ImRead(style_reference_image_path);
      var style_reference_image_mat2 = new Mat(img_height, img_width, style_reference_image_mat.Type());
      Cv2.Resize(style_reference_image_mat, style_reference_image_mat2, new Size(img_width, img_height));
      Cv2.ImShow("Style", style_reference_image_mat2);

      var combination_mat = new OpenCvSharp.Mat(img_height, img_width, OpenCvSharp.MatType.CV_8UC3, img_data, 3*img_width);
      Cv2.ImShow("Combination Image", combination_mat);
      Cv2.WaitKey(0);
    }

    readonly CNTK.DeviceDescriptor computeDevice;
  }
}
