using System;
using System.ComponentModel;
using System.IO;
using System.Net;
using System.Linq;
using System.Threading;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace DeepLearningWithCNTK {

#if USES_WPF
  static class WPFUtil {
    static public System.Windows.Media.Imaging.BitmapImage BitmapToImageSource(System.Drawing.Bitmap bitmap) {
      using (var memory = new System.IO.MemoryStream()) {
        bitmap.Save(memory, System.Drawing.Imaging.ImageFormat.Bmp);
        memory.Position = 0;
        var bitmapimage = new System.Windows.Media.Imaging.BitmapImage();
        bitmapimage.BeginInit();
        bitmapimage.StreamSource = memory;
        bitmapimage.CacheOption = System.Windows.Media.Imaging.BitmapCacheOption.OnLoad;
        bitmapimage.EndInit();

        return bitmapimage;
      }
    }

    static public System.Drawing.Bitmap createBitmap(float[] src, int gridIndex, int width, int height, bool adjustColorRange) {
      int numChannels = 3;
      var colorScaleFactor = 1.0;
      var numPixels = width * height;
      if (adjustColorRange) {
        var maxValue = src.Skip(gridIndex * numPixels * numChannels).Take(numPixels * numChannels).Max();
        var minValue = src.Skip(gridIndex * numPixels * numChannels).Take(numPixels * numChannels).Min();
        colorScaleFactor = (float)(254.0 / maxValue);
      }

      var bitmap = new System.Drawing.Bitmap(width, height);

      var srcStart = gridIndex * numPixels;
      for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
          var pos = srcStart + row * width + col;
          var b = (int)(colorScaleFactor * src[pos]);
          var g = (numChannels == 1) ? b : (int)(colorScaleFactor * src[pos + numPixels]);
          var r = (numChannels == 1) ? b : (int)(colorScaleFactor * src[pos + 2 * numPixels]);
          bitmap.SetPixel(col, row, System.Drawing.Color.FromArgb(r, g, b));
        }
      }
      return bitmap;
    }
  }

  class PlotWindowBitMap : System.Windows.Window {

    public PlotWindowBitMap(string title, float[] images, int width, int height, int numChannels) {
      var numPixels = width * height;
      var gridLength = (int)Math.Sqrt(images.Length / numPixels);
      var grid = new System.Windows.Controls.Grid();
      for (int row = 0; row < gridLength; row++) {
        grid.RowDefinitions.Add(new System.Windows.Controls.RowDefinition());
        for (int column = 0; column < gridLength; column++) {
          if (row == 0) { grid.ColumnDefinitions.Add(new System.Windows.Controls.ColumnDefinition()); }
          var gridIndex = (row * gridLength + column);
          var bitmap = WPFUtil.createBitmap(images, gridIndex, width, height, adjustColorRange: false);
          var image = new System.Windows.Controls.Image();
          image.Source = WPFUtil.BitmapToImageSource(bitmap);
          image.Stretch = System.Windows.Media.Stretch.Fill;
          grid.Children.Add(image);
          System.Windows.Controls.Grid.SetRow(image, row);
          System.Windows.Controls.Grid.SetColumn(image, column);
        }
      }
      this.Title = title;
      this.Content = grid;
      SizeToContent = System.Windows.SizeToContent.WidthAndHeight;
    }
  }
#endif

  static public class CPPUtil {
    const string CPPUtilDll = "CPPUtil.dll";

    [DllImport(CPPUtilDll)]
    public static extern double version();

    [DllImport(CPPUtilDll)]
    public static extern void compute_image([In, Out]float[] image, [MarshalAs(UnmanagedType.LPWStr)]string pathToVGG16model, int filterIndex);

    [DllImport(CPPUtilDll)]
    public static extern void load_image([MarshalAs(UnmanagedType.LPWStr)]string imagePath, [In, Out]float[] image);

    [DllImport(CPPUtilDll)]
    public static extern void evaluate_vgg16([MarshalAs(UnmanagedType.LPWStr)]string pathToVGG16model, [MarshalAs(UnmanagedType.LPWStr)]string imagePath, [In, Out]float[] predictions, int num_classes);

    [DllImport(CPPUtilDll)]
    public static extern void visualize_heatmap(
      [MarshalAs(UnmanagedType.LPWStr)]string pathToVGG16model, 
      [MarshalAs(UnmanagedType.LPWStr)]string imagePath, 
      [MarshalAs(UnmanagedType.LPWStr)]string layerName, 
      int predictionIndex, 
      [In, Out]float[] imageWithOverlayedHitmap);
  }

  static class VGG19 {
    static readonly string vgg19_filename = "VGG19_ImageNet_Caffe.model";
    static readonly string downlink_url = "https://www.cntk.ai/Models/Caffe_Converted/VGG19_ImageNet_Caffe.model";
    static string fullpath = null;

    static string download_model_if_needed() {
      fullpath = Util.download_model_if_needed(fullpath, vgg19_filename, downlink_url);
      return fullpath;
    }

#if (!NO_CNTK)
    static public CNTK.Function get_model(CNTK.Variable features_node, CNTK.DeviceDescriptor computeDevice, bool include_top=false, bool freeze=false, bool use_finetuning=false) {
      var cloningMethod = freeze ? CNTK.ParameterCloningMethod.Freeze : CNTK.ParameterCloningMethod.Clone;

      // load the original VGG19 model
      download_model_if_needed();
      var model = CNTK.Function.Load(fullpath, computeDevice);
      if ( include_top ) { return model; }

      var pool5_node = model.FindByName("pool5");
      CNTK.Function cloned_model = null;
      if (features_node == null) {
        cloned_model = CNTK.Function.Combine(new CNTK.Variable[] { pool5_node }).Clone(cloningMethod);
        return cloned_model;
      }

      System.Diagnostics.Debug.Assert(use_finetuning == false);
      System.Diagnostics.Debug.Assert(model.Arguments.Count == 1);
      var replacements = new Dictionary<CNTK.Variable, CNTK.Variable>() { {model.Arguments[0], features_node } };
      cloned_model = CNTK.Function.Combine(new CNTK.Variable[] { pool5_node }).Clone(cloningMethod, replacements);
      return cloned_model;
    }
#endif
    /*
    @staticmethod
    def get_model(features_node, include_top=False, use_finetuning=False):

        assert use_finetuning is False
        cloned_model = cntk.ops.combine(pool5_node).clone(cntk.ops.CloneMethod.freeze, substitutions={data_node: features_node})
        return cloned_model

     */
  }

  static class VGG16 {
    static readonly string vgg16_filename = "VGG16_ImageNet_Caffe.model";
    static readonly string downlink_url = "https://www.cntk.ai/Models/Caffe_Converted/VGG16_ImageNet_Caffe.model";
    static string fullpath = null;

    static public string download_model_if_needed() {
      fullpath = Util.download_model_if_needed(fullpath, vgg16_filename, downlink_url);
      return fullpath;
    }

#if (!NO_CNTK)
    static public CNTK.Function get_model(CNTK.Variable features, CNTK.DeviceDescriptor computeDevice, bool allow_block5_finetuning=false) {
      // load the original VGG16 model
      download_model_if_needed();
      var model = CNTK.Function.Load(fullpath, computeDevice);

      // get the last VGG16 layer before the first fully connected layer
      var last_frozen_layer = model.FindByName(allow_block5_finetuning ? "pool4" : "pool5");

      // get the first layer, and the "data" input variable
      var conv1_1_layer = model.FindByName("conv1_1");
      var data = conv1_1_layer.Inputs.First((v) => v.Name == "data");

      // the data should be a 224x224x3 input tensor
      if (!data.Shape.Dimensions.SequenceEqual(new int[] { 224, 224, 3 })) {
        System.Console.WriteLine("There's a problem here. Please email");
        System.Environment.Exit(2);
      }

      // allow different dimensions for input (e.g., 150x150x3)
      var replacements = new Dictionary<CNTK.Variable, CNTK.Variable>() { { data, features } };

      // clone the original VGG16 model up to the pool_node, freeze all weights, and use a custom input tensor
      var frozen_model = CNTK.CNTKLib
        .Combine(new CNTK.VariableVector() { last_frozen_layer.Output }, "frozen_output")
        .Clone(CNTK.ParameterCloningMethod.Freeze, replacements);

      if ( !allow_block5_finetuning ) { return frozen_model; }

      var pool5_layer = model.FindByName("pool5");
      replacements = new Dictionary<CNTK.Variable, CNTK.Variable>() { { last_frozen_layer.Output, frozen_model.Output} };

      var model_with_finetuning = CNTK.CNTKLib
        .Combine(new CNTK.VariableVector() { pool5_layer.Output }, "finetuning_output")
        .Clone(CNTK.ParameterCloningMethod.Clone, replacements);

      return model_with_finetuning;
    }
#endif
  }

  static class Util {
    public static string get_the_path_of_the_elephant_image() {
      var cwd = System.IO.Directory.GetCurrentDirectory();
      var pos = cwd.LastIndexOf("DeepLearning\\");
      var base_path = cwd.Substring(0, pos);
      var image_path = System.IO.Path.Combine(base_path, "DeepLearning", "Ch_05_Class_Activation_Heatmaps", "creative_commons_elephant.jpg");
      return image_path;
    }

    public static string download_model_if_needed(string fullpath, string dstFilename, string downlink_url) {
      System.Net.ServicePointManager.SecurityProtocol = System.Net.SecurityProtocolType.Tls12;
      if (fullpath == null) {
        fullpath = Util.fullpathForDownloadedFile(dstDir: "DownloadedModels", dstFilename: dstFilename);
      }
      if (System.IO.File.Exists(fullpath)) {
        return fullpath;
      }
      var success = FromStackOverflow.FileDownloader.DownloadFile(downlink_url, fullpath, timeoutInMilliSec: 3600000);
      if (!success) {
        System.Console.WriteLine("There was a problem with the download of the caffe model");
        System.Environment.Exit(1);
      }
      return fullpath;
    }


    public static string fullpathForDownloadedFile(string dstDir, string dstFilename) {
      var fullpath = string.Empty;
      var path = System.IO.Directory.GetCurrentDirectory();
      var modelsPath = System.IO.Path.Combine("DeepLearning", dstDir);
      var pos = path.LastIndexOf("DeepLearning");
      if (pos >= 0) {
        fullpath = System.IO.Path.Combine(path.Substring(0, pos), modelsPath);
      }
      if (!System.IO.Directory.Exists(fullpath)) {
        try {
          System.IO.Directory.CreateDirectory(fullpath);
        }
        catch (Exception e) {
          System.Console.WriteLine("Could not create directory " + fullpath + "," + e.Message);
          fullpath = System.IO.Directory.GetCurrentDirectory();
        }
      }
      fullpath = System.IO.Path.Combine(fullpath, dstFilename);
      return fullpath;
    }


    public static void print<T>(IList<T> v, string prefix = null) {
      if (prefix != null) {
        Console.Write(prefix);
      }
      Console.Write("{");
      for (int column = 0; column < v.Count; column++) {
        Console.Write(v[column]);
        if (column != (v.Count - 1)) { Console.Write(", "); }
      }
      Console.WriteLine("}");
    }

    public static void print<T>(T[,,] values) {
      for (int i = 0; i < values.GetLength(0); i++) {
        Console.Write("{\n");
        for (int j = 0; j < values.GetLength(1); j++) {
          Console.Write("\t{");
          for (int k = 0; k < values.GetLength(2); k++) {
            Console.Write(values[i, j, k]);
            if (k != (values.GetLength(2) - 1)) { Console.Write(", "); }
          }
          Console.WriteLine("}");
        }
        Console.WriteLine("}");
      }
      Console.WriteLine();
    }

    public static void save_binary_file(float[][] src, string filepath) {
      Console.WriteLine("Saving " + filepath);
      var buffer = new byte[sizeof(float) * src[0].Length];
      using (var writer = new System.IO.BinaryWriter(System.IO.File.OpenWrite(filepath))) {
        for (int row = 0; row < src.Length; row++) {
          System.Buffer.BlockCopy(src[row], 0, buffer, 0, buffer.Length);
          writer.Write(buffer, 0, buffer.Length);
        }
      }
    }

    public static float[][] load_binary_file(string filepath, int numRows, int numColumns) {
      Console.WriteLine("Loading " + filepath);
      var buffer = new byte[sizeof(float) * numRows * numColumns];
      using (var reader = new System.IO.BinaryReader(System.IO.File.OpenRead(filepath))) {
        reader.Read(buffer, 0, buffer.Length);
      }
      var dst = new float[numRows][];
      for (int row = 0; row < dst.Length; row++) {
        dst[row] = new float[numColumns];
        System.Buffer.BlockCopy(buffer, row * numColumns * sizeof(float), dst[row], 0, numColumns * sizeof(float));
      }
      return dst;
    }

    // https://github.com/dotnet/roslyn/issues/3208#issuecomment-210134781
    public static int SizeOf<T>() where T : struct {
      return Marshal.SizeOf(default(T));
    }

    public static T[] convert_jagged_array_to_single_dimensional_array<T>(T[][] src) where T : struct {
      var numRows = src.Length;
      var numColumns = src[0].Length;
      var numBytesInRow = numColumns * SizeOf<T>();
      var dst = new T[numRows * numColumns];
      var dstOffset = 0;
      for (int row=0; row<numRows; row++) {
        System.Diagnostics.Debug.Assert(src[row].Length == numColumns);
        System.Buffer.BlockCopy(src[row], 0, dst, dstOffset, numBytesInRow);
        dstOffset += numBytesInRow;
      }
      return dst;
    }

    public static float[] load_binary_file(string filepath, int N) {
      Console.WriteLine("Loading " + filepath);
      var buffer = new byte[sizeof(float) * N];
      using (var reader = new System.IO.BinaryReader(System.IO.File.OpenRead(filepath))) {
        reader.Read(buffer, 0, buffer.Length);
      }
      var dst = new float[N];
      System.Buffer.BlockCopy(buffer, 0, dst, 0, buffer.Length);
      return dst;
    }

    public static void swap<T>(T[] array, int n, int k) {
      var temp = array[n];
      array[n] = array[k];
      array[k] = temp;
    }

    public static void shuffle<T>(T[] array) {
      // https://stackoverflow.com/a/110570
      var rng = new Random();
      var n = array.Length;
      while (n > 1) {
        var k = rng.Next(n--);
        swap(array, n, k);
      }
    }

    public static void shuffle<T1, T2>(T1[] array1, T2[] array2) {
      System.Diagnostics.Debug.Assert(array1.Length == array2.Length);
      var rng = new Random();
      var n = array1.Length;
      while (n > 1) {
        var k = rng.Next(n--);
        swap(array1, n, k);
        swap(array2, n, k);
      }
    }

    public static int[] shuffled_indices(int N) {
      var array = new int[N];
      for (int i = 0; i < N; i++) { array[i] = i; }
      shuffle(array);
      return array;
    }

#if (!NO_CNTK)
    static CNTK.NDArrayView[] get_minibatch_data_CPU(CNTK.NDShape shape, float[][] src, int indices_begin, int indices_end) {
      var num_indices = indices_end - indices_begin;
      var result = new CNTK.NDArrayView[num_indices];

      var row_index = 0;
      for (var index = indices_begin; index != indices_end; index++) {
        var dataBuffer = src[index];
        var ndArrayView = new CNTK.NDArrayView(shape, dataBuffer, CNTK.DeviceDescriptor.CPUDevice, true);
        result[row_index++] = ndArrayView;
      }
      return result;
    }

    static float[] get_minibatch_data_CPU(CNTK.NDShape shape, float[] src, int indices_begin, int indices_end) {
      // it would be nice if we avoid the copy here
      var result = new float[indices_end - indices_begin];
      Array.Copy(src, indices_begin, result, 0, result.Length);
      return result;
    }

    static float[] get_minibatch_data_CPU(CNTK.NDShape shape, float[] src, int[] indices, int indices_begin, int indices_end) {
      var num_indices = indices_end - indices_begin;
      var row_length = shape.TotalSize;
      var result = new float[num_indices];
      var row_index = 0;
      for (var index = indices_begin; index != indices_end; index++) {
        result[row_index++] = src[indices[index]];
      }
      return result;
    }

    static CNTK.NDArrayView[] get_minibatch_data_CPU(CNTK.NDShape shape, float[][] src, int[] indices, int indices_begin, int indices_end) {
      var num_indices = indices_end - indices_begin;
      var row_length = shape.TotalSize;
      var result = new CNTK.NDArrayView[num_indices];

      var row_index = 0;
      for (var index = indices_begin; index != indices_end; index++) {
        var dataBuffer = src[indices[index]];
        var ndArrayView = new CNTK.NDArrayView(shape, dataBuffer, CNTK.DeviceDescriptor.CPUDevice, true);
        result[row_index++] = ndArrayView;
      }
      return result;
    }

    static float[][] get_minibatch_data_CPU_sequence(int sequence_length, CNTK.NDShape shape, float[][] src, int[] indices, int indices_begin, int indices_end) {
      System.Diagnostics.Debug.Assert((shape.Dimensions.Count == 0) || ((shape.Dimensions.Count == 1) && (shape.Dimensions[0] == 1)));
      System.Diagnostics.Debug.Assert(src[0].Length == sequence_length);
      var num_indices = indices_end - indices_begin;
      var result = new float[num_indices][];
      var row_index = 0;
      for (var index = indices_begin; index != indices_end; index++) {
        result[row_index] = new float[sequence_length];
        System.Buffer.BlockCopy(src[indices[index]], 0, result[row_index], 0, sequence_length*sizeof(float));
        row_index++;
      }
      return result;
    }

    static float[] get_minibatch_data_CPU_sequence_blob(int sequence_length, CNTK.NDShape shape, float[][] src, int[] indices, int indices_begin, int indices_end) {
      System.Diagnostics.Debug.Assert((shape.Dimensions.Count == 0) || ((shape.Dimensions.Count == 1) && (shape.Dimensions[0] == 1)));
      System.Diagnostics.Debug.Assert(src[0].Length == sequence_length);
      var num_indices = indices_end - indices_begin;
      var result = new float[num_indices * sequence_length];
      var row_index = 0;
      for (var index = indices_begin; index != indices_end; index++) {
        System.Buffer.BlockCopy(src[indices[index]], 0, result, row_index*sequence_length*sizeof(float), sequence_length * sizeof(float));
        row_index++;
      }
      return result;
    }

    static float[] get_minibatch_data_CPU_sequence_blob(int sequence_length, CNTK.NDShape shape, float[][] src, int indices_begin, int indices_end) {
      System.Diagnostics.Debug.Assert((shape.Dimensions.Count == 0) || ((shape.Dimensions.Count == 1) && (shape.Dimensions[0] == 1)));
      System.Diagnostics.Debug.Assert(src[0].Length == sequence_length);
      var num_indices = indices_end - indices_begin;
      var result = new float[num_indices * sequence_length];
      var row_index = 0;
      for (var index = indices_begin; index != indices_end; index++) {
        System.Buffer.BlockCopy(src[index], 0, result, row_index*sequence_length*sizeof(float), sequence_length * sizeof(float));
        row_index++;
      }
      return result;
    }

    static float[][] get_minibatch_data_CPU_sequence(int sequence_length, CNTK.NDShape shape, float[][] src, int indices_begin, int indices_end) {
      System.Diagnostics.Debug.Assert((shape.Dimensions.Count == 0) || ((shape.Dimensions.Count == 1) && (shape.Dimensions[0] == 1)));
      System.Diagnostics.Debug.Assert(src[0].Length == sequence_length);
      var num_indices = indices_end - indices_begin;
      var result = new float[num_indices][];
      var row_index = 0;
      for (var index = indices_begin; index != indices_end; index++) {
        result[row_index] = new float[sequence_length];
        System.Buffer.BlockCopy(src[index], 0, result[row_index], 0, sequence_length * sizeof(float));
        row_index++;
      }
      return result;
    }

    static float[][] get_minibatch_data_CPU_sequence(int sequence_length, CNTK.NDShape shape, float[] src, int[] indices, int indices_begin, int indices_end) {
      System.Diagnostics.Debug.Assert((shape.Dimensions.Count == 0) || ((shape.Dimensions.Count == 1) && (shape.Dimensions[0] == 1)));
      var num_indices = indices_end - indices_begin;
      var result = new float[num_indices][];
      var row_index = 0;
      for (var index = indices_begin; index != indices_end; index++) {
        result[row_index] = new float[sequence_length];
        result[row_index][sequence_length - 1] = src[indices[index]];
        row_index++;
      }
      return result;
    }

    static float[][] get_minibatch_data_CPU_sequence(int sequence_length, CNTK.NDShape shape, float[] src, int indices_begin, int indices_end) {
      System.Diagnostics.Debug.Assert((shape.Dimensions.Count == 0) || ((shape.Dimensions.Count == 1) && (shape.Dimensions[0] == 1)));
      var num_indices = indices_end - indices_begin;
      var result = new float[num_indices][];
      var row_index = 0;
      for (var index = indices_begin; index != indices_end; index++) {
        result[row_index] = new float[sequence_length];
        result[row_index][sequence_length - 1] = src[index];
        row_index++;
      }
      return result;
    }

    public static CNTK.Value get_tensors(CNTK.NDShape shape, float[][] src, int indices_begin, int indices_end, CNTK.DeviceDescriptor device) {
      var cpu_tensors = Util.get_minibatch_data_CPU(shape, src, indices_begin, indices_end);
      var result = CNTK.Value.Create(shape, cpu_tensors, device, true);
      return result;
    }

    static readonly bool USE_CREATE_BATCH_OF_SEQUENCES = false;

    public static CNTK.Value get_tensors_sequence(int sequence_length, CNTK.NDShape shape, float[][] src, int indices_begin, int indices_end, CNTK.DeviceDescriptor device) {
      CNTK.Value result = null;
      if (USE_CREATE_BATCH_OF_SEQUENCES) {
        var cpu_tensors = Util.get_minibatch_data_CPU_sequence(sequence_length, shape, src, indices_begin, indices_end);
        result = CNTK.Value.CreateBatchOfSequences(shape, cpu_tensors, device, true);
      }
      else {
        var cpu_blob = Util.get_minibatch_data_CPU_sequence_blob(sequence_length, shape, src, indices_begin, indices_end);
        var blob_shape = shape.AppendShape(new int[] { sequence_length, indices_end - indices_begin });
        var ndArrayView = new CNTK.NDArrayView(blob_shape, cpu_blob, device);
        result = new CNTK.Value(ndArrayView);
      }
      return result;
    }

    public static CNTK.Value get_tensors(CNTK.NDShape shape, float[][] src, int[] indices, int indices_begin, int indices_end, CNTK.DeviceDescriptor device) {
      var cpu_tensors = Util.get_minibatch_data_CPU(shape, src, indices, indices_begin, indices_end);
      var result = CNTK.Value.Create(shape, cpu_tensors, device, true);
      return result;
    }

    public static CNTK.Value get_tensors_sequence(int sequence_length, CNTK.NDShape shape, float[][] src, int[] indices, int indices_begin, int indices_end, CNTK.DeviceDescriptor device) {
      CNTK.Value result = null;
      if (USE_CREATE_BATCH_OF_SEQUENCES) {
        var cpu_tensors = Util.get_minibatch_data_CPU_sequence(sequence_length, shape, src, indices, indices_begin, indices_end);
        result = CNTK.Value.CreateBatchOfSequences(shape, cpu_tensors, device, true);
      }
      else {
        var cpu_blob = Util.get_minibatch_data_CPU_sequence_blob(sequence_length, shape, src, indices, indices_begin, indices_end);
        var blob_shape = shape.AppendShape(new int[] { sequence_length, indices_end - indices_begin });
        var ndArrayView = new CNTK.NDArrayView(blob_shape, cpu_blob, device);
        result = new CNTK.Value(ndArrayView);
      }
      return result;
    }

    public static CNTK.Value get_tensors(CNTK.NDShape shape, float[] src, int indices_begin, int indices_end, CNTK.DeviceDescriptor device) {
      var cpu_tensors = Util.get_minibatch_data_CPU(shape, src, indices_begin, indices_end);
      var result = CNTK.Value.CreateBatch(shape, cpu_tensors, device, true);
      return result;
    }

    public static CNTK.Value get_tensors_sequence(int sequence_length, CNTK.NDShape shape, float[] src, int indices_begin, int indices_end, CNTK.DeviceDescriptor device) {
      var cpu_tensors = Util.get_minibatch_data_CPU_sequence(sequence_length, shape, src, indices_begin, indices_end);
      var result = CNTK.Value.CreateBatchOfSequences(shape, cpu_tensors, device, true);
      return result;
    }


    public static CNTK.Value get_tensors(CNTK.NDShape shape, float[] src, int[] indices, int indices_begin, int indices_end, CNTK.DeviceDescriptor device) {
      var cpu_tensors = Util.get_minibatch_data_CPU(shape, src, indices, indices_begin, indices_end);
      var result = CNTK.Value.CreateBatch(shape, cpu_tensors, device, true);
      return result;
    }

    public static CNTK.Value get_tensors_sequence(int sequence_length, CNTK.NDShape shape, float[] src, int[] indices, int indices_begin, int indices_end, CNTK.DeviceDescriptor device) {
      var cpu_tensors = Util.get_minibatch_data_CPU_sequence(sequence_length, shape, src, indices, indices_begin, indices_end);
      var result = CNTK.Value.CreateBatchOfSequences(shape, cpu_tensors, device, true);
      return result;
    }


    static public CNTK.DeviceDescriptor get_compute_device() {
      foreach (var gpuDevice in CNTK.DeviceDescriptor.AllDevices()) {
        if (gpuDevice.Type == CNTK.DeviceKind.GPU) { return gpuDevice; }
      }
      return CNTK.DeviceDescriptor.CPUDevice;
    }

    static readonly string equalSigns =  "=============================================================================";
    static readonly string underscores = "_____________________________________________________________________________";
    static void summary(CNTK.Function rootFunction, List<string> entries, ISet<string> visited) {
      if (visited == null) {
        visited = new HashSet<string>();
      }

      if (rootFunction.IsComposite) {
        summary(rootFunction.RootFunction, entries, visited);
        return;
      }

      if ( visited.Contains(rootFunction.Uid)) {
        return;
      }
      visited.Add(rootFunction.Uid);

      var numParameters = 0;
      foreach (var rootInput in rootFunction.Inputs) {
        if (rootInput.IsParameter && !rootInput.IsConstant) {
          numParameters += rootInput.Shape.TotalSize;
        }
        if ((rootInput.Owner == null) || visited.Contains(rootInput.Owner.Uid) ) { continue; }
        summary(rootInput.Owner, entries, visited);
      }
      for (int i = 0; i < rootFunction.Outputs.Count; i++) {
        var line = $"{rootFunction.Name,-29}{rootFunction.Outputs[i].Shape.AsString(),-26}{numParameters}";
        entries.Add(line);
      }
      entries.Add(underscores);
    }

    public static string shape_desc(CNTK.Variable node) {
      var static_shape = node.Shape.AsString();
      var dyn_axes = node.DynamicAxes;
      var dyn_axes_description = (dyn_axes.Count == 1) ? "[#]" : "[*, #]";
      return $"({static_shape}, {dyn_axes_description})";
    }

    public static IList<CNTK.Function> find_function_with_input(CNTK.Function root, CNTK.Function inputFunction) {
      var list = new List<CNTK.Function>();
      root = root.RootFunction;
      inputFunction = inputFunction.RootFunction;
      var root_uid = root.Uid;
      var stack = new Stack<object>();
      stack.Push(root);
      var visited_uids = new HashSet<string>();
      while (stack.Count > 0) {
        var popped = stack.Pop();
        if (popped is CNTK.Variable) {
          var v = (CNTK.Variable)popped;
          if (v.IsOutput) {
            stack.Push(v.Owner);
          }
          continue;
        }
        if (popped is IList<CNTK.Variable>) {
          foreach (var pv in (IList<CNTK.Variable>)popped) {
            stack.Push(pv);
          }
          continue;
        }
        var node = (CNTK.Function)popped;
        if (visited_uids.Contains(node.Uid)) { continue; }
        node = node.RootFunction;
        stack.Push(node.RootFunction.Inputs);

        for (int i = 0; i < node.Inputs.Count; i++) {
          var input = node.Inputs[i];
          if (input.Uid==inputFunction.Output.Uid) {
            list.Add(node);
          }
        }
        visited_uids.Add(node.Uid);
      }
      return list;
    }

    public static List<string> detailed_summary(CNTK.Function root, bool print=true) {
      // This is based on the cntk.logging.graph.plot, but without the actual plotting part
      // Walks through every node of the graph starting at ``root``, creates a network graph, and returns a network description
      var model = new List<string>();
      root = root.RootFunction;
      var root_uid = root.Uid;
      var stack = new Stack<object>();
      stack.Push(root);
      var visited_uids = new HashSet<string>();
      while (stack.Count>0) {
        var popped = stack.Pop();
        if (popped is CNTK.Variable) {
          var v = (CNTK.Variable)popped;
          if (v.IsOutput) {
            stack.Push(v.Owner);
          }
          else if (v.IsInput) {
            model.Add(v.AsString());
          }
          continue;
        }
        if ( popped is IList<CNTK.Variable> ) {
          foreach(var pv in (IList<CNTK.Variable>)popped ) {
            stack.Push(pv);
          }
          continue;
        }
        var node = (CNTK.Function)popped;
        if (visited_uids.Contains(node.Uid)) { continue; }
        node = node.RootFunction;
        stack.Push(node.RootFunction.Inputs);
        var line = new System.Text.StringBuilder($"{node.Name} : {node.OpName} ({node.Uid}) :");
        line.Append("(");
        for (int i=0; i<node.Inputs.Count; i++) {
          var input = node.Inputs[i];
          if (node.IsBlock && input.IsConstant) { continue; }
          line.Append(input.Uid);
          if ( i!=(node.Inputs.Count-1)) {
            line.Append(", ");
          }
        }
        line.Append(") -> ");
        foreach(var v in node.Outputs) {
          model.Add(line.ToString() + "\t" + shape_desc(v));
        }
        visited_uids.Add(node.Uid);
      }
      model.Reverse();
      
      if (print) {
        for (int i=0; i<model.Count; i++) {
          Console.WriteLine(model[i]);
        }
      }
      return model;
    }

    public static void summary(CNTK.Function rootFunction) {
      var entries = new List<string>();
      entries.Add(underscores);
      entries.Add("Layer (type)                 Output Shape              Trainable Parameters #");
      entries.Add(equalSigns);
      summary(rootFunction, entries, null);
      foreach(var v in entries) {
        Console.WriteLine(v);
      }
    }

    static public void log_number_of_parameters(CNTK.Function model) {
      Console.WriteLine("\nModel Summary");
      Console.WriteLine("\tInput = " + model.Arguments[0].Shape.AsString());
      for (int i = 0; i < model.Outputs.Count; i++) {
        Console.WriteLine("\tOutput = " + model.Outputs[i].Shape.AsString());
      }
      Console.WriteLine("");
   
      var numParameters = 0;
      foreach (var x in model.Parameters()) {
        var shape = x.Shape;
        var p = shape.TotalSize;
        Console.WriteLine(string.Format("\tFilter Shape:{0,-30} Param #:{1}", shape.AsString(), p));
        numParameters += p;
      }
      Console.WriteLine(string.Format("\nTotal Number of Parameters: {0:N0}", numParameters));
      Console.WriteLine("---\n");
    }

    static public CNTK.Function Dense(CNTK.Variable input, int outputDim, CNTK.DeviceDescriptor device, string outputName = "") {
      var shape = CNTK.NDShape.CreateNDShape(new int[] { outputDim, CNTK.NDShape.InferredDimension });
      var timesParam = new CNTK.Parameter(shape, CNTK.DataType.Float, CNTK.CNTKLib.GlorotUniformInitializer(CNTK.CNTKLib.DefaultParamInitScale, CNTK.CNTKLib.SentinelValueForInferParamInitRank, CNTK.CNTKLib.SentinelValueForInferParamInitRank, 1), device, "timesParam_"+outputName);
      var timesFunction = CNTK.CNTKLib.Times(timesParam, input, 1 /* output dimension */, 0 /* CNTK should infer the input dimensions */);
      var plusParam = new CNTK.Parameter(CNTK.NDShape.CreateNDShape(new int[] { CNTK.NDShape.InferredDimension }), 0.0f, device, "plusParam_"+outputName);
      var result = CNTK.CNTKLib.Plus(plusParam, timesFunction, outputName);
      return result;
    }

    static public CNTK.Function Embedding(CNTK.Variable x, int shape, CNTK.DeviceDescriptor computeDevice, float[][] weights=null, string name = "") {
      CNTK.Function result;
      // based on the Embedding defined in https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/layers/layers.py      
      if (weights == null) {
        var weight_shape = new int[] { shape, CNTK.NDShape.InferredDimension };
        var E = new CNTK.Parameter(
          weight_shape, 
          CNTK.DataType.Float, 
          CNTK.CNTKLib.GlorotUniformInitializer(), 
          computeDevice, name: "embedding_" + name);
        result = CNTK.CNTKLib.Times(E, x);
      }
      else {
        var weight_shape = new int[] { shape, x.Shape.Dimensions[0] };
        System.Diagnostics.Debug.Assert(shape == weights[0].Length);
        System.Diagnostics.Debug.Assert(weight_shape[1] == weights.Length);
        var w = convert_jagged_array_to_single_dimensional_array(weights);
        var ndArrayView = new CNTK.NDArrayView(weight_shape, w, computeDevice, readOnly: true);
        var E = new CNTK.Constant(ndArrayView, name: "fixed_embedding_"+name);
        result = CNTK.CNTKLib.Times(E, x);
      }
      return result;
    }

    static T[] concat<T>(params T[][] arguments) where T: struct {
      var list = new List<T>();
      for (int i=0; i<arguments.Length; i++) {
        list.AddRange(arguments[i]);
      }
      return list.ToArray();
    }

    static int[] make_ones(int numOnes) {
      var ones = new int[numOnes];
      for (int i=0; i<numOnes; i++) { ones[i] = 1; }
      return ones;
    }

    static T[] pad_to_shape<T>(int[] filter_shape, T value) where T: struct {
      var result = new T[filter_shape.Length];
      for (int i=0; i<result.Length; i++) { result[i] = value; }
      return result;
    }

    static public CNTK.Function ConvolutionTranspose(
      CNTK.Variable x,
      CNTK.DeviceDescriptor computeDevice,
      int[] filter_shape,
      int num_filters,
      Func<CNTK.Variable, CNTK.Function> activation = null,
      bool use_padding = true,
      int[] strides = null,
      bool use_bias = true,
      int[] output_shape = null,
      uint reduction_rank = 1,
      int[] dilation = null,
      uint max_temp_mem_size_in_samples = 0,
      string name = ""
      ) {
      if (strides == null) { strides = new int[] { 1 }; }
      var sharing = pad_to_shape(filter_shape, true);
      var padding = pad_to_shape(filter_shape, use_padding);
      if ( reduction_rank!=1 ) { throw new NotSupportedException("reduction_rank should be 1"); }
      padding = concat(padding, new bool[] { false });
      if ( dilation==null ) {
        dilation = pad_to_shape(filter_shape, 1);
      }
      var output_channels_shape = new int[] { num_filters };
      var kernel_shape = concat(filter_shape, output_channels_shape, new int[] { CNTK.NDShape.InferredDimension });
      var output_full_shape = output_shape;
      if (output_full_shape != null) {
        output_full_shape = concat(output_shape, output_channels_shape);
      }
      var filter_rank = filter_shape.Length;
      var init = CNTK.CNTKLib.GlorotUniformInitializer(CNTK.CNTKLib.DefaultParamInitScale, CNTK.CNTKLib.SentinelValueForInferParamInitRank, CNTK.CNTKLib.SentinelValueForInferParamInitRank, 1);
      var W = new CNTK.Parameter(kernel_shape, CNTK.DataType.Float, init, computeDevice, name = "W");
      var r = CNTK.CNTKLib.ConvolutionTranspose(
        convolutionMap: W,
        operand: x,
        strides: strides,
        sharing: new CNTK.BoolVector(sharing),
        autoPadding: new CNTK.BoolVector(padding),
        outputShape: output_full_shape,
        dilation: dilation,
        reductionRank: reduction_rank,
        maxTempMemSizeInSamples: max_temp_mem_size_in_samples);
      if (use_bias) {
        var b_shape = concat(make_ones(filter_shape.Length), output_channels_shape);
        var b = new CNTK.Parameter(b_shape, 0.0f, computeDevice, "B");
        r = CNTK.CNTKLib.Plus(r, b);
      }
      if (activation != null) {
        r = activation(r);
      }
      return r;
    }

    static public CNTK.Function Convolution1DWithReLU(
      CNTK.Variable input,
      int num_output_channels,
      int filter_shape,
      CNTK.DeviceDescriptor device,
      bool use_padding = false,
      bool use_bias = true,
      int[] strides = null, 
      string outputName = "") {
      var convolution_map_size = new int[] { filter_shape, CNTK.NDShape.InferredDimension, num_output_channels };
      if (strides == null) { strides = new int[] { 1 }; }
      var rtrn = Convolution(convolution_map_size, input, device, use_padding, use_bias, strides, CNTK.CNTKLib.ReLU, outputName);
      return rtrn;
    }

    static public CNTK.Function Convolution2DWithReLU(
      CNTK.Variable input, 
      int num_output_channels, 
      int[] filter_shape, 
      CNTK.DeviceDescriptor device, 
      bool use_padding = false, 
      bool use_bias = true, 
      int[] strides = null,
      string outputName = "") {
      var convolution_map_size = new int[] { filter_shape[0], filter_shape[1], CNTK.NDShape.InferredDimension, num_output_channels };
      if ( strides==null ) { strides = new int[] { 1 }; }
      var rtrn = Convolution(convolution_map_size, input, device, use_padding, use_bias, strides, CNTK.CNTKLib.ReLU, outputName);
      return rtrn;
    }

    static public CNTK.Function Convolution2DWithSigmoid(
      CNTK.Variable input,
      int num_output_channels,
      int[] filter_shape,
      CNTK.DeviceDescriptor device,
      bool use_padding = false,
      bool use_bias = true,
      int[] strides = null,
      string outputName = "") {
      var convolution_map_size = new int[] { filter_shape[0], filter_shape[1], CNTK.NDShape.InferredDimension, num_output_channels };
      if (strides == null) { strides = new int[] { 1 }; }
      var rtrn = Convolution(convolution_map_size, input, device, use_padding, use_bias, strides, CNTK.CNTKLib.Sigmoid, outputName);
      return rtrn;
    }

    static CNTK.Function Convolution(
      int[] convolution_map_size, 
      CNTK.Variable input, 
      CNTK.DeviceDescriptor device, 
      bool use_padding, 
      bool use_bias, 
      int[] strides,
      Func<CNTK.Variable, string, CNTK.Function> activation=null,
      string outputName="") { 
      var W = new CNTK.Parameter(
        CNTK.NDShape.CreateNDShape(convolution_map_size),
        CNTK.DataType.Float,
        CNTK.CNTKLib.GlorotUniformInitializer(CNTK.CNTKLib.DefaultParamInitScale, CNTK.CNTKLib.SentinelValueForInferParamInitRank, CNTK.CNTKLib.SentinelValueForInferParamInitRank, 1),
        device, outputName + "_W");
      
      var result = CNTK.CNTKLib.Convolution(W, input, strides, new CNTK.BoolVector(new bool[] { true }) /* sharing */, new CNTK.BoolVector(new bool[] { use_padding }));

      if ( use_bias ) {
        var num_output_channels = convolution_map_size[convolution_map_size.Length - 1];
        var b_shape = concat(make_ones(convolution_map_size.Length - 2), new int[] { num_output_channels });
        var b = new CNTK.Parameter(b_shape, 0.0f, device, outputName + "_b");
        result = CNTK.CNTKLib.Plus(result, b);
      }

      if (activation != null) {
        result = activation(result, outputName);
      }
      return result;
    }

    static public CNTK.Function BinaryAccuracy(CNTK.Variable prediction, CNTK.Variable labels) {
      var round_predictions = CNTK.CNTKLib.Round(prediction);
      var equal_elements = CNTK.CNTKLib.Equal(round_predictions, labels);
      var result = CNTK.CNTKLib.ReduceMean(equal_elements, CNTK.Axis.AllStaticAxes());
      return result;
    }

    static public CNTK.Function MeanSquaredError(CNTK.Variable prediction, CNTK.Variable labels) {
      var squared_errors = CNTK.CNTKLib.Square(CNTK.CNTKLib.Minus(prediction, labels));
      var result = CNTK.CNTKLib.ReduceMean(squared_errors, new CNTK.Axis(0)); // TODO -- allStaticAxes?
      return result;
    }

    static public CNTK.Function MeanAbsoluteError(CNTK.Variable prediction, CNTK.Variable labels) {
      var absolute_errors = CNTK.CNTKLib.Abs(CNTK.CNTKLib.Minus(prediction, labels));
      var result = CNTK.CNTKLib.ReduceMean(absolute_errors, new CNTK.Axis(0)); // TODO -- allStaticAxes? 
      return result;
    }
#endif
    }

#if (!NO_CNTK)

  class ReduceLROnPlateau {
    readonly CNTK.Learner learner;
    double lr = 0;
    double best_metric = 1e-5;
    int slot_since_last_update = 0;

    public ReduceLROnPlateau(CNTK.Learner learner, double lr) {
      this.learner = learner;
      this.lr = lr;
    }

    public bool update(double current_metric) {
      bool should_stop = false;
      if (current_metric < best_metric) {
        best_metric = current_metric;
        slot_since_last_update = 0;
        return should_stop;
      }
      slot_since_last_update++;
      if (slot_since_last_update > 10) {
        lr *= 0.75;
        learner.ResetLearningRate(new CNTK.TrainingParameterScheduleDouble(lr));
        Console.WriteLine($"Learning rate set to {lr}");
        slot_since_last_update = 0;
        should_stop = (lr < 1e-6);
      }
      return should_stop;
    }
  }

  abstract class TrainingEngine {
    public enum LossFunctionType { BinaryCrossEntropy, MSE, CrossEntropyWithSoftmax, CrossEntropyWithSoftmaxWithOneHotEncodedLabel, Custom };
    public enum AccuracyFunctionType { BinaryAccuracy, SameAsLoss};
    public enum MetricType { Loss, Accuracy }

    public LossFunctionType lossFunctionType = LossFunctionType.BinaryCrossEntropy;
    public AccuracyFunctionType accuracyFunctionType = AccuracyFunctionType.BinaryAccuracy;
    public MetricType metricType = MetricType.Accuracy;
    public double lr = 0.1;    

    public int num_epochs { get; set; }
    public int batch_size { get; set; }
    public int sequence_length { get; set; } = 1;
    public readonly CNTK.DeviceDescriptor computeDevice;
    public CNTK.Function model=null;
    protected CNTK.Variable x=null;
    protected CNTK.Variable y=null;
    protected CNTK.Trainer trainer;
    protected CNTK.Evaluator evaluator;
    protected float[][] x_train;
    protected float[] y_train;
    protected float[][] x_val;
    protected float[] y_val;
    protected ReduceLROnPlateau scheduler;

    public TrainingEngine() {
      computeDevice = Util.get_compute_device();
    }

    public void setData(float[][] x_train, float[] y_train, float[][] x_val, float[] y_val) {
      this.x_train = x_train;
      this.y_train = y_train;
      this.x_val = x_val;
      this.y_val = y_val;
    }

    protected abstract void createVariables();
    protected abstract void createModel();
    protected virtual CNTK.Function custom_loss_function() { return null; }

    void assertSequenceLength() {
      if ( sequence_length==1 ) { return; }
      if (x.Shape.Dimensions.Count>=2) { throw new NotImplementedException(); }
      if ((x.Shape.Dimensions.Count==1) && (x.Shape.Dimensions[0]!=1)) { throw new NotImplementedException(); }
    }

    public void train(double threshold=0) {
      createVariables();
      createModel();
      assertSequenceLength();

      CNTK.Function loss_function = null;
      switch (lossFunctionType ) {
        case LossFunctionType.BinaryCrossEntropy: loss_function = CNTK.CNTKLib.BinaryCrossEntropy(model, y); break;
        case LossFunctionType.MSE: loss_function = CNTK.CNTKLib.SquaredError(model, y); break;
        case LossFunctionType.CrossEntropyWithSoftmax: loss_function = CNTK.CNTKLib.CrossEntropyWithSoftmax(model, y); break;
        case LossFunctionType.Custom: loss_function = custom_loss_function(); break;
      }

      CNTK.Function accuracy_function = null;
      switch (accuracyFunctionType) {
        case AccuracyFunctionType.SameAsLoss: accuracy_function = loss_function; break;
        case AccuracyFunctionType.BinaryAccuracy: accuracy_function = Util.BinaryAccuracy(model, y); break;
      }

      var parameterVector = new CNTK.ParameterVector((System.Collections.ICollection)model.Parameters());
      var learner = CNTK.CNTKLib.AdamLearner(parameterVector,
        new CNTK.TrainingParameterScheduleDouble(lr /*, (uint)batch_size*/),
        new CNTK.TrainingParameterScheduleDouble(0.9 /*, (uint)batch_size*/),
        unitGain: false);
      trainer = CNTK.CNTKLib.CreateTrainer(model, loss_function, accuracy_function, new CNTK.LearnerVector() { learner });
      scheduler = new ReduceLROnPlateau(learner, lr);
      if ( x_val!= null ) {
        evaluator = CNTK.CNTKLib.CreateEvaluator(accuracy_function);
      }

      Util.log_number_of_parameters(model);

      Console.WriteLine("\nCompute Device: " + computeDevice.AsString());
      for (int epoch = 0; epoch < num_epochs; epoch++) {
        var epoch_start_time = DateTime.Now;
        var epoch_training_metric = training_phase();
        var epoch_validation_accuracy = evaluation_phase();
        var elapsedTime = DateTime.Now.Subtract(epoch_start_time);
        if (metricType == MetricType.Accuracy) {
          Console.WriteLine($"Epoch {epoch + 1:D2}/{num_epochs}, Elapsed time: {elapsedTime.TotalSeconds:F3} seconds. " +
            $"Training Accuracy: {epoch_training_metric:F3}. Validation Accuracy: {epoch_validation_accuracy:F3}.");
        }
        else {
          Console.WriteLine($"Epoch {epoch + 1:D2}/{num_epochs}, Elapsed time: {elapsedTime.TotalSeconds:F3} seconds, Training Loss: {epoch_training_metric:F3}");
        }
        if (scheduler.update(epoch_training_metric)) {
          break;
        }
        if ( (threshold!=0) && (epoch_training_metric<threshold) ) {
          break;
        }
      }
    }

    public IList<IList<float>> evaluate(float[][] x_data, CNTK.Function f=null) {
      if ( f==null ) { f = model; }
      var x_minibatch = (sequence_length == 1) ?
        Util.get_tensors(x.Shape, x_data, 0, x_data.Length, computeDevice) :
        Util.get_tensors_sequence(sequence_length, x.Shape, x_data, 0, x_data.Length, computeDevice);
      var inputs = new Dictionary<CNTK.Variable, CNTK.Value>() { {x, x_minibatch } };
      var outputs = new Dictionary<CNTK.Variable, CNTK.Value>() { { f.Output, null } };
      f.Evaluate(inputs, outputs, computeDevice);
      var result = outputs[f.Output];
      var outputData = result.GetDenseData<float>(f.Output);
      return outputData;
    }

    double evaluation_phase() {
      if ( evaluator==null ) { return 0.0; }
      var pos = 0;
      var metric = 0.0;
      if (x_val.Length == 0) { return 0.0; }
      while (pos < x_val.Length) {
        var pos_end = Math.Min(pos + batch_size, x_val.Length);
        var x_minibatch = (sequence_length == 1) ?
          Util.get_tensors(x.Shape, x_val, pos, pos_end, computeDevice) :
          Util.get_tensors_sequence(sequence_length, x.Shape, x_val, pos, pos_end, computeDevice);
        var y_minibatch = Util.get_tensors(y.Shape, y_val, pos, pos_end, computeDevice);
        var feed_dictionary = new CNTK.UnorderedMapVariableValuePtr() { { x, x_minibatch }, { y, y_minibatch } };
        var minibatch_metric = evaluator.TestMinibatch(feed_dictionary, computeDevice);
        metric += minibatch_metric * (pos_end - pos);
        pos = pos_end;
        x_minibatch.Erase();
        y_minibatch.Erase();
      }
      metric /= x_val.Length;
      return metric;
    }

    double training_phase() {
      var train_indices = Util.shuffled_indices(x_train.Length);
      var pos = 0;
      var metric = 0.0;
      while (pos < train_indices.Length) {
        var pos_end = Math.Min(pos + batch_size, train_indices.Length);
        var x_minibatch = (sequence_length==1) ? 
          Util.get_tensors(x.Shape, x_train, train_indices, pos, pos_end, computeDevice) :
          Util.get_tensors_sequence(sequence_length, x.Shape, x_train, train_indices, pos, pos_end, computeDevice);
        var y_minibatch = Util.get_tensors(y.Shape, y_train, train_indices, pos, pos_end, computeDevice);

        var feed_dictionary = new Dictionary<CNTK.Variable, CNTK.Value> { { x, x_minibatch }, { y, y_minibatch } };
        bool isSweepEndInArguments = (pos_end == train_indices.Length);
        trainer.TrainMinibatch(feed_dictionary, isSweepEndInArguments, computeDevice);
        var minibatch_metric =  (metricType==MetricType.Loss) ? trainer.PreviousMinibatchLossAverage() : trainer.PreviousMinibatchEvaluationAverage();
        metric += minibatch_metric * (pos_end - pos);
        pos = pos_end;
        x_minibatch.Erase();
        y_minibatch.Erase();
      }
      metric /= train_indices.Length;
      return metric;
    }
  }
#endif
}

#if (!NO_CNTK)
namespace CNTK.CSTrainingExamples {
  public class LSTMSequenceClassifier {
    // Original Code: https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/LSTMSequenceClassifier.cs
    static Function Stabilize<ElementType>(Variable x, DeviceDescriptor device) {
      bool isFloatType = typeof(ElementType).Equals(typeof(float));
      Constant f, fInv;
      if (isFloatType) {
        f = Constant.Scalar(4.0f, device);
        fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
      }
      else {
        f = Constant.Scalar(4.0, device);
        fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
      }

      var beta = CNTKLib.ElementTimes(
          fInv,
          CNTKLib.Log(
              Constant.Scalar(f.DataType, 1.0) +
              CNTKLib.Exp(CNTKLib.ElementTimes(f, new Parameter(new NDShape(), f.DataType, 0.99537863 /* 1/f*ln (e^f-1) */, device)))));
      return CNTKLib.ElementTimes(beta, x);
    }

    static Tuple<Function, Function> LSTMPCellWithSelfStabilization<ElementType>(Variable input, Variable prevOutput, Variable prevCellState, DeviceDescriptor device) {
      int outputDim = prevOutput.Shape[0];
      int cellDim = prevCellState.Shape[0];

      bool isFloatType = typeof(ElementType).Equals(typeof(float));
      DataType dataType = isFloatType ? DataType.Float : DataType.Double;

      Func<int, Parameter> createBiasParam;
      if (isFloatType)
        createBiasParam = (dim) => new Parameter(new int[] { dim }, 0.01f, device, "");
      else
        createBiasParam = (dim) => new Parameter(new int[] { dim }, 0.01, device, "");

      uint seed2 = 1;
      Func<int, Parameter> createProjectionParam = (oDim) => new Parameter(new int[] { oDim, NDShape.InferredDimension },
              dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device);

      Func<int, Parameter> createDiagWeightParam = (dim) =>
          new Parameter(new int[] { dim }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device);

      Function stabilizedPrevOutput = Stabilize<ElementType>(prevOutput, device);
      Function stabilizedPrevCellState = Stabilize<ElementType>(prevCellState, device);

      Func<Variable> projectInput = () =>
          createBiasParam(cellDim) + (createProjectionParam(cellDim) * input);

      // Input gate
      Function it =
          CNTKLib.Sigmoid(
              (Variable)(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) +
              CNTKLib.ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
      Function bit = CNTKLib.ElementTimes(
          it,
          CNTKLib.Tanh(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)));

      // Forget-me-not gate
      Function ft = CNTKLib.Sigmoid(
          (Variable)(
                  projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) +
                  CNTKLib.ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
      Function bft = CNTKLib.ElementTimes(ft, prevCellState);

      Function ct = (Variable)bft + bit;

      // Output gate
      Function ot = CNTKLib.Sigmoid(
          (Variable)(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) +
          CNTKLib.ElementTimes(createDiagWeightParam(cellDim), Stabilize<ElementType>(ct, device)));
      Function ht = CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct));

      Function c = ct;
      Function h = (outputDim != cellDim) ? (createProjectionParam(outputDim) * Stabilize<ElementType>(ht, device)) : ht;

      return new Tuple<Function, Function>(h, c);
    }

    static Tuple<Function, Function> LSTMPComponentWithSelfStabilization<ElementType>(Variable input,
        NDShape outputShape, NDShape cellShape,
        Func<Variable, Function> recurrenceHookH,
        Func<Variable, Function> recurrenceHookC,
        DeviceDescriptor device) {
      var dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes);
      var dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes);

      var LSTMCell = LSTMPCellWithSelfStabilization<ElementType>(input, dh, dc, device);
      var actualDh = recurrenceHookH(LSTMCell.Item1);
      var actualDc = recurrenceHookC(LSTMCell.Item2);

      // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
      (LSTMCell.Item1).ReplacePlaceholders(new Dictionary<Variable, Variable> { { dh, actualDh }, { dc, actualDc } });

      return new Tuple<Function, Function>(LSTMCell.Item1, LSTMCell.Item2);
    }

    static public Function Embedding(Variable input, int embeddingDim, DeviceDescriptor device) {
      System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);
      int inputDim = input.Shape[0];
      var embeddingParameters = new Parameter(new int[] { embeddingDim, inputDim }, DataType.Float, CNTKLib.GlorotUniformInitializer(), device);
      return CNTKLib.Times(embeddingParameters, input);
    }

    static public Function LSTM(Variable input, int LSTMDim, int cellDim, DeviceDescriptor device, string outputName) {
      Func<Variable, Function> pastValueRecurrenceHook = (x) => CNTKLib.PastValue(x);
      var LSTMFunction = LSTMPComponentWithSelfStabilization<float>(
          input,
          new int[] { LSTMDim },
          new int[] { cellDim },
          pastValueRecurrenceHook,
          pastValueRecurrenceHook,
          device).Item1;
      var rtrn = CNTK.CNTKLib.SequenceLast(LSTMFunction);
      return rtrn;
    }
  }
}
#endif

namespace FromKeras {
  static class Preprocessing {
    // https://github.com/keras-team/keras/blob/master/keras/preprocessing/sequence.py
    static public float[][] pad_sequences<T>(List<List<T>> sequences, int maxlen) {
      var padded_sequences = new float[sequences.Count][];
      for (int i = 0; i < sequences.Count; i++) {
        padded_sequences[i] = new float[maxlen];
        var src = sequences[i];
        var dst = padded_sequences[i];
        var offset = src.Count - dst.Length;
        for (int dst_index = Math.Max(0, -offset); dst_index < dst.Length; dst_index++) {
          var src_index = dst_index + offset;
          dst[dst_index] = Convert.ToSingle(src[src_index]);
        }
      }
      return padded_sequences;
    }
  }

  class Tokenizer {
    public static string python_string_printable() {
      // https://github.com/python/cpython/blob/master/Lib/string.py
      var whitespace = " \t\n\r\v\f";
      var ascii_lowercase = "abcdefghijklmnopqrstuvwxyz";
      var ascii_uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
      var ascii_letters = ascii_lowercase + ascii_uppercase;
      var digits = "0123456789";
      var hexdigits = digits + "abcdef" + "ABCDEF";
      var punctuation = "\"!#$%&()*+,-./:;<=>?@[\\]^_`{|}~";
      var printable = digits + ascii_letters + punctuation + whitespace;
      return printable;
    }

    readonly int num_words;
    readonly Dictionary<string, int> _word_indices;

    public Dictionary<string, int> word_index { get { return _word_indices; } }

    public Tokenizer(int num_words) {
      this.num_words = num_words;
      _word_indices = new Dictionary<string, int>();
    }

    public List<List<int>> texts_to_sequences(string[] samples) {
      var results = new List<List<int>>();
      for (int i = 0; i < samples.Length; i++) {
        var sequence = new List<int>();
        foreach (var word in text_to_word_sequence(samples[i])) {
          int word_index;
          _word_indices.TryGetValue(word, out word_index);
          sequence.Add(word_index);
        }
        results.Add(sequence);
      }
      return results;
    }

    // See https://github.com/keras-team/keras/blob/master/keras/preprocessing/text.py
    static readonly char[] filters = " !\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n".ToArray();

    string[] text_to_word_sequence(string text) {
      return text.ToLower().Split(filters).Where(v => v.Length > 0).ToArray();
    }

    public void fit_on_texts(string[] samples) {
      _word_indices.Clear();
      for (int i = 0; i < samples.Length; i++) {
        foreach (var word in text_to_word_sequence(samples[i])) {
          var num_occurences = _word_indices.ContainsKey(word) ? _word_indices[word] : 0;
          _word_indices[word] = num_occurences + 1;
        }
      }

      if (_word_indices.Count >= this.num_words) {
        var words_in_descending_order_of_occurences = _word_indices.Keys.OrderByDescending(v => _word_indices[v]).ToArray();
        for (int i = this.num_words; i < words_in_descending_order_of_occurences.Length; i++) {
          _word_indices.Remove(words_in_descending_order_of_occurences[i]);
        }
      }

      var keys = _word_indices.Keys.ToArray();
      for (int i = 0; i < keys.Length; i++) {
        _word_indices[keys[i]] = i + 1;
      }
    }
  }

}

namespace FromStackOverflow {
  // https://stackoverflow.com/a/35936119
  class FileDownloader {
    private readonly string _url;
    private readonly string _fullPathWhereToSave;
    private bool _result = false;
    private readonly SemaphoreSlim _semaphore = new SemaphoreSlim(0);

    public FileDownloader(string url, string fullPathWhereToSave) {
      if (string.IsNullOrEmpty(url)) throw new ArgumentNullException("url");
      if (string.IsNullOrEmpty(fullPathWhereToSave)) throw new ArgumentNullException("fullPathWhereToSave");

      this._url = url;
      this._fullPathWhereToSave = fullPathWhereToSave;
    }

    public bool StartDownload(int timeout) {
      try {
        System.IO.Directory.CreateDirectory(Path.GetDirectoryName(_fullPathWhereToSave));

        if (File.Exists(_fullPathWhereToSave)) {
          File.Delete(_fullPathWhereToSave);
        }
        using (WebClient client = new WebClient()) {
          var ur = new Uri(_url);
          // client.Credentials = new NetworkCredential("username", "password");
          client.DownloadProgressChanged += WebClientDownloadProgressChanged;
          client.DownloadFileCompleted += WebClientDownloadCompleted;
          Console.WriteLine("Downloading " + ur);
          client.DownloadFileAsync(ur, _fullPathWhereToSave);
          _semaphore.Wait(timeout);
          return _result && File.Exists(_fullPathWhereToSave);
        }
      }
      catch (Exception e) {
        Console.WriteLine("Was not able to download file!");
        Console.Write(e);
        return false;
      }
      finally {
        this._semaphore.Dispose();
      }
    }

    private void WebClientDownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e) {
      Console.Write("\r     -->    {0}%.", e.ProgressPercentage);
    }

    private void WebClientDownloadCompleted(object sender, AsyncCompletedEventArgs args) {
      _result = !args.Cancelled;
      if (!_result) {
        Console.Write(args.Error.ToString());
      }
      Console.WriteLine(" Download finished!");
      _semaphore.Release();
    }

    public static bool DownloadFile(string url, string fullPathWhereToSave, int timeoutInMilliSec) {
      return new FileDownloader(url, fullPathWhereToSave).StartDownload(timeoutInMilliSec);
    }
  }
}