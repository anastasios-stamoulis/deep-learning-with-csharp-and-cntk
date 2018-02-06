using System;
using System.ComponentModel;
using System.IO;
using System.Net;
using System.Linq;
using System.Threading;
using System.Runtime.InteropServices;

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

  static class VGG16 {
    static readonly string vgg16_filename = "VGG16_ImageNet_Caffe.model";
    static readonly string downlink_url = "https://www.cntk.ai/Models/Caffe_Converted/VGG16_ImageNet_Caffe.model";
    static string fullpath = null;

    static void initFullpath() {
      if (fullpath != null) { return; }
      var path = System.IO.Directory.GetCurrentDirectory();
      var modelsPath = System.IO.Path.Combine("DeepLearning", "DownloadedModels");
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
      fullpath = System.IO.Path.Combine(fullpath, VGG16.vgg16_filename);
    }

    static public string download_model_if_needed() {
      initFullpath();
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
      var replacements = new System.Collections.Generic.Dictionary<CNTK.Variable, CNTK.Variable>() { { data, features } };

      // clone the original VGG16 model up to the pool_node, freeze all weights, and use a custom input tensor
      var frozen_model = CNTK.CNTKLib
        .Combine(new CNTK.VariableVector() { last_frozen_layer.Output }, "frozen_output")
        .Clone(CNTK.ParameterCloningMethod.Freeze, replacements);

      if ( !allow_block5_finetuning ) { return frozen_model; }

      var pool5_layer = model.FindByName("pool5");
      replacements = new System.Collections.Generic.Dictionary<CNTK.Variable, CNTK.Variable>() { { last_frozen_layer.Output, frozen_model.Output} };

      var model_with_finetuning = CNTK.CNTKLib
        .Combine(new CNTK.VariableVector() { pool5_layer.Output }, "finetuning_output")
        .Clone(CNTK.ParameterCloningMethod.Clone, replacements);

      return model_with_finetuning;
    }
#endif
  }

#if (!NO_CNTK)
  static class Util {

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

    public static CNTK.Value get_tensors(CNTK.NDShape shape, float[][] src, int indices_begin, int indices_end, CNTK.DeviceDescriptor device) {
      var cpu_tensors = Util.get_minibatch_data_CPU(shape, src, indices_begin, indices_end);
      var result = CNTK.Value.Create(shape, cpu_tensors, device, true);
      return result;
    }

    public static CNTK.Value get_tensors(CNTK.NDShape shape, float[][] src, int[] indices, int indices_begin, int indices_end, CNTK.DeviceDescriptor device) {
      var cpu_tensors = Util.get_minibatch_data_CPU(shape, src, indices, indices_begin, indices_end);
      var result = CNTK.Value.Create(shape, cpu_tensors, device, true);
      return result;
    }

    public static CNTK.Value get_tensors(CNTK.NDShape shape, float[] src, int indices_begin, int indices_end, CNTK.DeviceDescriptor device) {
      var cpu_tensors = Util.get_minibatch_data_CPU(shape, src, indices_begin, indices_end);
      var result = CNTK.Value.CreateBatch(shape, cpu_tensors, device, true);
      return result;
    }

    public static CNTK.Value get_tensors(CNTK.NDShape shape, float[] src, int[] indices, int indices_begin, int indices_end, CNTK.DeviceDescriptor device) {
      var cpu_tensors = Util.get_minibatch_data_CPU(shape, src, indices, indices_begin, indices_end);
      var result = CNTK.Value.CreateBatch(shape, cpu_tensors, device, true);
      return result;
    }

    public static void shuffle(int[] array) {
      // https://stackoverflow.com/a/110570
      var rng = new Random();
      int n = array.Length;
      while (n > 1) {
        int k = rng.Next(n--);
        var temp = array[n];
        array[n] = array[k];
        array[k] = temp;
      }
    }

    public static int[] shuffled_indices(int N) {
      var array = new int[N];
      for (int i = 0; i < N; i++) { array[i] = i; }
      shuffle(array);
      return array;
    }

    static public CNTK.DeviceDescriptor get_compute_device() {
      foreach (var gpuDevice in CNTK.DeviceDescriptor.AllDevices()) {
        if (gpuDevice.Type == CNTK.DeviceKind.GPU) { return gpuDevice; }
      }
      return CNTK.DeviceDescriptor.CPUDevice;
    }

    static readonly string equalSigns =  "=============================================================================";
    static readonly string underscores = "_____________________________________________________________________________";
    static void summary(CNTK.Function rootFunction, System.Collections.Generic.List<string> entries, System.Collections.Generic.ISet<CNTK.Function> visited) {
      if (visited == null) {
        visited = new System.Collections.Generic.HashSet<CNTK.Function>();
      }
      if (rootFunction.IsComposite) {
        summary(rootFunction.RootFunction, entries, visited);
        return;
      }
      var numParameters = 0;
      foreach (var rootInput in rootFunction.Inputs) {
        if (rootInput.IsParameter) {
          numParameters += rootInput.Shape.TotalSize;
        }
        if (visited.Contains(rootInput.Owner) || (rootInput.Owner==null)) { continue; }
        summary(rootInput.Owner, entries, visited);
      }
      var line = $"{rootFunction.Name,-29}{rootFunction.Output.Shape.AsString(),-26}{numParameters}";
      entries.Add(line);
      entries.Add(underscores);
    }

    public static void summary(CNTK.Function rootFunction) {
      var entries = new System.Collections.Generic.List<string>();

      

      entries.Add(underscores);
      entries.Add("Layer (type)                 Output Shape              Trainable Parameters #");
      entries.Add(equalSigns);

      summary(rootFunction, entries, null);
      
      foreach(var v in entries) {
        Console.WriteLine(v);
      }
    }

    public static void PredorderTraverse(CNTK.Function rootFunction, int log_level=0, System.Collections.Generic.ISet<CNTK.Function> visited=null) {
      Console.WriteLine($"{rootFunction.Name} -> {rootFunction.Output.Shape.AsString()}");
      if (visited == null) {
        visited = new System.Collections.Generic.HashSet<CNTK.Function>();
      }
      visited.Add(rootFunction);

      if (rootFunction.IsComposite) {
        PredorderTraverse(rootFunction.RootFunction, log_level, visited);
        return;
      }

      foreach (var rootInput in rootFunction.Inputs) {
        if (!rootInput.IsOutput) {
          Console.WriteLine($"\t{rootInput.Name}\t{rootInput.Shape.AsString()}");
          if (log_level == 1) {
            Console.WriteLine("\t\t constant:" + rootInput.IsConstant);
            Console.WriteLine("\t\t input:" + rootInput.IsInput);
            Console.WriteLine("\t\t parameter:" + rootInput.IsInput);
            Console.WriteLine("\t\t placeholder:" + rootInput.IsPlaceholder);
          }
          continue;
        }
        if (visited.Contains(rootInput.Owner)) { continue; }
        PredorderTraverse(rootInput.Owner, log_level, visited);
      }
    }

    static public void log_number_of_parameters(CNTK.Function model) {
      Console.WriteLine("\nModel Summary");
      Console.WriteLine("\tInput = " + model.Arguments[0].Shape.AsString());
      Console.WriteLine("\tOutput = " + model.Output.Shape.AsString());
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
      var timesParam = new CNTK.Parameter(shape, CNTK.DataType.Float, CNTK.CNTKLib.GlorotUniformInitializer(CNTK.CNTKLib.DefaultParamInitScale, CNTK.CNTKLib.SentinelValueForInferParamInitRank, CNTK.CNTKLib.SentinelValueForInferParamInitRank, 1), device, "timesParam");
      var timesFunction = CNTK.CNTKLib.Times(timesParam, input, 1 /* output dimension */, 0 /* CNTK should infer the input dimensions */);
      var plusParam = new CNTK.Parameter(CNTK.NDShape.CreateNDShape(new int[] { CNTK.NDShape.InferredDimension }), 0.0f, device, "plusParam");
      var result = CNTK.CNTKLib.Plus(plusParam, timesFunction, outputName);
      return result;
    }

    static public CNTK.Function Convolution2DWithReLU(CNTK.Variable input, int num_output_channels, int[] filter_shape, CNTK.DeviceDescriptor device, bool use_padding=false, bool use_bias=true, string outputName = "") {
      var convolution_map_size = new int[] { filter_shape[0], filter_shape[1], CNTK.NDShape.InferredDimension, num_output_channels };

      var W = new CNTK.Parameter(
        CNTK.NDShape.CreateNDShape(convolution_map_size),
        CNTK.DataType.Float,
        CNTK.CNTKLib.GlorotUniformInitializer(CNTK.CNTKLib.DefaultParamInitScale, CNTK.CNTKLib.SentinelValueForInferParamInitRank, CNTK.CNTKLib.SentinelValueForInferParamInitRank, 1),
        device, outputName + "_W");
      
      var result = CNTK.CNTKLib.Convolution(W, input, CNTK.NDShape.CreateNDShape(new int[] { 1 }) /*strides*/, new CNTK.BoolVector(new bool[] { true }) /* sharing */, new CNTK.BoolVector(new bool[] { use_padding }));

      if ( use_bias ) {
        var b = new CNTK.Parameter(CNTK.NDShape.CreateNDShape(new int[] { 1, 1, CNTK.NDShape.InferredDimension }), 0.0f, device, outputName + "_b");
        result = CNTK.CNTKLib.Plus(result, b);
      }
           
      result = CNTK.CNTKLib.ReLU(result, outputName);

      return result;
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

    static public CNTK.Function BinaryAccuracy(CNTK.Variable prediction, CNTK.Variable labels) {
      var round_predictions = CNTK.CNTKLib.Round(prediction);
      var equal_elements = CNTK.CNTKLib.Equal(round_predictions, labels);
      var result = CNTK.CNTKLib.ReduceMean(equal_elements, new CNTK.Axis(0));
      return result;
    }

    static public CNTK.Function MeanSquaredError(CNTK.Variable prediction, CNTK.Variable labels) {
      var squared_errors = CNTK.CNTKLib.Square(CNTK.CNTKLib.Minus(prediction, labels));
      var result = CNTK.CNTKLib.ReduceMean(squared_errors, new CNTK.Axis(0));
      return result;
    }

    static public CNTK.Function MeanAbsoluteError(CNTK.Variable prediction, CNTK.Variable labels) {
      var absolute_errors = CNTK.CNTKLib.Abs(CNTK.CNTKLib.Minus(prediction, labels));
      var result = CNTK.CNTKLib.ReduceMean(absolute_errors, new CNTK.Axis(0));
      return result;
    }

  }
#endif
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
      Console.WriteLine(Environment.NewLine + "Download finished!");
      _semaphore.Release();
    }

    public static bool DownloadFile(string url, string fullPathWhereToSave, int timeoutInMilliSec) {
      return new FileDownloader(url, fullPathWhereToSave).StartDownload(timeoutInMilliSec);
    }
  }
}