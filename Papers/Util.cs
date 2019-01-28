using System;
using System.ComponentModel;
using System.IO;
using System.Net;
using System.Linq;
using System.Threading;
using System.Runtime.InteropServices;
using System.Collections.Generic;

using CC = CNTK.CNTKLib;
using C = CNTK;

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

  public class MiniBatch: IDisposable {
    public MiniBatch(
      float[][] buffers,
      C.NDShape[] shapes,
      int num_samples,
      variableLookup variableLookup_) {
      _buffers = buffers;
      _shapes = shapes;
      _num_samples = num_samples;
      _variableLookup = variableLookup_;
      _backingStorage_cpu = new C.NDArrayView[shapes.Length];
      _backingStorage_gpu = new C.NDArrayView[shapes.Length];
    }

    readonly float[][] _buffers;
    readonly C.NDShape[] _shapes;
    readonly int _num_samples;
    variableLookup _variableLookup;
    C.UnorderedMapVariableValuePtr _testMiniBatch = null;
    Dictionary<C.Variable, C.Value> _trainMiniBatch = null;
    C.NDArrayView[] _backingStorage_gpu;
    C.NDArrayView[] _backingStorage_cpu;

    public void Dispose() {
      if (_trainMiniBatch != null) {
        foreach (var v in _trainMiniBatch.Values) {
          v.Erase();
          v.Dispose();
        }
        _trainMiniBatch = null;
      }
      if (_testMiniBatch != null) {
        foreach (var v in _testMiniBatch.Values) {
          v.Erase();
          v.Dispose();
        }
        _testMiniBatch = null;
      }
      if ( _backingStorage_cpu!=null ) {
        for (int i=0; i<_backingStorage_cpu.Length; ++i) {
          _backingStorage_cpu[i].Dispose();
          _backingStorage_gpu[i].Dispose();
        }
        _backingStorage_cpu = null;
        _backingStorage_gpu = null;
      }
      _variableLookup = null;
    }

    public int get_num_samples() { return _num_samples; }

    public C.UnorderedMapVariableValuePtr testMiniBatch(C.DeviceDescriptor computeDevice) {
      if ( _testMiniBatch!=null ) { throw new InvalidOperationException();  }
      _testMiniBatch = new C.UnorderedMapVariableValuePtr();
      for (int i=0; i<_buffers.Length; i++) {
        _testMiniBatch.Add(_variableLookup(i), to_device(i, computeDevice));
      }
      return _testMiniBatch;
    }

    public Dictionary<C.Variable, C.Value> trainingMiniBatch(C.DeviceDescriptor computeDevice) {
      if ( _trainMiniBatch != null) { throw new InvalidOperationException(); }
      _trainMiniBatch = new Dictionary<C.Variable, C.Value>();
      for (int i = 0; i < _buffers.Length; i++) {
        _trainMiniBatch.Add(_variableLookup(i), to_device(i, computeDevice));
      }
      return _trainMiniBatch;
    }

    C.Value to_device(int index, C.DeviceDescriptor computeDevice) {
      var cpuDevice = C.DeviceDescriptor.CPUDevice;
      _backingStorage_cpu[index] = new C.NDArrayView(_shapes[index], _buffers[index], (uint)_buffers[index].Length, cpuDevice);
      _backingStorage_gpu[index] = _backingStorage_cpu[index].DeepClone(computeDevice);
      var rtrn = new C.Value(_backingStorage_gpu[index]);
      return rtrn;
    }
  }

  public interface IMiniBatchGenerator {
    MiniBatch next();
  }

  class MultiThreadedGenerator : IMiniBatchGenerator, IDisposable {
    const int _sleep_timeout_milliseconds = 50;
    const int MAX_QUEUE_SIZE = 64;
    IMiniBatchGenerator _generator;
    System.Collections.Concurrent.BlockingCollection<MiniBatch> _queue;
    List<System.Threading.Thread> _background_threads;
    bool _keep_going = true;

    void _backround_work() {
      var need_to_adjust_timeout = false;
      var sleep_timeout = _sleep_timeout_milliseconds;
      // Console.WriteLine($"Started background thread:{System.Threading.Thread.CurrentThread.ManagedThreadId}");

      while (_keep_going) {
        var queue_count = _queue.Count;
        if (queue_count < MAX_QUEUE_SIZE) {
          if ( need_to_adjust_timeout ) {
            need_to_adjust_timeout = false;
            if ( queue_count<MAX_QUEUE_SIZE/2) {
              sleep_timeout = Math.Max(sleep_timeout/2, 5);
            }
          }
          _queue.Add(_generator.next());
        }
        else {
          if ( need_to_adjust_timeout ) {
            need_to_adjust_timeout = false;
            sleep_timeout = (int)Math.Min(1.1 * sleep_timeout, 100 * _sleep_timeout_milliseconds);
          }
          // Console.WriteLine($"Queue Full. Thread:{System.Threading.Thread.CurrentThread.ManagedThreadId} about to sleep for {sleep_timeout}ms");
          System.Threading.Thread.Sleep(_sleep_timeout_milliseconds);
          need_to_adjust_timeout = true;
        }
      }
      // Console.WriteLine($"Background thread: {System.Threading.Thread.CurrentThread.ManagedThreadId} is done."); 
    }

    public MultiThreadedGenerator(int workers, IMiniBatchGenerator generator) {
      _generator = generator;
      _background_threads = new List<Thread>();
      _queue = new System.Collections.Concurrent.BlockingCollection<MiniBatch>();
      for (int i = 0; i < workers; i++) {
        var thread = new System.Threading.Thread(_backround_work);
        _background_threads.Add(thread);
        thread.Start();
      }
    }

    public MiniBatch next() {
      var rtrn = _queue.Take();
      return rtrn;
    }

    public void stopThreads() {
      if ( _background_threads==null ) { return; }
      foreach (var t in _background_threads) {
        if (t.IsAlive) { t.Join(); }
        // Console.WriteLine($"Background Thread:{t.ManagedThreadId} joined");
      }
      _background_threads = null;
    }

    public void Dispose() {
      // Console.WriteLine("Enqueuer.Dispose()");
      _keep_going = false;
      stopThreads();
    }
  }

  public delegate C.Variable variableLookup(int index);

  public class BaseMiniBatchGenerator : IDisposable {
    int _begin_pos = 0;
    int _end_pos = -1;
    int[] _indices;
    readonly object _lock = new object();

    public void Dispose() { }

    protected readonly C.Variable[] _input_variables;
    protected readonly int _batch_size;
    protected readonly bool _shuffle;
    protected readonly bool _only_once;
    protected readonly string _name;

    public C.Variable get_input_variable(int index) {
      return _input_variables[index];
    }

    int _epoch_counter = 0;

    protected int[] _update_pos_and_indices(int num_indices) {
      lock (_lock) {

        int[] result = null;

        // update the starting position in the indexing
        _begin_pos = _end_pos;

        // step 1: let's see if we need to re-compute the indices
        var this_is_a_new_epoch = ((_indices == null) || (_indices.Length == 0) || (_begin_pos >= _indices.Length));
        if (this_is_a_new_epoch) {
          _epoch_counter++;

          // if we have go over the data only once, then need to check how many epochs we have done so far
          if (_only_once && (_epoch_counter > 1)) {
            _indices = null;
            _begin_pos = _end_pos = -1;
            return result;
          }

          _indices = Engine.shuffled_indices(num_indices, should_shuffle: _shuffle);
          _begin_pos = 0;
        }
        _end_pos = Math.Min(_indices.Length, _begin_pos + _batch_size);
        // std::wcout << L"_begin_pos=" << _begin_pos << L", _end_pos=" << _end_pos << std::endl;

        result = new int[_end_pos - _begin_pos];
        Array.Copy(_indices, _begin_pos, result, 0, result.Length);
        return result;
      }
    }

    public BaseMiniBatchGenerator(
      C.Variable[] input_variables,
      int batch_size,
      bool shuffle,
      bool only_once,
      String name) {
      _input_variables = input_variables;
      _batch_size = batch_size;
      _shuffle = shuffle;
      _only_once = only_once;
      _name = name;
    }
  }

  class InMemoryMiniBatchGenerator : BaseMiniBatchGenerator, IMiniBatchGenerator {
    // [input_variable_index][row][column]
    readonly float[][][] _data;

    public InMemoryMiniBatchGenerator(
      float[][][] data,
      C.Variable[] input_variables,
      int batch_size,
      bool shuffle,
      bool only_once,
      string name) :
      base(input_variables, batch_size, shuffle, only_once, name) {
      _data = data;
      for (int i=0; i<_data.Length; ++i) {
        if (_data[i].Length != _data[0].Length) {
          throw new NotSupportedException();
        }
      }
    }

    public MiniBatch next() {
      var indices = _update_pos_and_indices(_data[0].Length);
      if (indices == null) { return null; }

      var buffers = new float[_data.Length][];
      var shapes = new C.NDShape[_data.Length];
      for (int i=0; i<_data.Length; ++i) {
        buffers[i] = Util.get_contiguous_buffer(_data[i], indices, 0, indices.Length);
        shapes[i] = get_input_variable(i).Shape.AppendShape(new int[] { indices.Length });
      }
      var rtrn = new MiniBatch(buffers, shapes, indices.Length, get_input_variable);

      return rtrn;
    }
  }

  public static class IO {
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

    public static string download_model_if_needed(string fullpath, string dstFilename, string downlink_url) {
      System.Net.ServicePointManager.SecurityProtocol = System.Net.SecurityProtocolType.Tls12;
      if (fullpath == null) {
        fullpath = fullpathForDownloadedFile(dstDir: "DownloadedModels", dstFilename: dstFilename);
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
  }

  static class Datasets {
    public class MNIST {
      public float[][] train_images { get; private set; }
      public float[][] test_images { get; private set; }
      public float[][] train_labels { get; private set; }
      public float[][] test_labels { get; private set; }

      public MNIST(int numTrainingImages=60000, int numTestImages=10000) {
        var cwd = System.IO.Directory.GetCurrentDirectory();
        var pos = cwd.LastIndexOf("Papers");
        var mnistPath = System.IO.Path.Combine(
          cwd.Substring(0, pos),
          "DeepLearning",
          "Ch_02_First_Look_At_A_Neural_Network",
          "mnist_data.zip");
        System.Diagnostics.Debug.Assert(System.IO.File.Exists(mnistPath));

        if (!System.IO.File.Exists("train_images.bin")) {
          System.IO.Compression.ZipFile.ExtractToDirectory(mnistPath, ".");
        }
        train_images = IO.load_binary_file("train_images.bin", numTrainingImages, 28 * 28);
        test_images = IO.load_binary_file("test_images.bin", numTestImages, 28 * 28);
        train_labels = IO.load_binary_file("train_labels.bin", numTrainingImages, 10);
        test_labels = IO.load_binary_file("test_labels.bin", numTestImages, 10);
      }
    }
  }

  static class Util {

    public static int argmax<T>(T[] values) where T: IComparable {
      if ( values==null ) { return -1; }
      if ( values.Length<=1 ) { return 0; }
      var max_value = values[0];
      var max_index = 0;
      for (int i=1; i<values.Length; i++) {
        if ( values[i].CompareTo(max_value)>0 ) {
          max_index = i;
          max_value = values[i];
        }
      }
      return max_index;
    }

    public static int np_prod(int[] values) { 
      // https://stackoverflow.com/a/20132916
      var product_of_all_elements = values.Aggregate(1, (a, b) => a * b);
      return product_of_all_elements;
    }

    public static float[] get_contiguous_buffer(float[][] data, int[] indices, int indices_begin, int indices_end) {
      var num_indices = indices_end - indices_begin;
      var buffer = new float[num_indices * data[0].Length];
      int pos = 0;
      for (int index = indices_begin; index != indices_end; index++) {
        var src = data[indices[index]];
        Array.Copy(src, 0, buffer, pos, src.Length);
        System.Diagnostics.Debug.Assert(src.Length == data[0].Length);
        pos += src.Length;
      }
      return buffer;
    }

    public static C.Variable inputVariable(int[] dimensions, String name="", bool needsGradient=false) {
      // only batch axis
      var dynamicAxes = new C.Axis[] { C.Axis.DefaultBatchAxis() };
      var rtrn = C.Variable.InputVariable(
        shape: dimensions, 
        dataType: C.DataType.Float, 
        name: name, 
        dynamicAxes: dynamicAxes, 
        needsGradient: needsGradient);
      return rtrn;
    }

    public static C.Variable placeholderVariable(int[] dimensions, String name = "") {
      // only batch axis
      var dynamicAxes = new CNTK.AxisVector() { C.Axis.DefaultBatchAxis() };
      var rtrn = CC.PlaceholderVariable(
        shape: dimensions,
        dataType: C.DataType.Float,
        name: name,
        dynamicAxes: dynamicAxes);
      return rtrn;
    }


    public static byte[] convert_from_channels_first(float[] img, float[] offsets = null, float scaling = 1.0f, bool invertOrder = false) {
      if (offsets == null) { offsets = new float[3]; }
      var img_data = new byte[img.Length];
      var image_size = img.Length / 3;
      for (int i = 0; i < img_data.Length; i += 3) {
        img_data[i + 1] = (byte)Math.Max(0, Math.Min(scaling * img[i / 3 + image_size] + offsets[1], 255));
        if (invertOrder) {
          img_data[i + 2] = (byte)Math.Max(0, Math.Min(scaling * img[i / 3] + offsets[0], 255));
          img_data[i] = (byte)Math.Max(0, Math.Min(scaling * img[i / 3 + 2 * image_size] + offsets[2], 255));
        }
        else {
          img_data[i] = (byte)Math.Max(0, Math.Min(scaling * img[i / 3] + offsets[0], 255));
          img_data[i + 2] = (byte)Math.Max(0, Math.Min(scaling * img[i / 3 + 2 * image_size] + offsets[2], 255));
        }
      }
      return img_data;
    }

    static public C.DeviceDescriptor get_compute_device() {
      foreach (var gpuDevice in C.DeviceDescriptor.AllDevices()) {
        if (gpuDevice.Type == C.DeviceKind.GPU) { return gpuDevice; }
      }
      return C.DeviceDescriptor.CPUDevice;
    }

  }

  static public class Logging {
    static readonly string equalSigns = "=============================================================================";
    static readonly string underscores = "_____________________________________________________________________________";
    static void summary(C.Function rootFunction, List<string> entries, ISet<string> visited) {
      if (visited == null) {
        visited = new HashSet<string>();
      }

      if (rootFunction.IsComposite) {
        summary(rootFunction.RootFunction, entries, visited);
        return;
      }

      if (visited.Contains(rootFunction.Uid)) {
        return;
      }
      visited.Add(rootFunction.Uid);

      var numParameters = 0;
      foreach (var rootInput in rootFunction.Inputs) {
        if (rootInput.IsParameter && !rootInput.IsConstant) {
          numParameters += rootInput.Shape.TotalSize;
        }
        if ((rootInput.Owner == null) || visited.Contains(rootInput.Owner.Uid)) { continue; }
        summary(rootInput.Owner, entries, visited);
      }
      for (int i = 0; i < rootFunction.Outputs.Count; i++) {
        var line = $"{rootFunction.Name,-29}{rootFunction.Outputs[i].Shape.AsString(),-26}{numParameters}";
        entries.Add(line);
      }
      entries.Add(underscores);
    }

    public static string shape_desc(C.Variable node) {
      var static_shape = node.Shape.AsString();
      var dyn_axes = node.DynamicAxes;
      var dyn_axes_description = (dyn_axes.Count == 1) ? "[#]" : "[*, #]";
      return $"({static_shape}, {dyn_axes_description})";
    }

    public static IList<C.Function> find_function_with_input(C.Function root, C.Function inputFunction) {
      var list = new List<C.Function>();
      root = root.RootFunction;
      inputFunction = inputFunction.RootFunction;
      var root_uid = root.Uid;
      var stack = new Stack<object>();
      stack.Push(root);
      var visited_uids = new HashSet<string>();
      while (stack.Count > 0) {
        var popped = stack.Pop();
        if (popped is C.Variable) {
          var v = (C.Variable)popped;
          if (v.IsOutput) {
            stack.Push(v.Owner);
          }
          continue;
        }
        if (popped is IList<C.Variable>) {
          foreach (var pv in (IList<C.Variable>)popped) {
            stack.Push(pv);
          }
          continue;
        }
        var node = (C.Function)popped;
        if (visited_uids.Contains(node.Uid)) { continue; }
        node = node.RootFunction;
        stack.Push(node.RootFunction.Inputs);

        for (int i = 0; i < node.Inputs.Count; i++) {
          var input = node.Inputs[i];
          if (input.Uid == inputFunction.Output.Uid) {
            list.Add(node);
          }
        }
        visited_uids.Add(node.Uid);
      }
      return list;
    }

    public static List<string> detailed_summary(C.Function root, bool print = true) {
      // This is based on the C.logging.graph.plot, but without the actual plotting part
      // Walks through every node of the graph starting at ``root``, creates a network graph, and returns a network description
      var model = new List<string>();
      root = root.RootFunction;
      var root_uid = root.Uid;
      var stack = new Stack<object>();
      stack.Push(root);
      var visited_uids = new HashSet<string>();
      while (stack.Count > 0) {
        var popped = stack.Pop();
        if (popped is C.Variable) {
          var v = (C.Variable)popped;
          if (v.IsOutput) {
            stack.Push(v.Owner);
          }
          else if (v.IsInput) {
            model.Add(v.AsString());
          }
          continue;
        }
        if (popped is IList<C.Variable>) {
          foreach (var pv in (IList<C.Variable>)popped) {
            stack.Push(pv);
          }
          continue;
        }
        var node = (C.Function)popped;
        if (visited_uids.Contains(node.Uid)) { continue; }
        node = node.RootFunction;
        stack.Push(node.RootFunction.Inputs);
        var line = new System.Text.StringBuilder($"{node.Name} : {node.OpName} ({node.Uid}) :");
        line.Append("(");
        for (int i = 0; i < node.Inputs.Count; i++) {
          var input = node.Inputs[i];
          if (node.IsBlock && input.IsConstant) { continue; }
          line.Append(input.Uid);
          if (i != (node.Inputs.Count - 1)) {
            line.Append(", ");
          }
        }
        line.Append(") -> ");
        foreach (var v in node.Outputs) {
          model.Add(line.ToString() + "\t" + shape_desc(v));
        }
        visited_uids.Add(node.Uid);
      }
      model.Reverse();

      if (print) {
        for (int i = 0; i < model.Count; i++) {
          Console.WriteLine(model[i]);
        }
      }
      return model;
    }

    public static void summary(C.Function rootFunction) {
      var entries = new List<string>();
      entries.Add(underscores);
      entries.Add("Layer (type)                 Output Shape              Trainable Parameters #");
      entries.Add(equalSigns);
      summary(rootFunction, entries, null);
      foreach (var v in entries) {
        Console.WriteLine(v);
      }
    }

    static public void log_number_of_parameters(C.Function model, bool show_filters=true) {
      Console.WriteLine("\nModel Summary");
      for (int i = 0; i < model.Arguments.Count; i++) {
        Console.WriteLine("\tInput = " + model.Arguments[i].Shape.AsString());
      }
      for (int i = 0; i < model.Outputs.Count; i++) {
        Console.WriteLine("\tOutput = " + model.Outputs[i].Shape.AsString());
      }
      Console.WriteLine("");

      var numParameters = 0;
      foreach (var x in model.Parameters()) {
        var shape = x.Shape;
        var p = shape.TotalSize;
        if (show_filters) { Console.WriteLine(string.Format("\tFilter Shape:{0,-30} Param #:{1}", shape.AsString(), p)); }
        numParameters += p;
      }
      Console.WriteLine(string.Format("\nTotal Number of Parameters: {0:N0}", numParameters));
      Console.WriteLine("---\n");
    }
  }


  static public class Engine {
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

    public static int SizeOf<T>() where T : struct {
      // https://github.com/dotnet/roslyn/issues/3208#issuecomment-210134781
      return Marshal.SizeOf(default(T));
    }

    public static T[] convert_jagged_array_to_single_dimensional_array<T>(T[][] src) where T : struct {
      var numRows = src.Length;
      var numColumns = src[0].Length;
      var numBytesInRow = numColumns * SizeOf<T>();
      var dst = new T[numRows * numColumns];
      var dstOffset = 0;
      for (int row = 0; row < numRows; row++) {
        System.Diagnostics.Debug.Assert(src[row].Length == numColumns);
        System.Buffer.BlockCopy(src[row], 0, dst, dstOffset, numBytesInRow);
        dstOffset += numBytesInRow;
      }
      return dst;
    }

    public static void swap<T>(T[] array, int n, int k) {
      var temp = array[n];
      array[n] = array[k];
      array[k] = temp;
    }

    public static void shuffle<T>(T[] array) {
      lock (_rng) {
        // https://stackoverflow.com/a/110570
        var n = array.Length;
        while (n > 1) {
          var k = _rng.Next(n--);
          swap(array, n, k);
        }
      }
    }

    static readonly Random _rng = new Random(Seed: 123456789);

    public static void shuffle<T1, T2>(T1[] array1, T2[] array2) {
      System.Diagnostics.Debug.Assert(array1.Length == array2.Length);
      lock (_rng) {
        var n = array1.Length;
        while (n > 1) {
          var k = _rng.Next(n--);
          swap(array1, n, k);
          swap(array2, n, k);
        }
      }
    }

    public static int[] shuffled_indices(int N, bool should_shuffle = true) {
      var array = new int[N];
      for (int i = 0; i < N; i++) { array[i] = i; }
      if (should_shuffle) {
        shuffle(array);
      }
      return array;
    }

    public static T[] concat<T>(params T[][] arguments) where T : struct {
      var list = new List<T>();
      for (int i = 0; i < arguments.Length; i++) {
        list.AddRange(arguments[i]);
      }
      return list.ToArray();
    }

    public static int[] make_ones(int numOnes) {
      var ones = new int[numOnes];
      for (int i = 0; i < numOnes; i++) { ones[i] = 1; }
      return ones;
    }

    public static T[] pad_to_shape<T>(int[] filter_shape, T value) where T : struct {
      var result = new T[filter_shape.Length];
      for (int i = 0; i < result.Length; i++) { result[i] = value; }
      return result;
    }
  }

  static public class Layers {
    static public C.Function Dense(
      C.Variable input, 
      int outputDim, 
      C.DeviceDescriptor device, 
      string outputName = "",
      Func<C.Variable, String, C.Function> activation=null) {
      var shape = C.NDShape.CreateNDShape(new int[] { outputDim, C.NDShape.InferredDimension });
      var timesParam = new C.Parameter(
        shape, 
        C.DataType.Float,
        CC.HeNormalInitializer(0.921, 1, CC.SentinelValueForInferParamInitRank, 1821),         
        device, "timesParam_" + outputName);
      var timesFunction = CC.Times(timesParam, input, 1 /* output dimension */, 0 /* CNTK should infer the input dimensions */);
      var plusParam = new C.Parameter(new int[] { outputDim }, 0.0f, device, "plusParam_" + outputName);
      var result = CC.Plus(plusParam, timesFunction, outputName);
      if ( activation!=null) {
        result = activation(result, outputName + "_activation");
      }
      return result;
    }

    static public C.Function Embedding(C.Variable x, int shape, C.DeviceDescriptor computeDevice, float[][] weights = null, string name = "") {
      C.Function result;
      // based on the Embedding defined in https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/layers/layers.py      
      if (weights == null) {
        var weight_shape = new int[] { shape, C.NDShape.InferredDimension };
        var E = new C.Parameter(
          weight_shape,
          C.DataType.Float,
          CC.GlorotUniformInitializer(),
          computeDevice, name: "embedding_" + name);
        result = CC.Times(E, x);
      }
      else {
        var weight_shape = new int[] { shape, x.Shape.Dimensions[0] };
        System.Diagnostics.Debug.Assert(shape == weights[0].Length);
        System.Diagnostics.Debug.Assert(weight_shape[1] == weights.Length);
        var w = Engine.convert_jagged_array_to_single_dimensional_array(weights);
        var ndArrayView = new C.NDArrayView(weight_shape, w, computeDevice, readOnly: true);
        var E = new C.Constant(ndArrayView, name: "fixed_embedding_" + name);
        result = CC.Times(E, x);
      }
      return result;
    }

    static public C.Function ConvolutionTranspose(
      C.Variable x,
      C.DeviceDescriptor computeDevice,
      int[] filter_shape,
      int num_filters,
      Func<C.Variable, C.Function> activation = null,
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
      var sharing = Engine.pad_to_shape(filter_shape, true);
      var padding = Engine.pad_to_shape(filter_shape, use_padding);
      if (reduction_rank != 1) { throw new NotSupportedException("reduction_rank should be 1"); }
      padding = Engine.concat(padding, new bool[] { false });
      if (dilation == null) {
        dilation = Engine.pad_to_shape(filter_shape, 1);
      }
      var output_channels_shape = new int[] { num_filters };
      var kernel_shape = Engine.concat(filter_shape, output_channels_shape, new int[] { C.NDShape.InferredDimension });
      var output_full_shape = output_shape;
      if (output_full_shape != null) {
        output_full_shape = Engine.concat(output_shape, output_channels_shape);
      }
      var filter_rank = filter_shape.Length;
      var init = CC.GlorotUniformInitializer(CC.DefaultParamInitScale, CC.SentinelValueForInferParamInitRank, CC.SentinelValueForInferParamInitRank, 1);
      var W = new C.Parameter(kernel_shape, C.DataType.Float, init, computeDevice, name = "W");
      var r = CC.ConvolutionTranspose(
        convolutionMap: W,
        operand: x,
        strides: strides,
        sharing: new C.BoolVector(sharing),
        autoPadding: new C.BoolVector(padding),
        outputShape: output_full_shape,
        dilation: dilation,
        reductionRank: reduction_rank,
        maxTempMemSizeInSamples: max_temp_mem_size_in_samples);
      if (use_bias) {
        var b_shape = Engine.concat(Engine.make_ones(filter_shape.Length), output_channels_shape);
        var b = new C.Parameter(b_shape, 0.0f, computeDevice, "B");
        r = CC.Plus(r, b);
      }
      if (activation != null) {
        r = activation(r);
      }
      return r;
    }

    static public C.Function Convolution1DWithReLU(
      C.Variable input,
      int num_output_channels,
      int filter_shape,
      C.DeviceDescriptor device,
      bool use_padding = false,
      bool use_bias = true,
      int[] strides = null,
      string outputName = "") {
      var convolution_map_size = new int[] { filter_shape, C.NDShape.InferredDimension, num_output_channels };
      if (strides == null) { strides = new int[] { 1 }; }
      var rtrn = _Convolution(convolution_map_size, input, device, use_padding, use_bias, strides, CC.ReLU, outputName);
      return rtrn;
    }

    static public C.Function Convolution2D(
      C.Variable input,
      int num_output_channels,
      int[] filter_shape,
      C.DeviceDescriptor device,
      Func<C.Variable, string, C.Function> activation = null,
      bool use_padding = false,
      bool use_bias = true,
      int[] strides = null,
      string name = "") {
      var convolution_map_size = new int[] { filter_shape[0], filter_shape[1], C.NDShape.InferredDimension, num_output_channels };
      if (strides == null) { strides = new int[] { 1 }; }
      var rtrn = _Convolution(convolution_map_size, input, device, use_padding, use_bias, strides, activation, name);
      return rtrn;
    }

    static C.Function _Convolution(
      int[] convolution_map_size,
      C.Variable input,
      C.DeviceDescriptor device,
      bool use_padding,
      bool use_bias,
      int[] strides,
      Func<C.Variable, string, C.Function> activation = null,
      string outputName = "") {
      var W = new C.Parameter(
        convolution_map_size,
        C.DataType.Float,
        CC.HeNormalInitializer(0.921, -1, 2, 1821),
        device, outputName + "_W");

      var result = CC.Convolution(W, input, strides, new C.BoolVector(new bool[] { true }) /* sharing */, new C.BoolVector(new bool[] { use_padding }));

      if (use_bias) {
        var num_output_channels = convolution_map_size[convolution_map_size.Length - 1];
        var b_shape = Engine.concat(Engine.make_ones(convolution_map_size.Length - 2), new int[] { num_output_channels });
        var b = new C.Parameter(b_shape, 0.0f, device, outputName + "_b");
        result = CC.Plus(result, b);
      }

      if (activation != null) {
        result = activation(result, outputName);
      }
      return result;
    }
  }

  public static class Model {
    public static C.Function invoke_model(C.Function model, C.Variable[] inputs) {
      var arguments = model.Arguments;
      System.Diagnostics.Debug.Assert(arguments.Count == inputs.Length);

      var replacements = new Dictionary<C.Variable, C.Variable>();
      for (int i = 0; i < arguments.Count; i++) {
        replacements.Add(arguments[i], inputs[i]);
      }
      var result = model.Clone(C.ParameterCloningMethod.Share, replacements);
      return result;
    }

    public static IList<IList<float>> predict(C.Function model, float[][] buffers, C.DeviceDescriptor computeDevice) {
      // For each input variable, we have a float[] with the input values
      var shapes = model.Arguments.Select((v) => v.Shape).ToArray();
      variableLookup vl = (int index) => { return model.Arguments[index]; };
      var miniBatch = new MiniBatch(buffers, shapes, 1, vl);
      var testMiniBatch = miniBatch.testMiniBatch(computeDevice);

      var outputShape = model.Output.Shape;
      var outputVariable = model.Output;
      var outputMap = new Dictionary<C.Variable, C.Value>() { { outputVariable, null } };

      model.Evaluate(testMiniBatch, outputMap, computeDevice);
      var outputValue = outputMap[outputVariable];
      var rtrn = outputValue.GetDenseData<float>(outputVariable);
      miniBatch.Dispose(); miniBatch = null;
      return rtrn;
    }

    static void process_epoch(
      IMiniBatchGenerator generator,
      bool training_mode,
      object minibatch_processor,
      int steps,
      C.DeviceDescriptor computeDevice, 
      out double metric_loss, out double metric_evaluation) {
      var result = new Tuple<double, double>(0, 0);
      var current_step = 0;
      var num_samples = 0;
      metric_loss = 0;
      metric_evaluation = 0;
      while (current_step < steps) {
        var minibatch = generator.next();
        if (minibatch.get_num_samples() == 0) { break; }
        var isSweepEndInArguments = (++current_step >= steps);        
        if (training_mode) {
          var trainer = (C.Trainer)minibatch_processor;
          var trainingMiniBatch = minibatch.trainingMiniBatch(computeDevice);
          trainer.TrainMinibatch(trainingMiniBatch, isSweepEndInArguments, computeDevice);
          metric_loss += trainer.PreviousMinibatchLossAverage() * minibatch.get_num_samples();
          if (trainer.EvaluationFunction() != null) {
            metric_evaluation += trainer.PreviousMinibatchEvaluationAverage() * minibatch.get_num_samples();
          }
          else {
            metric_evaluation = Double.MaxValue;
          }
        }
        else {
          var evaluator = (C.Evaluator)minibatch_processor;
          var testMiniBatch = minibatch.testMiniBatch(computeDevice);
          metric_evaluation += evaluator.TestMinibatch(testMiniBatch, computeDevice) * minibatch.get_num_samples();
        }
        num_samples += minibatch.get_num_samples();
        minibatch.Dispose(); minibatch = null;
      }
      metric_loss /= num_samples;
      if (metric_evaluation != Double.MaxValue) {
        metric_evaluation /= num_samples;
      }
    }

    public static void fit_generator(
      C.Function model,
      C.Learner learner,
      C.Trainer trainer,

      C.Evaluator evaluator,
      int batch_size,
      int epochs,
      IMiniBatchGenerator trainGenerator,
      int trainSteps,
      IMiniBatchGenerator validationGenerator,
      int validationSteps,
      C.DeviceDescriptor computeDevice,
      string prefix,
      string trainingLossMetricName="",
      string trainingEvaluationMetricName="",
      string validationMetricName=""
      ) {
      if ( string.IsNullOrEmpty(trainingLossMetricName) ) {
        trainingLossMetricName = "Training Metric";
      }
      if ( string.IsNullOrEmpty(trainingEvaluationMetricName) ) {
        trainingEvaluationMetricName = "Training Evaluation Metric";
      }
      if ( string.IsNullOrEmpty(validationMetricName)) {
        validationMetricName = "Validation Metric";
      }

      var min_validation_metric = 10.0;
      var last_epoch_of_loss_improvement = 0;
      var last_epoch_of_lr_reduction = Int32.MaxValue;
      var patience = 5;
      var min_lr = 0.00001;

      for (int current_epoch = 0; current_epoch < epochs; current_epoch++) {
        var epoch_start_time = DateTime.Now;
        process_epoch(trainGenerator, true, trainer, trainSteps, computeDevice, out var training_metric_loss, out var training_metric_evaluation);
        process_epoch(validationGenerator, false, evaluator, validationSteps, computeDevice, out var dummy, out var validation_metric);

        validation_metric = ((long)(1e6 * validation_metric)) / 1e6;
        if (validation_metric < min_validation_metric) {
          min_validation_metric = validation_metric;
          last_epoch_of_loss_improvement = current_epoch;

          var filename = $"{prefix}{current_epoch}_val_metric_{validation_metric:F4}.model";
          model.Save(filename);          
        }
        else if (current_epoch > (40 + last_epoch_of_loss_improvement)) {
          break;
        }
        else if (current_epoch > (patience + ((last_epoch_of_lr_reduction < epochs) ? last_epoch_of_lr_reduction : last_epoch_of_loss_improvement))) {
          var old_lr = learner.LearningRate();
          if (old_lr > min_lr) {
            var lr = Math.Max(0.5 * old_lr, min_lr);
            last_epoch_of_lr_reduction = current_epoch;
            learner.SetLearningRateSchedule(new C.TrainingParameterScheduleDouble(lr, (uint)batch_size));
            Console.WriteLine($"Reducing learning rate from {old_lr} to {lr}");
          }
        }

        var end_time = DateTime.Now;
        var elapsedTime = DateTime.Now.Subtract(epoch_start_time);

        Console.Write($"Epoch {current_epoch + 1:D2}/{epochs}, {elapsedTime.TotalSeconds:F3} seconds. ");
        Console.Write($"{trainingLossMetricName}: {training_metric_loss:F4}");
        if ( training_metric_evaluation!=Double.MaxValue) {
          Console.Write($", {trainingEvaluationMetricName}: {training_metric_evaluation:F4}");
        }
        Console.WriteLine($". {validationMetricName}: {validation_metric:F4}.");
      }
    }
  }

  static public class Losses {
    static public C.Function BinaryAccuracy(C.Variable prediction, C.Variable labels) {
      var round_predictions = CC.Round(prediction);
      var equal_elements = CC.Equal(round_predictions, labels);
      var result = CC.ReduceMean(equal_elements, C.Axis.AllStaticAxes());
      return result;
    }

    static public C.Function MeanSquaredError(C.Variable prediction, C.Variable labels) {
      var squared_errors = CC.Square(CC.Minus(prediction, labels));
      var result = CC.ReduceMean(squared_errors, new C.Axis(0)); // TODO -- allStaticAxes?
      return result;
    }

    static public C.Function MeanAbsoluteError(C.Variable prediction, C.Variable labels) {
      var absolute_errors = CC.Abs(CC.Minus(prediction, labels));
      var result = CC.ReduceMean(absolute_errors, new C.Axis(0)); // TODO -- allStaticAxes? 
      return result;
    }
  }

  static public class Augmentations {
    static public float[] translate(float[] src, int width, int height, int delta_x, int delta_y) {
      System.Diagnostics.Debug.Assert(src.Length == width * height);
      var dst = new float[src.Length];
      for (int row=0; row<height; row++) {
        var src_row = row - delta_y;
        if ( (src_row<0) || (src_row>=height) ) { continue; }
        for (int col=0; col<width; col++) {
          var src_col = col - delta_x;
          if ( (src_col<0) || (src_col>=width) ) { continue; }
          var p = row * width + col;
          var src_p = src_row * width + src_col;
          dst[p] = src[src_p];
        }
      }
      return dst;
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