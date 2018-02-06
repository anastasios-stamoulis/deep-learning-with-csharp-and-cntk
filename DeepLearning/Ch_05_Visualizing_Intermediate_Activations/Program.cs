using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

using DeepLearningWithCNTK;

public static class ColorUtils {
  public static System.Drawing.Color ToMediaColor(this byte[] rgb) {
    return System.Drawing.Color.FromArgb(rgb[0], rgb[1], rgb[2]);
  }
}


namespace Ch_05_Visualizing_Intermediate_Activations {

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

    static public System.Drawing.Bitmap createBitmap(float[] src, int gridIndex, int width, int height, bool adjustColorRange, int numChannels) {
      SciColorMaps.Portable.ColorMap cmap = null;
      var colorScaleFactor = 1.0;
      var numPixels = width * height;
      if (adjustColorRange) {        
        var maxValue = src.Skip(gridIndex * numPixels * numChannels).Take(numPixels * numChannels).Max();
        var minValue = src.Skip(gridIndex * numPixels * numChannels).Take(numPixels * numChannels).Min();
        if (numChannels == 1) {
          cmap = new SciColorMaps.Portable.ColorMap("viridis", minValue, maxValue);
        }
        colorScaleFactor = (float)(254.0 / maxValue);
      }

      var bitmap = new System.Drawing.Bitmap(width, height);
      
      var srcStart = gridIndex * numPixels;
      for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
          var pos = srcStart + row * width + col;

          if (cmap!=null) {
            var rgb = cmap[colorScaleFactor * src[pos]];
            var color = System.Drawing.Color.FromArgb(rgb[0], rgb[1], rgb[2]);
            bitmap.SetPixel(col, row, color);
          }
          else {
            var b = (int)(colorScaleFactor * src[pos]);
            var g = (numChannels == 1) ? b : (int)(colorScaleFactor * src[pos + numPixels]);
            var r = (numChannels == 1) ? b : (int)(colorScaleFactor * src[pos + 2 * numPixels]);
            bitmap.SetPixel(col, row, System.Drawing.Color.FromArgb(r, g, b));
          }
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
      for (int row=0; row<gridLength; row++) {
        grid.RowDefinitions.Add(new System.Windows.Controls.RowDefinition());
        for (int column=0; column<gridLength; column++) {
          if (row==0) { grid.ColumnDefinitions.Add(new System.Windows.Controls.ColumnDefinition()); }
          var gridIndex = (row * gridLength + column);
          var bitmap = WPFUtil.createBitmap(images, gridIndex, width, height, adjustColorRange: true, numChannels:1);
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

  class PlotWindow : System.Windows.Window {
    public PlotWindow(string imagefilename) {
      var image = new System.Windows.Controls.Image() {
        Source = new System.Windows.Media.Imaging.BitmapImage(new Uri(imagefilename, UriKind.Relative)),
        Stretch = System.Windows.Media.Stretch.None
      };
      Title = imagefilename;
      Content = image;
      SizeToContent = System.Windows.SizeToContent.WidthAndHeight;
    }

    public PlotWindow(OpenCvSharp.Mat src, string title) {
      var image = new System.Windows.Controls.Image() {
        Source = OpenCvSharp.Extensions.BitmapSourceConverter.ToBitmapSource(src),
        Stretch = System.Windows.Media.Stretch.None
      };
      Title = title;
      Content = image;
      SizeToContent = System.Windows.SizeToContent.WidthAndHeight;
    }

  }

  class Program {
    static readonly string cat_filename = "cat.1700.jpg";

    [STAThread]
    static void Main(string[] args) {
      //var caffeModelFilePath = VGG16.download_model_if_needed();
      //var image = new float[150 * 150 * 3];
      //CPPUtil.compute_image(image, caffeModelFilePath, 0);
      new Program().run();
    }

    void testing() {
      var app = new System.Windows.Application();
      var p1 = new PlotWindow(cat_filename);
      var p2 = new PlotWindow(cat_filename);
      p1.Show();
      Console.WriteLine("step 1");
      p2.Show();
      Console.WriteLine("step 2");

      // read, and resize an OpenCV matrix
      var src = OpenCvSharp.Cv2.ImRead(cat_filename);
      OpenCvSharp.Cv2.Resize(src, src, new OpenCvSharp.Size(150, 150));

      // Convert to 3 32-bit (float) channels format
      var dst_bgr = new OpenCvSharp.MatOfPoint3f();
      src.ConvertTo(dst_bgr, OpenCvSharp.MatType.CV_32FC3);

      // let's remove all the blue, green
      var buffer = dst_bgr.ToArray();
      for (int i = 0; i < buffer.Length; i++) {
        buffer[i].X = 0;
        buffer[i].Y = 0;
      }
      dst_bgr = new OpenCvSharp.MatOfPoint3f(150, 150, buffer);

      // let's convert it back to 3 8-bit channels
      var recovered_src = new OpenCvSharp.Mat();
      dst_bgr.ConvertTo(recovered_src, OpenCvSharp.MatType.CV_8UC3);

      // display an OpenCV matrix -- works as long as the underlying format is OpenCvSharp.MatType.CV_8UC3
      new PlotWindow(recovered_src, "from opencv").Show();

      app.Run();
    }

    void print_debugging_info() {
      if ( computeDevice==null ) {
        computeDevice = Util.get_compute_device();
      }

      var features = CNTK.Variable.InputVariable(new int[] { 150, 150, 3 }, CNTK.DataType.Float, "features");
      var adjusted_features = CNTK.CNTKLib.Plus(CNTK.Constant.Scalar<float>((float)(-110), computeDevice), features);
      var scalar_factor = CNTK.Constant.Scalar<float>((float)(1.0 / 255.0), computeDevice);
      var scaled_features = CNTK.CNTKLib.ElementTimes(scalar_factor, adjusted_features);

      var convolution_map_size = new int[] { 1, 1, CNTK.NDShape.InferredDimension, 3 };

      var W = new CNTK.Parameter(
        CNTK.NDShape.CreateNDShape(convolution_map_size),
        CNTK.DataType.Float,
        CNTK.CNTKLib.GlorotUniformInitializer(CNTK.CNTKLib.DefaultParamInitScale, CNTK.CNTKLib.SentinelValueForInferParamInitRank, CNTK.CNTKLib.SentinelValueForInferParamInitRank, 1),
        computeDevice);

      var result = CNTK.CNTKLib.Convolution(W, scaled_features, 
        strides: CNTK.NDShape.CreateNDShape(new int[] { 1 }), 
        sharing: new CNTK.BoolVector(new bool[] { false }), 
        autoPadding: new CNTK.BoolVector(new bool[] { true }) );

      var model = VGG16.get_model(result, computeDevice);

      Util.PredorderTraverse(model);
      var shape = model.Output.Shape;
      Console.WriteLine(shape.AsString());
    }

    float[] load_image_in_channels_first_format(string path, int width, int height) {
      var src = OpenCvSharp.Cv2.ImRead(path);
      OpenCvSharp.Cv2.Resize(src, src, new OpenCvSharp.Size(width, height));
      var dst_bgr = new OpenCvSharp.MatOfPoint3f();
      src.ConvertTo(dst_bgr, OpenCvSharp.MatType.CV_32FC3);

      int numPixels = width * height;
      var result = new float[3 * numPixels];
      var buffer = dst_bgr.ToArray();
      var pos = 0;
      for (int row = 0; row < height; row++) {
        for (int column = 0; column < width; column++, pos++) {
          result[pos] = buffer[pos].X;
          result[pos + numPixels] = buffer[pos].Y;
          result[pos + 2 * numPixels] = buffer[pos].Z;
        }
      }
      return result;
    }

    void run() {
      Console.Title = "Ch_05_Visualizing_Intermediate_Activations";
      computeDevice = Util.get_compute_device();

      var features = CNTK.Variable.InputVariable(new int[] { 150, 150, 3 }, CNTK.DataType.Float, "features");
      var adjusted_features = CNTK.CNTKLib.Plus(CNTK.Constant.Scalar<float>((float)(-110), computeDevice), features, "adjusted features");

      var scalar_factor = CNTK.Constant.Scalar<float>((float)(1.0 / 255.0), computeDevice);
      var scaled_features = CNTK.CNTKLib.ElementTimes(scalar_factor, adjusted_features, "scaled features");
      var base_model = VGG16.get_model(scaled_features, computeDevice);
      Util.summary(base_model);

      var app = new System.Windows.Application();

      var layer_names = new string[] { "pool1", "pool2", "pool3" };
      var num_entries = new int[] { 64, 64, 256 };
      for (int i = 0; i < layer_names.Length; i++) {

        var intermediate_node = base_model.FindByName(layer_names[i]);
        var model = CNTK.CNTKLib.Combine(new CNTK.VariableVector() { intermediate_node.Output });

        var image = load_image_in_channels_first_format(cat_filename, 150, 150);
        var image_tensor = CNTK.Value.CreateBatch(features.Shape, image, computeDevice);

        var input_d = new Dictionary<CNTK.Variable, CNTK.Value>() { { features, image_tensor } };
        var output_d = new Dictionary<CNTK.Variable, CNTK.Value>() { { model.Output, null } };
        model.Evaluate(input_d, output_d, computeDevice);

        var outputValues = output_d[intermediate_node.Output].GetDenseData<float>(intermediate_node.Output);
        var feature_height = intermediate_node.Output.Shape[0];
        var feature_width = intermediate_node.Output.Shape[1];
        var activations = outputValues[0].Take(num_entries[i] * feature_width * feature_height).ToArray();

        var window = new PlotWindowBitMap(layer_names[i], activations, feature_height, feature_width, 1);
        window.Show();
      }
      app.Run();
    }

    CNTK.DeviceDescriptor computeDevice;
  }
}
