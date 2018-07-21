using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ch_08_Deep_Dream {
  class Program {
    const string PYTHON_HOME_ = "C:\\local_tools\\Python3.6.6";
    const string PYTHONNET_DLL_ = "C:\\local_tools\\Python3.6.6\\Lib\\site-packages\\Python.Runtime.dll";
    const string KERAS_PORTED_CODE_ = "D:\\GitHub\\deep-learning-with-cntk-and-csharp\\Python";

    static string get_the_path_of_the_elephant_image() {
      var cwd = System.IO.Directory.GetCurrentDirectory();
      var pos = cwd.LastIndexOf("DeepLearning\\Ch_08_Deep_Dream");
      var base_path = cwd.Substring(0, pos);
      var image_path = System.IO.Path.Combine(base_path, "DeepLearning", "Ch_05_Class_Activation_Heatmaps", "creative_commons_elephant.jpg");
      return image_path;
    }

    static void Main(string[] args) {
      // make sure that the hard-coded values have been set up correctly
      if ( System.IO.Directory.Exists(PYTHON_HOME_)==false ) {
        throw new NotSupportedException("Please set PYTHON_HOME_ properly");
      }
      if ( System.IO.File.Exists(PYTHONNET_DLL_)==false ) {
        throw new NotSupportedException("Probably you have not pip-installed pythonnet");
      }
      if ( System.IO.Directory.Exists(KERAS_PORTED_CODE_)==false ) {
        throw new NotSupportedException("Need to initialize KERAS_PORTED_CODE_");
      }
      System.Console.Title = "Ch_08_Deep_Dream";

      // set model_path, and image_path
      var vgg16_model_path = DeepLearningWithCNTK.VGG16.download_model_if_needed();
      var image_path = get_the_path_of_the_elephant_image();

      // modify the environment variables
      var to_be_added_to_path = PYTHON_HOME_ + ";" + KERAS_PORTED_CODE_;
      var path = Environment.GetEnvironmentVariable("PATH");
      path = to_be_added_to_path + ";" + path;
      Environment.SetEnvironmentVariable("PATH", path);
      Environment.SetEnvironmentVariable("PYTHONPATH", path);

      // load the Python.NET dll, and start the (embedded) Python engine
      var dll = System.Reflection.Assembly.LoadFile(PYTHONNET_DLL_);
      var PythonEngine = dll.GetType("Python.Runtime.PythonEngine");

      // to be on the safe side, update the PythonPath of the local engine
      var PythonPathProperty = PythonEngine.GetProperty("PythonPath");
      var pythonPath = (string)PythonPathProperty.GetValue(null);
      pythonPath += ";" + KERAS_PORTED_CODE_;
      pythonPath += ";" + PYTHON_HOME_ + "\\Lib\\site-packages";
      PythonPathProperty.SetValue(null, pythonPath);

      // let's start executing some python code
      dynamic Py = dll.GetType("Python.Runtime.Py");
      dynamic GIL = Py.GetMethod("GIL").Invoke(null, null);

      // import "ch8-2.py"
      dynamic ch8_2 = Py.GetMethod("Import").Invoke(null, new object[] { "ch8-2" });

      // response = ch8_2.get_response("Hi From C#")
      var response = ch8_2.get_response("Hi From C#");
      Console.WriteLine("C# received: "+response);
      Console.WriteLine("\n\n");

      // let's call the run_cntk function from the Python script
      var img = ch8_2.run_cntk(image_path, vgg16_model_path);

      // convert the python numpy array to byte[]
      byte[] img_data = convert_uint8_numpy_array_to_byte_array(img);

      // display the image with OpenCV
      var mat = new OpenCvSharp.Mat(224, 224, OpenCvSharp.MatType.CV_8UC3, img_data, 3 * 224);
      OpenCvSharp.Cv2.ImShow("The Dream", mat);

      // Show also the original image
      OpenCvSharp.Cv2.ImShow("Original", OpenCvSharp.Cv2.ImRead(image_path).Resize(new OpenCvSharp.Size(224, 224)));

      OpenCvSharp.Cv2.WaitKey(0);

      GIL.Dispose();
    }

    static byte[] convert_uint8_numpy_array_to_byte_array(dynamic d) {
      // See also: https://github.com/yagweb/pythonnetLab/blob/master/pynetLab/Numpy.cs
      var d_len = (int)d.nbytes;
      var buffer = new byte[d_len];      
      var meta = d.GetAttr("__array_interface__");
      var address = new System.IntPtr((long)meta["data"][0]);
      System.Runtime.InteropServices.Marshal.Copy(address, buffer, 0, d_len);
      return buffer;
    }
  }
}
