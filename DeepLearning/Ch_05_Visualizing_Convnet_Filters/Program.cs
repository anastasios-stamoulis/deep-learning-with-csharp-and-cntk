using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;

namespace Ch_05_Visualizing_Convnet_Filters {

  class Program {
    [STAThread]
    static void Main(string[] args) {
      new Program().run();
    }

    void deprocess_image(float[] x) {
      var meanValue = x.Average();
      var sumSquares = 0.0;
      for (int i = 0; i < x.Length; i++) {
        x[i] -= meanValue;
        sumSquares += x[i] * x[i];
      }
      var std = (float)(Math.Sqrt(sumSquares / x.Length) + 1e-5);
      for (int i = 0; i < x.Length; i++) {
        x[i] /= std;
        x[i] *= 0.1f;
        x[i] += 0.5f;
        x[i] = Math.Min(Math.Max(0, x[i]), 1);
        x[i] *= 255;
      }
    }

    void debugging() {
      var a = CPPUtil.version();
      Console.WriteLine("the value:" + a);
    }

    float[] compute_image(string caffeModelFilePath, int filterIndex) {
      var image = new float[150 * 150 * 3];
      CPPUtil.compute_image(image, caffeModelFilePath, filterIndex);
      deprocess_image(image);
      return image;
    }

    void run() {
      var caffeModelFilePath = VGG16.download_model_if_needed();

      var N = 4;
      var images = new float[150 * 150 * 3 * N];
      for (int i=0; i<N; i++) {
        var image = compute_image(caffeModelFilePath, i);
        Array.Copy(image, 0, images, i * image.Length, image.Length);
      }
                 
      var app = new System.Windows.Application();
      var window = new PlotWindowBitMap("Filters", images, 150, 150, 3);
      app.Run(window);
    }
  }
}
