using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;

namespace Ch_05_Class_Activation_Heatmaps {
  class Program {
    readonly string imageName = "creative_commons_elephant.jpg";

    [STAThread]
    static void Main(string[] args) {
      new Program().run();
    }

    void run() {
      Console.Title = "Ch_05_Class_Activation_Heatmaps";
      var text = System.IO.File.ReadAllText("imagenet_class_index.json");
      var imagenetInfo = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<int, List<string>>>(text);

      var imagePath = System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), imageName);
      var pathToVGG16model = VGG16.download_model_if_needed();
      var image = new float[224 * 224 * 3];
      CPPUtil.load_image(imagePath, image);

      int num_classes = 1000;
      var predictions = new float[num_classes];
      CPPUtil.evaluate_vgg16(pathToVGG16model, imagePath, predictions, num_classes);

      var indices = Enumerable.Range(0, num_classes).ToArray<int>();
      var floatComparer = Comparer<float>.Default;
      Array.Sort(indices, (a, b) => floatComparer.Compare(predictions[b], predictions[a]));

      Console.WriteLine("Predictions:");
      for (int i=0; i<3; i++) {
        var imagenetClass = imagenetInfo[indices[i]];
        var imagenetClassName = imagenetClass[1];
        var predicted_score = predictions[indices[i]];
        Console.WriteLine($"\t({imagenetClassName} -> {predicted_score:f3})");
      }

      var imageWithHeatMap = new float[image.Length];
      CPPUtil.visualize_heatmap(pathToVGG16model, imagePath, "conv5_3", 386, imageWithHeatMap);

      var app = new System.Windows.Application();
      var window = new PlotWindowBitMap("Original Image", image, 224, 224, 3);
      window.Show();
      var windowHeat = new PlotWindowBitMap("Class Activation Heatmap [386]", imageWithHeatMap, 224, 224, 3);
      windowHeat.Show();
      app.Run();
    }
  }
}
