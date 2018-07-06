using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;

namespace Ch_08_Text_Generation_With_LSTM {
  class Program {
    static void Main(string[] args) {
      new Program().run();
    }

    // Length of extracted character sequences
    static readonly int maxlen = 60;
    static readonly int alphabet_size = 59;

    class DataInfo {
      public string text;
      public char[] chars;
      public Dictionary<char, int> char_indices;
      public float[][] x;
      public float[] y;

      void init_text() {
        var url = "https://s3.amazonaws.com/text-datasets/nietzsche.txt";
        var text_path = Util.fullpathForDownloadedFile("text-datasets", "nietzsche.txt");
        if (System.IO.File.Exists(text_path) == false) {
          var success = FromStackOverflow.FileDownloader.DownloadFile(url, text_path, timeoutInMilliSec: 360000);
          if (!success) {
            Console.WriteLine("Could not download " + url);
            return;
          }
        }
        text = System.IO.File.ReadAllText(text_path, Encoding.UTF8).ToLowerInvariant();
        Console.WriteLine("Corpus length:" + text.Length);       
      }

      public DataInfo() {
        init_text();
        Console.WriteLine("Vectorization...");

        // We sample a new sequence every `step` characters
        var step = 3;

        // This holds our extracted sequences
        var sentences = new List<string>();

        // This holds the targets (the follow-up characters)
        var next_chars = new List<char>();

        for (int i = 0; i < text.Length - maxlen; i += step) {
          sentences.Add(text.Substring(i, maxlen));
          next_chars.Add(text[maxlen + i]);
        }
        Console.WriteLine("Number of sequences:" + sentences.Count);

        // List of unique characters in the corpus
        chars = text.Distinct().ToArray();
        Array.Sort(chars);
        Console.WriteLine("Unique characters:" + chars.Length);

        // Dictionary mapping unique characters to their index in `chars`
        char_indices = new Dictionary<char, int>();
        for (int i = 0; i < chars.Length; i++) {
          char_indices.Add(chars[i], i);
        }

        x = new float[sentences.Count][];
        y = new float[sentences.Count];
        for (int i = 0; i < sentences.Count; i++) {
          x[i] = new float[maxlen];
          var sentence = sentences[i];

          for (int t = 0; t < sentence.Length; t++) {
            x[i][t] = char_indices[sentence[t]];
          }
          y[i] = char_indices[next_chars[i]];
        }
      }
    }

    class TextGeneratingTrainingEngine: TrainingEngine {
      public CNTK.Function softmaxOutput;

      protected override void createVariables() {
        x = CNTK.Variable.InputVariable(new CNTK.NDShape(0), CNTK.DataType.Float, name: "x");
        var y_axis = new List<CNTK.Axis>() { CNTK.Axis.DefaultBatchAxis() };
        y = CNTK.Variable.InputVariable(new CNTK.NDShape(0), CNTK.DataType.Float, dynamicAxes: y_axis, name: "y");
      }

      protected override void createModel() {
        var model_path = "ch8-1_cntk.model";
        model = CNTK.Function.Load(model_path, computeDevice);
        var replacements = new CNTK.UnorderedMapVariableVariable() { { model.Placeholders()[0], x } };
        model.ReplacePlaceholders(replacements);
        softmaxOutput = CNTK.CNTKLib.Softmax(model.Output);
      }

      protected override CNTK.Function custom_loss_function() {
        var y_oneHot = CNTK.CNTKLib.OneHotOp(y, (uint)alphabet_size, false, new CNTK.Axis(0));
        var rtrn = CNTK.CNTKLib.CrossEntropyWithSoftmax(model.Output, y_oneHot);
        return rtrn;
      }
    }

    int sample(Random random, float[] preds, double temperature=1.0) {
      // step 1: apply temperature to predictions, and normalize them to create a probability distribution
      float sum = 0;
      for (int i=0; i<preds.Length; i++) {
        var p = (float)Math.Exp((Math.Log(Math.Max(preds[i], 1e-10)) / temperature));
        sum += p;
        preds[i] = p;
      }
      for (int i = 0; i < preds.Length; i++) { preds[i] /= sum; }

      // step 2: draw a random sample from this distribution
      var d = random.NextDouble();
      sum = 0;
      for (int i=0; i<preds.Length; i++) {
        sum += preds[i];
        if ( d<sum ) { return i; }
      }
      return preds.Length - 1;
    }

    void generate_text(TextGeneratingTrainingEngine engine, DataInfo di) {
      var random = new Random(2018);

      var start_index = (int)(random.NextDouble() * (di.text.Length - maxlen - 1));
      var seed_generated_text = di.text.Substring(start_index, maxlen).Replace('\n', ' ');
      Console.WriteLine($"\nSeed: {seed_generated_text}");

      var temperatures = new double[] { 0.2, 0.5, 1.0, 1.2 };
      foreach(var temperature in temperatures) {
        var generated_text = seed_generated_text;

        for (int i=0; i<400; i++) {
          var sampled = generated_text.Select(v => (float)(di.char_indices[v])).ToArray();
          var preds = engine.evaluate(new float[][] { sampled }, engine.softmaxOutput)[0].Take(di.chars.Length).ToArray();
          var next_index = sample(random, preds, temperature);
          var next_char = di.chars[next_index];
          if ( next_char=='\n' ) { next_char = ' '; }
          generated_text = generated_text.Substring(1) + next_char;
        }
        Console.WriteLine($"Randomly generated with temperature {temperature:F1}: {generated_text}");
      }
    }

    void run() {
      var di = new DataInfo();
      var engine = new TextGeneratingTrainingEngine() {
        num_epochs = 32,
        batch_size = 128,
        sequence_length = maxlen,
        lossFunctionType = TrainingEngine.LossFunctionType.Custom,
        accuracyFunctionType = TrainingEngine.AccuracyFunctionType.SameAsLoss,
        metricType = TrainingEngine.MetricType.Loss
      };
      engine.setData(di.x, di.y, null, null);
      engine.train();
      generate_text(engine, di);
    }
  }
}
