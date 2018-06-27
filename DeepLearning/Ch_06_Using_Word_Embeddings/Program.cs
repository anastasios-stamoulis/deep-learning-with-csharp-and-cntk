using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;
using FromKeras;

namespace Ch_06_Using_Word_Embeddings {

  class Program {
    class Constants {
      public static readonly int maxlen = 100;
      public static readonly int max_words = 10000;
      public static readonly int training_samples = 200;      // We will be training on 200 samples
      public static readonly int validation_samples = 10000;  // We will be validating on 10000 samples
      public static readonly int embedding_dim = 100;
      public static readonly string imdb_dir = "C:\\Users\\anastasios\\Downloads\\aclImdb";
    }

    static void Main(string[] args) {
      //new Program().learning_word_embeddings_with_the_embedding_layer();
      new Program().use_glove_word_embeddings(preload_weights: false);
    }

    float[][] compute_embedding_matrix(FromKeras.Tokenizer tokenizer) {
      var embedding_matrix = new float[Constants.max_words][];
      var embeddings_index = preprocess_embeddings();
      foreach (var entry in tokenizer.word_index) {
        var word = entry.Key;
        var i = entry.Value;
        if (i>=Constants.max_words) { continue; }
        float[] embedding_vector;
        embeddings_index.TryGetValue(word, out embedding_vector);
        if (embedding_vector == null) {
          // Words not found in embedding index will be all-zeros.
          embedding_vector = new float[Constants.embedding_dim];
        }
        else {
          System.Diagnostics.Debug.Assert(embedding_vector.Length == Constants.embedding_dim);
        }
        embedding_matrix[i] = embedding_vector;
      }
      for (int i=0; i<embedding_matrix.Length; i++) {
        if ( embedding_matrix[i]!=null ) { continue; }
        embedding_matrix[i] = new float[Constants.embedding_dim];
      }
      return embedding_matrix;
    }

    Dictionary<string, float[]> preprocess_embeddings() {
      var glove_dir = "C:\\Users\\anastasios\\Downloads\\glove.6B";
      var embeddings_index = new Dictionary<string, float[]>();
      var glove_path = System.IO.Path.Combine(glove_dir, "glove.6B.100d.txt");
      Console.WriteLine($"Processing {glove_path}");
      foreach(var line in System.IO.File.ReadLines(glove_path, Encoding.UTF8)) {
        var values = line.Split(' ');
        var word = values[0];
        var coefs = values.Skip(1).Select(v => Single.Parse(v)).ToArray();
        System.Diagnostics.Debug.Assert(coefs.Length == Constants.embedding_dim);
        embeddings_index[word] = coefs;        
      }
      Console.WriteLine($"Found {embeddings_index.Keys.Count:n0} word vectors.");
      return embeddings_index;
    }

    class GloVeTrainingEngine : TrainingEngine {
      public float[][] embedding_weights = null;
      protected override void createVariables() {
        x = CNTK.Variable.InputVariable(new int[] { x_train[0].Length }, CNTK.DataType.Float);
        y = CNTK.Variable.InputVariable(new int[] { 1 }, CNTK.DataType.Float);
      }

      protected override void createModel() {       
        model = CNTK.CNTKLib.OneHotOp(x, numClass: (uint)Constants.max_words, outputSparse: true, axis: new CNTK.Axis(0));
        model = Util.Embedding(model, Constants.embedding_dim, computeDevice, weights: embedding_weights);
        model = Util.Dense(model, 32, computeDevice);
        model = CNTK.CNTKLib.ReLU(model);
        model = Util.Dense(model, 1, computeDevice);
        model = CNTK.CNTKLib.Sigmoid(model);
      }
    }

    void use_glove_word_embeddings(bool preload_weights=true) {
      float[][] x_train, x_val;
      float[] y_train, y_val;
      Tokenizer tokenizer;
      from_raw_text_to_word_embeddings(out tokenizer, out x_train, out y_train, out x_val, out y_val);
      
      var engine = new GloVeTrainingEngine() { num_epochs = 10, batch_size = 32, lr=0.001 };
      engine.embedding_weights = preload_weights ? compute_embedding_matrix(tokenizer) : null;

      engine.setData(x_train, y_train, x_val, y_val);
      engine.train();
    }

    void from_raw_text_to_word_embeddings(out Tokenizer tokenizer, out float[][] x_train, out float[] y_train, out float[][] x_val, out float[] y_val) {
      List<string> texts;
      List<float> labels;
      tokenize_alImdb(out tokenizer, out texts, out labels);
      var sequences = tokenizer.texts_to_sequences(texts.ToArray());
      var word_index = tokenizer.word_index;
      Console.WriteLine($"Found {word_index.Keys.Count:n0} unique tokens.");

      var data_array = Preprocessing.pad_sequences(sequences, maxlen: Constants.maxlen);
      var labels_array = labels.ToArray();
      Util.shuffle(data_array, labels_array);

      x_train = data_array.Take(Constants.training_samples).ToArray();
      y_train = labels_array.Take(Constants.training_samples).ToArray();
      x_val = data_array.Skip(Constants.training_samples).Take(Constants.validation_samples).ToArray();
      y_val = labels_array.Skip(Constants.training_samples).Take(Constants.validation_samples).ToArray();
    }

    void load_text_labels(string path, List<string> texts, List<float> labels) {
      var label_types = new string[] { "neg", "pos" };
      foreach(var label_type in label_types) {
        var dir_name = System.IO.Path.Combine(path, label_type);
        foreach(var fname in System.IO.Directory.GetFiles(dir_name)) {
          if (fname.EndsWith(".txt")) {
            texts.Add(System.IO.File.ReadAllText(System.IO.Path.Combine(dir_name, fname), Encoding.UTF8));
            labels.Add((label_type == "neg") ? 0 : 1);
          }
        }
      }
    }

    void tokenize_alImdb(out Tokenizer tokenizer, out List<string> texts, out List<float> labels) {
      texts = new List<string>();
      labels = new List<float>();
      var train_dir = System.IO.Path.Combine(Constants.imdb_dir, "train");
      load_text_labels(train_dir, texts, labels);
      tokenizer = new Tokenizer(num_words: Constants.max_words);
      tokenizer.fit_on_texts(texts.ToArray());
    }

    class LearningWordEmbeddings: TrainingEngine {
      protected override void createVariables() {
        x = CNTK.Variable.InputVariable(new int[] { x_train[0].Length }, CNTK.DataType.Float);
        y = CNTK.Variable.InputVariable(new int[] { 1 }, CNTK.DataType.Float);
      }

      protected override void createModel() {
        uint numClasses = 10000;
        model = CNTK.CNTKLib.OneHotOp(x, numClass: numClasses, outputSparse: true, axis: new CNTK.Axis(0));
        model = Util.Embedding(model, 8, computeDevice);
        model = Util.Dense(model, 1, computeDevice);
        model = CNTK.CNTKLib.Sigmoid(model);
      }
    }

    void learning_word_embeddings_with_the_embedding_layer() {
      var x_train = Util.load_binary_file("x_train_imdb.bin", 25000, 20);
      var y_train = Util.load_binary_file("y_train_imdb.bin", 25000);
      var x_test = Util.load_binary_file("x_test_imdb.bin", 25000, 20);
      var y_test = Util.load_binary_file("y_test_imdb.bin", 25000);
      
      var engine = new LearningWordEmbeddings() { num_epochs = 20, batch_size = 32, lr=0.01 };
      engine.setData(x_train, y_train, x_test, y_test);
      engine.train();
    }
  }
}
