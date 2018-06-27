using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeepLearningWithCNTK;
using FromKeras;

namespace Ch_06_One_Hot_Encoding {
  class Program {
    static void Main(string[] args) {
      new Program().run();
    }

    void test_tokenizer() {
      Console.WriteLine("Test Tokenizer");
      var samples = new string[] { "The cat sat on the mat.", "The dog ate my homework." };
      var tokenizer = new Tokenizer(num_words: 1000);
      tokenizer.fit_on_texts(samples);
      var sequences = tokenizer.texts_to_sequences(samples);

      Console.WriteLine("\n\n*** Test Tokenizer ***\n");
      for (int i=0; i<samples.Length; i++) {
        Console.WriteLine($"{samples[i]} : {{ {string.Join(", ", sequences[i])} }}");
      }
    }

    void character_level_encoding() {
      var samples = new string[] { "The cat sat on the mat.", "The dog ate my homework." };
      var token_index = new Dictionary<char, int>();
      var printable = Tokenizer.python_string_printable();
      for (int i=0; i<printable.Length; i++) {
        token_index[printable[i]] = i + 1;
      }

      Console.WriteLine("\n\n*** Character Level Encoding ***\n");
      var max_length = 50;
      var results = new int[samples.Length, max_length, token_index.Values.Max() + 1];
      for (int i=0; i<samples.Length; i++) {
        var sample = samples[i].Substring(0, Math.Min(max_length, samples[i].Length));
        for (int j=0; j<sample.Length; j++) {
          var index = token_index[sample[j]];
          results[i, j, index] = 1;
          Console.WriteLine($"results[{i}, {j}, {index}] = 1");
        }
      }
    }

    void word_level_encoding() {
      var samples = new string[] { "The cat sat on the mat.", "The dog ate my homework." };
      var token_index = new Dictionary<string, int>();
      foreach (var sample in samples) {
        foreach (var word in sample.Split(' ')) {
          if ( token_index.ContainsKey(word)==false) {
            token_index.Add(word, token_index.Keys.Count + 1);
          }
        }
      }

      Console.WriteLine("\n\n*** Word Level Encoding ***\n");
      var max_length = 10;
      var results = new int[samples.Length, max_length, token_index.Values.Max() + 1];
      for (int i=0; i<samples.Length; i++) {
        var sample = samples[i];
        var words = sample.Split(' ');
        for (int j=0; j<words.Length; j++) {
          var word = words[j];
          var index = token_index[word];
          results[i, j, index] = 1;
          Console.WriteLine($"results[{i}, {j}, {index}] = 1");
        }
      }
    }

    void run() {
      character_level_encoding();
      word_level_encoding();
      test_tokenizer();
    }
  }
}
