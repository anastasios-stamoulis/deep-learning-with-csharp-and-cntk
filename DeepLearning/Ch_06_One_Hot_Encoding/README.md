# One Hot Encoding

The original Python code can be found in [ch6-1_a.py](../../Python/ch6-1_a.py)

This section is effectively a pre-processing step before we feed data into a network. 

It is rather straightforward, but for completeness, let's quickly  go over the port from Python to C#. 

In Python, a toy example of word level-one hot encoding is: 

```
import numpy as np

# This is our initial data; one entry per "sample"
# (in this toy example, a "sample" is just a sentence, but
# it could be an entire document).
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# First, build an index of all tokens in the data.
token_index = {}
for sample in samples:
    # We simply tokenize the samples via the `split` method.
    # in real life, we would also strip punctuation and special characters
    # from the samples.
    for word in sample.split():
        if word not in token_index:
            # Assign a unique index to each unique word
            token_index[word] = len(token_index) + 1
            # Note that we don't attribute index 0 to anything.

# Next, we vectorize our samples.
# We will only consider the first `max_length` words in each sample.
max_length = 10

# This is where we store our results:
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.
```

In C#, we have the method 

```
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
```

Keras also contains the helper class `Tokenizer`, which can be used as follows: 

```
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# We create a tokenizer, configured to only take
# into account the top-1000 most common words
tokenizer = Tokenizer(num_words=1000)
# This builds the word index
tokenizer.fit_on_texts(samples)

# This turns strings into lists of integer indices.
sequences = tokenizer.texts_to_sequences(samples)

# You could also directly get the one-hot binary representations.
# Note that other vectorization modes than one-hot encoding are supported!
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# This is how you can recover the word index that was computed
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
```

In [Util.cs](../Util.cs) we have a simple port of the `Tokenizer` class to C#, which can be used as follows: 

```
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
```