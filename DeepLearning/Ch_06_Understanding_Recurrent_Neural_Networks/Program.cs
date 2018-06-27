using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;

namespace Ch_06_Understanding_Recurrent_Neural_Networks {
  class Program {
    static void Main(string[] args) {
      new Program().run();
    }

    class TrainingEngine_with_LSTMs : TrainingEngine {
      protected override void createVariables() {
        x = CNTK.Variable.InputVariable(new int[] { 1 }, CNTK.DataType.Float, name: "x");
        var y_axis = new List<CNTK.Axis>() { CNTK.Axis.DefaultBatchAxis() };
        y = CNTK.Variable.InputVariable(new int[] { 1 }, CNTK.DataType.Float, dynamicAxes: y_axis, name: "y");
      }

      protected override void createModel() {
        bool use_saved_model = true;
        if (use_saved_model) {
          var model_path = "ch6-2.cntk.model";
          model = CNTK.Function.Load(model_path, computeDevice);
          var replacements = new CNTK.UnorderedMapVariableVariable() { { model.Placeholders()[0], x } };
          model.ReplacePlaceholders(replacements);
        }
        else {
          uint numClasses = 10000;
          int embedding_dim = 32;
          int hidden_units = 32;
          model = CNTK.CNTKLib.OneHotOp(x, numClass: numClasses, outputSparse: true, axis: new CNTK.Axis(0));
          model = Util.Embedding(model, embedding_dim, computeDevice);
          model = CNTK.CSTrainingExamples.LSTMSequenceClassifier.LSTM(model, hidden_units, hidden_units, computeDevice, "lstm");
          model = Util.Dense(model, 1, computeDevice);
          model = CNTK.CNTKLib.Sigmoid(model);
        }
      }
    }

    void run() {
      var x_train = Util.load_binary_file("x_train_imdb.bin", 25000, 500);
      var y_train = Util.load_binary_file("y_train_imdb.bin", 25000);
      var x_test = Util.load_binary_file("x_test_imdb.bin", 25000, 500);
      var y_test = Util.load_binary_file("y_test_imdb.bin", 25000);

      var engine = new TrainingEngine_with_LSTMs() { num_epochs = 10, batch_size = 128, sequence_length = 500 };
      engine.setData(x_train, y_train, x_test, y_test);
      engine.train();
    }
  }
}
