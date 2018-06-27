using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DeepLearningWithCNTK;

namespace Ch_06_Sequence_Processing_With_Convnets {
  class Program {
    static void Main(string[] args) {
      new Program().run();
    }

    class Conv1TrainingEngine : TrainingEngine {
      protected override void createVariables() {
        x = CNTK.Variable.InputVariable(new int[] { 500 }, CNTK.DataType.Float, name: "x");
        y = CNTK.Variable.InputVariable(new int[] { 1 }, CNTK.DataType.Float, name: "y");
      }

      protected override void createModel() {
        model = CNTK.CNTKLib.OneHotOp(x, 10000, true, new CNTK.Axis(0));
        model = Util.Embedding(model, 128, computeDevice);
        model = CNTK.CNTKLib.TransposeAxes(model, new CNTK.Axis(1), new CNTK.Axis(0));
        model = Util.Convolution1DWithReLU(model, 32, 7, computeDevice);
        model = CNTK.CNTKLib.Pooling(model, CNTK.PoolingType.Max, new int[] { 5 }, new int[] { 5 });
        model = Util.Convolution1DWithReLU(model, 32, 7, computeDevice);
        model = CNTK.CNTKLib.Pooling(model, CNTK.PoolingType.Max, CNTK.NDShape.Unknown(), new int[] { 1 });
        model = Util.Dense(model, 1, computeDevice);
        model = CNTK.CNTKLib.Sigmoid(model);
      }
    }

    void run() {
      var x_train = Util.load_binary_file("ch6-4_x_train_imdb.bin", 25000, 500);
      var y_train = Util.load_binary_file("ch6-4_y_train_imdb.bin", 25000);
      var x_test = Util.load_binary_file("ch6-4_x_test_imdb.bin", 25000, 500);
      var y_test = Util.load_binary_file("ch6-4_y_test_imdb.bin", 25000);

      var numTraining = (int)(x_train.Length * 0.8);
      var x_val = x_train.Skip(numTraining).ToArray();
      var y_val = y_train.Skip(numTraining).ToArray();
      x_train = x_train.Take(numTraining).ToArray();
      y_train = y_train.Take(numTraining).ToArray();
    
      var engine = new Conv1TrainingEngine() { num_epochs = 10, batch_size = 32, lr=0.0001 };
      engine.setData(x_train, y_train, x_val, y_val);
      engine.train();

    }
  }
}
