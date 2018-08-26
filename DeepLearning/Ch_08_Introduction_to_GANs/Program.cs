using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeepLearningWithCNTK;

namespace Ch_08_Introduction_to_GANs {
  class Program {
    static void Main(string[] args) {
      new Program().run();
    }

    static readonly int latent_dim = 32;
    static readonly int height = 32;
    static readonly int width = 32;
    static readonly int channels = 3;

    CNTK.Function create_generator() {
      var input_variable = CNTK.Variable.InputVariable(new int[] { latent_dim }, CNTK.DataType.Float, name: "generator_input");
      var x = Util.Dense(input_variable, 128 * 16 * 16, computeDevice);
      x = CNTK.CNTKLib.LeakyReLU(x);
      x = CNTK.CNTKLib.Reshape(x, new int[] { 16, 16, 128 });
      x = Util.Convolution2D(x, 256, new int[] { 5, 5 }, computeDevice, use_padding: true, activation: CNTK.CNTKLib.LeakyReLU);
      x = Util.ConvolutionTranspose(x, computeDevice,
        filter_shape: new int[] { 4, 4 },
        num_filters: 256,
        strides: new int[] { 2, 2 },
        output_shape: new int[] { 32, 32 },
        use_padding: true,
        activation: CNTK.CNTKLib.LeakyReLU);
      x = Util.Convolution2D(x, 256, new int[] { 5, 5 }, computeDevice, use_padding: true, activation: CNTK.CNTKLib.LeakyReLU);
      x = Util.Convolution2D(x, 256, new int[] { 5, 5 }, computeDevice, use_padding: true, activation: CNTK.CNTKLib.LeakyReLU);
      x = Util.Convolution2D(x, channels, new int[] { 7, 7 }, computeDevice, use_padding: true, activation: CNTK.CNTKLib.Tanh);
      return x;      
    }

    CNTK.Function create_discriminator() {
      var input_variable = CNTK.Variable.InputVariable(new int[] { width, height, channels }, CNTK.DataType.Float, name: "discriminator_input");
      var x = Util.Convolution2D(input_variable, 128, new int[] { 3, 3 }, computeDevice, activation: CNTK.CNTKLib.LeakyReLU);
      x = Util.Convolution2D(x, 128, new int[] { 4, 4 }, computeDevice, strides: new int[] { 2, 2 }, activation: CNTK.CNTKLib.LeakyReLU);
      x = Util.Convolution2D(x, 128, new int[] { 4, 4 }, computeDevice, strides: new int[] { 2, 2 }, activation: CNTK.CNTKLib.LeakyReLU);
      x = Util.Convolution2D(x, 128, new int[] { 4, 4 }, computeDevice, strides: new int[] { 2, 2 }, activation: CNTK.CNTKLib.LeakyReLU);
      x = CNTK.CNTKLib.Dropout(x, 0.4);
      x = Util.Dense(x, 1, computeDevice);
      x = CNTK.CNTKLib.Sigmoid(x);
      return x;
    }

    void create_gan() {
      label_var = CNTK.Variable.InputVariable(shape: new CNTK.NDShape(0), dataType: CNTK.DataType.Float, name: "label_var");
      generator = create_generator();
      discriminator = create_discriminator();
      gan = discriminator.Clone(CNTK.ParameterCloningMethod.Share, 
        replacements: new Dictionary<CNTK.Variable, CNTK.Variable>() { { discriminator.Arguments[0], generator } });

      discriminator_loss = CNTK.CNTKLib.BinaryCrossEntropy(discriminator, label_var);
      discriminator_learner = CNTK.CNTKLib.AdaDeltaLearner(
        parameters: new CNTK.ParameterVector((System.Collections.ICollection)discriminator.Parameters()),
        learningRateSchedule: new CNTK.TrainingParameterScheduleDouble(1));        
      discriminator_trainer = CNTK.CNTKLib.CreateTrainer(discriminator, discriminator_loss, discriminator_loss, new CNTK.LearnerVector() { discriminator_learner });

      gan_loss = CNTK.CNTKLib.BinaryCrossEntropy(gan, label_var);
      gan_learner = CNTK.CNTKLib.AdaDeltaLearner(
        parameters: new CNTK.ParameterVector((System.Collections.ICollection)generator.Parameters()),
        learningRateSchedule: new CNTK.TrainingParameterScheduleDouble(1));
      gan_trainer = CNTK.CNTKLib.CreateTrainer(gan, gan_loss, gan_loss, new CNTK.LearnerVector() { gan_learner });
    }

    void load_data() {
      var num_images = 5000;
      var data_filename = "x_channels_first_8_5.bin";
      var num_bytes_per_image = sizeof(float) * channels * width * height;
      System.Diagnostics.Debug.Assert(System.IO.File.Exists(data_filename));
      System.Diagnostics.Debug.Assert(new System.IO.FileInfo(data_filename).Length == num_images * num_bytes_per_image);
      x_train = Util.load_binary_file(data_filename, num_images, channels * width * height);
    }

    void train() {
      var random = new Random();
      var gaussianRandom = new FromStackOverflow.GaussianRandom(random);

      create_gan();
      load_data();
      var iterations = 100000;
      var batch_size = 20;
      var save_dir = "images";
      System.IO.Directory.CreateDirectory(save_dir);
      var start = 0;
      for (int step=0; step<iterations; step++) {
        // use the generator to generate fake images
        var random_latent_vectors = gaussianRandom.getFloatSamples(batch_size * latent_dim);
        var random_latent_vectors_nd = new CNTK.NDArrayView(new int[] { latent_dim, 1, batch_size }, random_latent_vectors, computeDevice);        
        var generator_inputs = new Dictionary<CNTK.Variable, CNTK.Value>() { { generator.Arguments[0], new CNTK.Value(random_latent_vectors_nd) } };
        var generator_outputs = new Dictionary<CNTK.Variable, CNTK.Value>() { { generator.Output, null } };
        generator.Evaluate(generator_inputs, generator_outputs, computeDevice);
        var generated_images = generator_outputs[generator.Output].GetDenseData<float>(generator.Output);

        // train the discriminator: the first half of the mini-batch are the fake images (marked with label='1')
        // whereas the second half are real images (marked with label='0')
        var combined_images = new float[2 * batch_size][];
        var labels = new float[2 * batch_size];
        start = Math.Min(start, x_train.Length - batch_size);
        for (int i=0; i<batch_size; i++) {
          combined_images[i] = generated_images[i].ToArray();
          labels[i] = (float)(1 + 0.05 * gaussianRandom.NextGaussian());

          combined_images[i + batch_size] = x_train[start + i];
          labels[i+batch_size] = (float)(0.05 * gaussianRandom.NextGaussian());
        }
        start += batch_size;
        if ( start>=x_train.Length ) { start = 0; }
        
        var combined_images_minibatch = Util.get_tensors(new int[] { width, height, channels }, combined_images, 0, combined_images.Length, computeDevice);
        var labels_minibatch = CNTK.Value.CreateBatch(new CNTK.NDShape(0), labels, computeDevice, true);
        var discriminator_minibatch = new Dictionary<CNTK.Variable, CNTK.Value>() {
          { discriminator.Arguments[0], combined_images_minibatch},
          {label_var, labels_minibatch }
        };
        discriminator_trainer.TrainMinibatch(discriminator_minibatch, true, computeDevice);
        var d_loss = discriminator_trainer.PreviousMinibatchLossAverage();

        // train the gan: the generator will try to fool the discriminator: we generate fake
        // images, but we label them as "real" (with label='0') 
        random_latent_vectors = gaussianRandom.getFloatSamples(batch_size * latent_dim);
        var misleading_targets = new float[batch_size];
        random_latent_vectors_nd = new CNTK.NDArrayView(new int[] { latent_dim, 1, batch_size }, random_latent_vectors, computeDevice);
        var gan_inputs = new Dictionary<CNTK.Variable, CNTK.Value>() {
          { gan.Arguments[0], new CNTK.Value(random_latent_vectors_nd) },
          { label_var, CNTK.Value.CreateBatch(new CNTK.NDShape(0), misleading_targets, computeDevice, true) }
        };
        gan_trainer.TrainMinibatch(gan_inputs, true, computeDevice);
        var g_loss = gan_trainer.PreviousMinibatchLossAverage();

        if ( step%100==0 ) {
          Console.WriteLine($"discriminator loss at step {step}: {d_loss:F3}");
          Console.WriteLine($"adversarial loss at step {step}: {g_loss:F3}");

          // Save one generated image
          var img = generated_images[0].ToArray();
          var img_bytes = Util.convert_from_channels_first(img, scaling: 255, invertOrder: true);
          var mat = new OpenCvSharp.Mat(height, width, OpenCvSharp.MatType.CV_8UC3, img_bytes, 3*width);
          var image_filename = $"generated_frog_{step}.png";
          var image_path = System.IO.Path.Combine(save_dir, image_filename);
          mat.SaveImage(image_path);
          mat.Dispose(); mat = null;

          // Save one real image for comparison
          img = x_train[Math.Max(start - batch_size, 0)];
          img_bytes = Util.convert_from_channels_first(img, scaling: 255, invertOrder: true);
          mat = new OpenCvSharp.Mat(height, width, OpenCvSharp.MatType.CV_8UC3, img_bytes, 3 * width);
          image_filename = $"real_frog_{step}.png";
          image_path = System.IO.Path.Combine(save_dir, image_filename);
          mat.SaveImage(image_path);
          mat.Dispose(); mat = null;
        }
      }
    }

    //void debug_data() {
    //  load_data();
    //  var img = x_train[0];
    //  var img_bytes = Util.convert_from_channels_first(img, scaling: 255, invertOrder: true);
    //  var mat = new OpenCvSharp.Mat(height, width, OpenCvSharp.MatType.CV_8UC3, img_bytes, 3 * width);
    //  OpenCvSharp.Cv2.ImShow("the frog", mat);
    //  OpenCvSharp.Cv2.WaitKey(0);
    //}

    void run() {
      train();
    }

    Program() {
      computeDevice = Util.get_compute_device();
      Console.Title = "Introduction to Generative Adversarial Networks";
    }

    float[][] x_train;
    CNTK.DeviceDescriptor computeDevice;
    CNTK.Variable label_var;
    CNTK.Function generator;
    CNTK.Function discriminator;
    CNTK.Function gan;
    CNTK.Function discriminator_loss;
    CNTK.Trainer discriminator_trainer;
    CNTK.Learner discriminator_learner;
    CNTK.Function gan_loss;
    CNTK.Trainer gan_trainer;
    CNTK.Learner gan_learner;
  }
}
