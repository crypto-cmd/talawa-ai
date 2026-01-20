#include "talawa/core/Activation.hpp"
#include "talawa/core/Optimizer.hpp"
#include "talawa/neuralnetwork/Loss.hpp"
#include "talawa/neuralnetwork/NeuralNetwork.hpp"
#include "talawa/utils/DataLoader.hpp"
#include "talawa/utils/Timer.hpp"

using namespace talawa;
using namespace talawa::nn;
using namespace talawa::core;
using namespace talawa::utils;

float get_accuracy(NeuralNetwork& model, const Matrix& X, const Matrix& Y) {
  // 1. Get all predictions at once
  Matrix predictions = model.predict(X);

  int correct = 0;
  int total = X.rows;

  // 2. Compare Prediction vs Target for every row
  for (int i = 0; i < total; ++i) {
    // Find Predicted Class (ArgMax)
    int pred_class = 0;
    float max_val = -1e9f;
    for (int j = 0; j < 10; ++j) {
      if (predictions(i, j) > max_val) {
        max_val = predictions(i, j);
        pred_class = j;
      }
    }

    // Find True Class (ArgMax of One-Hot encoded label)
    int true_class = 0;
    for (int j = 0; j < 10; ++j) {
      if (Y(i, j) == 1.0f) {
        true_class = j;
        break;
      }
    }

    if (pred_class == true_class) correct++;
  }

  return (float)correct / total * 100.0f;
}
int main() {
  auto model = NeuralNetworkBuilder::create({1, 28, 28})
                   .add(DenseLayerConfig{
                       .neurons = 12,
                       .act = Activation::TANH,
                       .init = Initializer::GLOROT_UNIFORM,
                   })
                   .add(DenseLayerConfig{
                       .neurons = 10,
                       .act = Activation::SOFTMAX,
                       .init = Initializer::GLOROT_UNIFORM,
                   })
                   .setOptimizer(std::make_unique<Adam>())
                   .setLossFunction(std::make_unique<loss::MeanSquaredError>())
                   .build();

  // 1. Load Data
  // MNIST format: Label is column 0. There are 10 classes.
  // Pixel values are 0-255, so we scale by 255.0f to get 0-1 range.
  auto data = talawa::utils::DataLoader::loadCSV("./build/mnist_train.csv", 0,
                                                 10, 255.0f);
  int epochs = 25;
  int batch_size = 64;
  int num_samples = data.features.rows;
  std::cout << "Starting training on " << num_samples << " samples for "
            << epochs << " epochs." << std::endl;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    if (epoch == 15) {
      // Learning rate decay
      model->set_learning_rate(model->get_learning_rate() * 0.1f);
      std::cout << "Learning rate decayed to " << model->get_learning_rate()
                << " at epoch " << epoch + 1 << std::endl;
      batch_size = 128;  // Increase batch size for stability
    }
    data.shuffle();
    MEASURE_SCOPE("Epoch " + std::to_string(epoch + 1));

    float total_loss = 0.0f;
    int batches = 0;

    int numBatches = num_samples / batch_size;
    Matrix batch_X(batch_size, data.features.cols);
    Matrix batch_Y(batch_size, data.labels.cols);
    // Loop over batches
    for (int i = 0; i < numBatches; i++) {
      int start = i * batch_size;
      int end = std::min(start + batch_size, num_samples);
      std::cout << "\r Processing Batch " << (i + 1) << "/" << numBatches
                << std::flush;
      // 1. Get Batch
      data.splice(start, end, batch_X, batch_Y);

      // 2. Train on Batch
      float loss = model->train(batch_X, batch_Y);

      total_loss += loss;
      batches++;
    }

    float avg_loss = total_loss / batches;
    std::cout << "Epoch " << epoch + 1 << "/" << epochs
              << " - Avg Loss: " << avg_loss << " - Batch Size: " << batch_size
              << std::endl;
  }

  std::cout << "\nTraining Finished. Loading Test Set..." << std::endl;

  // Load the separate test file
  auto test_data = talawa::utils::DataLoader::loadCSV("./build/mnist_test.csv",
                                                      0, 10, 255.0f);

  std::cout << "Evaluating on " << test_data.features.rows << " test images..."
            << std::endl;

  float accuracy = get_accuracy(*model, test_data.features, test_data.labels);

  std::cout << "===================================" << std::endl;
  std::cout << " FINAL ACCURACY: " << accuracy << "%" << std::endl;
  std::cout << "===================================" << std::endl;

  // Print the neural network structure
  // model->print();

}  // 91.21% ACCURACY ON MNIST TEST SET WITH 5 EPOCHS
