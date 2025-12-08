#include "talawa-ai/core/Activation.hpp"
#include "talawa-ai/core/Optimizer.hpp"
#include "talawa-ai/neuralnetwork/Loss.hpp"
#include "talawa-ai/neuralnetwork/NeuralNetwork.hpp"
#include "talawa-ai/utils/DataLoader.hpp"
#include "talawa-ai/utils/Timer.hpp"

using namespace talawa_ai;
using namespace talawa_ai::nn;
using namespace talawa_ai::core;
using namespace talawa_ai::utils;

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
                       .neurons = 128,
                       .init = Initializer::GLOROT_UNIFORM,
                       .act = Activation::TANH,
                   })
                   .add(DenseLayerConfig{
                       .neurons = 10,
                       .init = Initializer::GLOROT_UNIFORM,
                       .act = Activation::SOFTMAX,
                   })
                   .setOptimizer(std::make_unique<SGD>(0.05f))
                   .setLossFunction(std::make_unique<loss::MeanSquaredError>())
                   .build();

  // 1. Load Data
  // MNIST format: Label is column 0. There are 10 classes.
  // Pixel values are 0-255, so we scale by 255.0f to get 0-1 range.
  try {
    auto data =
        talawa_ai::utils::DataLoader::loadCSV("mnist_train.csv", 0, 10, 255.0f);
    int epochs = 25;
    int batch_size = 16;
    int num_samples = data.features.rows;
    std::cout << "Starting training on " << num_samples << " samples for "
              << epochs << " epochs." << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
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
                << " - Avg Loss: " << avg_loss
                << " - Batch Size: " << batch_size << std::endl;
    }

    std::cout << "\nTraining Finished. Loading Test Set..." << std::endl;

    // Load the separate test file
    auto test_data =
        talawa_ai::utils::DataLoader::loadCSV("mnist_test.csv", 0, 10, 255.0f);

    std::cout << "Evaluating on " << test_data.features.rows
              << " test images..." << std::endl;

    float accuracy = get_accuracy(*model, test_data.features, test_data.labels);

    std::cout << "===================================" << std::endl;
    std::cout << " FINAL ACCURACY: " << accuracy << "%" << std::endl;
    std::cout << "===================================" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Skipping real data training: " << e.what() << std::endl;
    std::cerr << "Make sure 'mnist_train.csv' is in your build directory."
              << std::endl;

    // Fallback to random data test...
  } catch (const std::exception& e) {
    std::cerr << "Error loading data: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}  // 91.21% ACCURACY ON MNIST TEST SET WITH 5 EPOCHS
