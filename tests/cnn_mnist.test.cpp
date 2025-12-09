#include <iostream>

#include "talawa-ai/core/Optimizer.hpp"
#include "talawa-ai/neuralnetwork/NeuralNetwork.hpp"
#include "talawa-ai/utils/DataLoader.hpp"
#include "talawa-ai/utils/Timer.hpp"

using namespace talawa_ai;
using namespace talawa_ai::nn;
using namespace talawa_ai::core;

float get_accuracy(NeuralNetwork& model, const Matrix& X, const Matrix& Y) {
  int correct = 0;
  int total = X.rows;
  int batch_size = 32;  // Safe batch size

  // Loop through data in chunks
  for (int i = 0; i < total; i += batch_size) {
    int end = std::min(i + batch_size, total);
    int current_batch_size = end - i;

    // 1. Slice the batch
    Matrix batch_X = X.slice(i, end);

    // 2. Predict on just this small batch
    Matrix predictions = model.predict(batch_X);

    // 3. Compare with targets
    for (int k = 0; k < current_batch_size; ++k) {
      // Find Predicted Class
      int pred_class = 0;
      float max_val = -1e9f;
      for (int c = 0; c < 10; ++c) {
        if (predictions(k, c) > max_val) {
          max_val = predictions(k, c);
          pred_class = c;
        }
      }

      // Find True Class (from One-Hot Y)
      // Note: We need to look at the correct row in the global Y matrix
      int global_idx = i + k;
      int true_class = 0;
      for (int c = 0; c < 10; ++c) {
        if (Y(global_idx, c) == 1.0f) {
          true_class = c;
          break;
        }
      }

      if (pred_class == true_class) correct++;
    }
  }

  return (float)correct / total * 100.0f;
}
int main() {
  std::cout << "--- CNN Training on MNIST ---" << std::endl;

  // 1. Build CNN
  // Input shape is critical: {Depth, Height, Width}
  auto model = NeuralNetworkBuilder::create({1, 28, 28})
                   .add(Conv2DLayerConfig{
                       .filters = 32,
                       .kernel_size = 3,
                       .stride = 1,
                       .padding = 0,
                       .act = Activation::RELU,
                   })
                   .add(Pooling2DLayerConfig{
                       .type = PoolingType::MAX, .pool_size = 2, .stride = 2})

                   .add(DenseLayerConfig{
                       .neurons = 64,
                       .act = Activation::RELU,
                   })
                   .add(DenseLayerConfig{
                       .neurons = 10,
                       .act = Activation::SOFTMAX,
                   })
                   .setOptimizer(std::make_unique<Adam>(0.001f))
                   .setLossFunction(std::make_unique<loss::MeanSquaredError>())
                   .build();

  // 2. Load Data
  // Ensure mnist_train.csv is in your build folder
  try {
    auto data = utils::DataLoader::loadCSV("mnist_train.csv", 0, 10, 255.0f);

    int epochs = 5;
    int batch_size = 128;
    int num_samples = data.features.rows;
    int num_batches = num_samples / batch_size;

    // 3. Train Loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
      data.shuffle();
      float total_loss = 0.0f;

      MEASURE_SCOPE("Epoch " + std::to_string(epoch + 1));

      Matrix batch_X(batch_size, data.features.cols);
      Matrix batch_Y(batch_size, data.labels.cols);

      for (int i = 0; i < num_batches; ++i) {
        int start = i * batch_size;
        int end = start + batch_size;

        // Get Batch
        data.splice(start, end, batch_X, batch_Y);

        // Train
        float loss = model->train(batch_X, batch_Y);
        total_loss += loss;

        if (i % 100 == 0) {
          std::cout << "\rBatch " << i << "/" << num_batches
                    << " Loss: " << loss << std::flush;
        }
      }
      std::cout << "\nEpoch " << epoch + 1
                << " Avg Loss: " << (total_loss / num_batches) << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  std::cout << "\nTraining Finished. Loading Test Set..." << std::endl;

  // Load the separate test file
  auto test_data =
      talawa_ai::utils::DataLoader::loadCSV("mnist_test.csv", 0, 10, 255.0f);

  std::cout << "Evaluating on " << test_data.features.rows << " test images..."
            << std::endl;

  float accuracy = get_accuracy(*model, test_data.features, test_data.labels);

  std::cout << "===================================" << std::endl;
  std::cout << " FINAL ACCURACY: " << accuracy << "%" << std::endl;
  std::cout << "===================================" << std::endl;

  return 0;
}
