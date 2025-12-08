#include <iostream>
#include <vector>

#include "talawa-ai/core/Matrix.hpp"
#include "talawa-ai/core/Optimizer.hpp"
#include "talawa-ai/neuralnetwork/Conv2DLayer.hpp"
#include "talawa-ai/neuralnetwork/DenseLayer.hpp"
#include "talawa-ai/neuralnetwork/Loss.hpp"
#include "talawa-ai/neuralnetwork/NeuralNetwork.hpp"

using namespace talawa_ai;
using namespace talawa_ai::nn;
using namespace talawa_ai::core;

int main() {
  // 1. Define Network Architecture
  // ==========================================
  // We want a network for MNIST digits (28x28 images = 784 inputs)
  // Structure: Input(784) -> Dense(128, ReLU) -> Output(10, Softmax)
  auto model = NeuralNetworkBuilder::create({1, 28, 28})
                   .add(DenseLayerConfig{
                       .neurons = 128,
                       .init = Initializer::GLOROT_UNIFORM,
                       .act = Activation::RELU,
                   })
                   .add(DenseLayerConfig{
                       .neurons = 10,
                       .init = Initializer::GLOROT_UNIFORM,
                       .act = Activation::SOFTMAX,
                   })
                   .setOptimizer(std::make_unique<Adam>(0.001f))
                   .setLossFunction(
                       std::make_unique<loss::CategoricalCrossEntropyLoss>())
                   .build();
  for (const auto& layer : model->layers) {
    std::cout << layer->info() << std::endl;
  }
  std::cout << "Neural Network built with " << model->optimizer->getName()
            << " optimizer and " << model->loss_fn->getName()
            << " loss function.\n";
  // 3. Dummy Data (Batch Size = 4)
  // ==========================================
  // Inputs: 4 flattened images (4 rows, 784 cols)
  Matrix inputs = talawa_ai::core::Matrix::random(4, 784);

  // Targets: One-hot encoded (4 rows, 10 cols)
  Matrix targets(4, 10);
  targets.fill(0.0f);
  targets(0, 3) = 1.0f;  // Image 0 is class 3
  targets(1, 0) = 1.0f;  // Image 1 is class 0
  targets(2, 9) = 1.0f;  // Image 2 is class 9
  targets(3, 1) = 1.0f;  // Image 3 is class 1

  // 4. Training Loop
  // ==========================================
  int epochs = 50;

  std::cout << "Starting Training..." << std::endl;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    // Note: In a real app, you would loop over batches here.
    // We pass the optimizer and loss function into the updated train method.

    float loss = model->train(inputs, targets);

    if (epoch % 1 == 0) {
      std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << loss
                << std::endl;
    }
  }

  // 5. Prediction
  // ==========================================
  Matrix predictions = model->predict(inputs);
  std::cout << "\nPrediction for first sample (raw probabilities):"
            << std::endl;

  // Print probabilities for the first image in the batch
  for (int i = 0; i < 10; ++i) {
    std::cout << "Class " << i << ": " << predictions(0, i) << std::endl;
  }

  return 0;
}
