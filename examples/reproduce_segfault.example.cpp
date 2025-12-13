#include <iostream>
#include <memory>

#include "talawa-ai/core/Optimizer.hpp"
#include "talawa-ai/neuralnetwork/DenseLayer.hpp"
#include "talawa-ai/neuralnetwork/Loss.hpp"
#include "talawa-ai/neuralnetwork/NeuralNetwork.hpp"

using namespace talawa_ai::core;
using namespace talawa_ai::nn;

int main() {
  std::cout << "Starting reproduction..." << std::endl;

  Shape input_shape{2, 3, 3}; // Match TicTacToe shape
  int hidden_neurons = 64;
  int action_size = 9;

  std::cout << "Building main_net..." << std::endl;
  auto main_net =
      NeuralNetworkBuilder::create(input_shape)
          .add(DenseLayerConfig{hidden_neurons, Initializer::HE_NORMAL,
                                Activation::RELU})
          .add(DenseLayerConfig{hidden_neurons, Initializer::HE_NORMAL,
                                Activation::RELU})
          .add(DenseLayerConfig{action_size, Initializer::GLOROT_UNIFORM,
                                Activation::LINEAR})
          .setOptimizer(std::make_unique<Adam>(0.001f))
          .setLossFunction(std::make_unique<loss::MeanSquaredError>())
          .build();

  std::cout << "main_net built. Cloning..." << std::endl;
  auto target_net = main_net->clone();
  std::cout << "Initial clone successful." << std::endl;

  std::cout << "Simulating training loop..." << std::endl;
  Matrix input = Matrix::random(1, 18);
  Matrix target = Matrix::random(1, 9);

  for (int i = 0; i < 200; ++i) {
      main_net->train(input, target);
      if (i % 10 == 0) {
           std::cout << "Updating target net at step " << i << std::endl;
           target_net = main_net->clone();
      }
  }
  std::cout << "Finished loop." << std::endl;

  return 0;
}
