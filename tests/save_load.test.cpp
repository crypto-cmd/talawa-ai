#include <fstream>
#include <iostream>

#include "talawa/core/Matrix.hpp"
#include "talawa/core/Optimizer.hpp"
#include "talawa/neuralnetwork/NeuralNetwork.hpp"

using namespace talawa;
using namespace talawa::nn;
using namespace talawa::core;

int main() {
  // Build a simple network
  auto model =
      NeuralNetworkBuilder::create({1, 28, 28})
          .add(DenseLayerConfig{.neurons = 32, .act = Activation::RELU})
          .add(DenseLayerConfig{.neurons = 10, .act = Activation::SOFTMAX})
          .build();

  // Verify total parameter count: (784*32 + 1*32) + (32*10 + 1*10) = 25450
  int expected = 784 * 32 + 1 * 32 + 32 * 10 + 1 * 10;
  if (model->getTotalParameters() != expected) {
    std::cerr << "Total parameter count incorrect: expected " << expected
              << " got " << model->getTotalParameters() << std::endl;
    return 1;
  }

  // Verify clone preserves parameter count
  auto cloned = model->clone();
  if (!cloned) {
    std::cerr << "Clone returned null" << std::endl;
    return 1;
  }
  if (cloned->getTotalParameters() != model->getTotalParameters()) {
    std::cerr << "Cloned network parameter count mismatch" << std::endl;
    return 1;
  }

  const std::string fname = "test_network.nn";

  if (!model->saveToFile(fname)) {
    std::cerr << "Failed to save network to " << fname << std::endl;
    return 1;
  }

  auto loaded = NeuralNetwork::loadFromFile(fname);
  if (!loaded) {
    std::cerr << "Failed to load network from " << fname << std::endl;
    return 1;
  }

  if (model->layers.size() != loaded->layers.size()) {
    std::cerr << "Layer count mismatch after load" << std::endl;
    return 1;
  }

  for (size_t i = 0; i < model->layers.size(); ++i) {
    auto p_src = model->layers[i]->getParameters();
    auto p_dst = loaded->layers[i]->getParameters();
    if (p_src.size() != p_dst.size()) {
      std::cerr << "Parameter count mismatch in layer " << i << std::endl;
      return 1;
    }
    for (size_t j = 0; j < p_src.size(); ++j) {
      if (!(*p_src[j] == *p_dst[j])) {
        std::cerr << "Parameter mismatch at layer " << i << " param " << j
                  << std::endl;
        return 1;
      }
    }
  }

  // Verify total parameter count matches after load
  if (loaded->getTotalParameters() != model->getTotalParameters()) {
    std::cerr << "Total parameter count mismatch after load: expected "
              << model->getTotalParameters() << " got "
              << loaded->getTotalParameters() << std::endl;
    return 1;
  }

  // Verify YAML metadata exists and looks sensible
  std::ifstream yfin(fname + std::string(".yaml"));
  if (!yfin) {
    std::cerr << "Missing YAML metadata file: " << fname << ".yaml"
              << std::endl;
    return 1;
  }
  std::string content((std::istreambuf_iterator<char>(yfin)),
                      std::istreambuf_iterator<char>());
  if (content.find("layers:") == std::string::npos) {
    std::cerr << "YAML metadata missing 'layers:' section" << std::endl;
    return 1;
  }
  // Count occurrences of "- type:" to ensure metadata has an entry per layer
  size_t count = 0;
  size_t pos = 0;
  std::string key = "- type:";
  while ((pos = content.find(key, pos)) != std::string::npos) {
    ++count;
    pos += key.size();
  }
  if (count != model->layers.size()) {
    std::cerr << "YAML layer count mismatch: expected " << model->layers.size()
              << " got " << count << std::endl;
    return 1;
  }

  std::cout << "Save/load roundtrip OK" << std::endl;
  std::cout << "YAML metadata OK" << std::endl;
  return 0;
}
