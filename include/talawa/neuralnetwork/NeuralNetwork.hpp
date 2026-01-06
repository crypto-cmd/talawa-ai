#pragma once
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "talawa/core/Matrix.hpp"
#include "talawa/core/Optimizer.hpp"
#include "talawa/neuralnetwork/Conv2DLayer.hpp"
#include "talawa/neuralnetwork/DenseLayer.hpp"
#include "talawa/neuralnetwork/Layer.hpp"
#include "talawa/neuralnetwork/Loss.hpp"
#include "talawa/neuralnetwork/Pooling2DLayer.hpp"
namespace talawa::nn {

using LayerConfigVariant =
    std::variant<DenseLayerConfig, Conv2DLayerConfig, Pooling2DLayerConfig>;
class NeuralNetwork;  // Forward declaration
class NeuralNetworkBuilder {
 public:
  static NeuralNetworkBuilder create(const Shape& shape);
  NeuralNetworkBuilder& add(LayerConfigVariant config);
  NeuralNetworkBuilder& setOptimizer(std::unique_ptr<core::Optimizer> opt);
  NeuralNetworkBuilder& setLossFunction(std::unique_ptr<loss::Loss> loss);
  std::unique_ptr<NeuralNetwork> build();

  NeuralNetworkBuilder(const NeuralNetworkBuilder& other) {
    // deep copy
    this->input_shape = other.input_shape;
    this->configs = other.configs;
    // Note: Optimizer and Loss are not copied to avoid shared ownership
    this->optimizer = nullptr;
    this->loss_fn = nullptr;
  }

 private:
  Shape input_shape;
  std::vector<LayerConfigVariant> configs;
  std::unique_ptr<core::Optimizer> optimizer =
      std::make_unique<core::SGD>(0.01f);
  std::unique_ptr<loss::Loss> loss_fn =
      std::make_unique<loss::MeanSquaredError>();
  NeuralNetworkBuilder() = default;  // Private constructor
};
class NeuralNetwork {
  friend class NeuralNetworkBuilder;

 private:
  int _totalParameters = 0;

 public:
  NeuralNetwork(const NeuralNetwork& other) {
    // --- DEBUG CHECK ---
    // Ensure the source network has a valid input shape (builder should set it)
    if (other.input_shape.depth == 0 && other.input_shape.height == 0) {
      std::cerr << "\n[FATAL ERROR] NeuralNetwork::copy() called but "
                   "source network input_shape is ZERO."
                << std::endl;
      std::cerr << "This means 'network->input_shape = this->input_shape;' is "
                   "MISSING from Builder::build()."
                << std::endl;
      std::exit(1);  // Force exit so you see the message
    }
    // -------------------

    // 1. Create a fresh builder with the same input shape
    auto builder = NeuralNetworkBuilder::create(other.get_input_shape());

    // 2. Add all the same layer configurations
    for (const auto& config : other.configs) {
      builder.add(config);
    }

    builder.setLossFunction(std::make_unique<loss::MeanSquaredError>());
    builder.setOptimizer(std::make_unique<core::SGD>(0.01f));

    // 3. Build the new network (Randomly initialized weights)
    // Note: Target networks don't need optimizers, so default SGD is fine.
    auto copy = builder.build();

    // 4. Copy the Weights (The most important part)
    // Iterate over the source network's layers ("other") and copy their
    // parameter matrices into the freshly-built target ("copy").
    for (size_t i = 0; i < other.layers.size(); ++i) {
      auto src_params = other.layers[i]->getParameters();
      auto dst_params = copy->layers[i]->getParameters();

      if (src_params.size() != dst_params.size()) {
        throw std::runtime_error(
            "Clone failed: Layer parameter count mismatch.");
      }

      for (size_t j = 0; j < src_params.size(); ++j) {
        *dst_params[j] = *src_params[j];
      }
    }

    // Check if every layer has the same weights (Debugging)
    for (size_t i = 0; i < other.layers.size(); ++i) {
      auto src_params = other.layers[i]->getParameters();
      auto dst_params = copy->layers[i]->getParameters();

      for (size_t j = 0; j < src_params.size(); ++j) {
        if (!(*dst_params[j] == *src_params[j])) {
          throw std::runtime_error(
              "Clone verification failed: Weights do not match after copy.");
        }
      }
    }

    // Finally, assign the copied network to this
    this->layers = std::move(copy->layers);
    this->optimizer = std::move(copy->optimizer);
    this->loss_fn = std::move(copy->loss_fn);
    this->input_shape = copy->input_shape;
    this->configs = copy->configs;
    // Preserve computed parameter count from the newly-built copy
    this->_totalParameters = copy->_totalParameters;
  }

  // Copy-assignment operator using copy-and-swap idiom
  NeuralNetwork& operator=(const NeuralNetwork& other) {
    if (this == &other) return *this;
    NeuralNetwork tmp(other);  // uses existing copy ctor
    std::swap(layers, tmp.layers);
    std::swap(optimizer, tmp.optimizer);
    std::swap(loss_fn, tmp.loss_fn);
    std::swap(input_shape, tmp.input_shape);
    std::swap(configs, tmp.configs);
    m_optimized_act = tmp.m_optimized_act;
    return *this;
  }
  core::Matrix predict(const core::Matrix& input) const;
  float train(const core::Matrix& input, const core::Matrix& target);
  // Persistence helpers
  bool saveToFile(const std::string& filename) const;
  static std::unique_ptr<NeuralNetwork> loadFromFile(
      const std::string& filename);

  std::vector<std::unique_ptr<ILayer>> layers;

  std::unique_ptr<core::Optimizer> optimizer;
  std::unique_ptr<loss::Loss> loss_fn;

  std::unique_ptr<NeuralNetwork> clone() const;

  int getTotalParameters() const { return _totalParameters; }

 private:
  Shape get_input_shape() const;
  void recalcTotalParameters();  // Recompute _totalParameters from layers
  NeuralNetwork() = default;     // Private constructor (use Builder)
  std::vector<LayerConfigVariant> configs;

  // If set, this activation is applied *after* the last layer during
  // prediction. This allows us to train on Linear (stable) but output Softmax
  // (user-friendly).
  std::optional<core::Activation::Type> m_optimized_act;
  Shape input_shape = {0, 0, 0};

  bool save(std::ostream& out) const;
  static std::unique_ptr<NeuralNetwork> load(std::istream& in);
};  // namespace talawa::nn

}  // namespace talawa::nn
