#pragma once
#include <functional>
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
#include "talawa/rl/IAgent.hpp"
namespace talawa::nn {

using LayerConfigVariant =
    std::variant<DenseLayerConfig, Conv2DLayerConfig, Pooling2DLayerConfig>;
class NeuralNetwork;  // Forward declaration
class NeuralNetworkBuilder {
 private:
  std::vector<std::unique_ptr<ILayer>> prebuilt_layers;
  int num_prebuilt_standard_layers_ = 0;
  void prebuild();

 public:
  static NeuralNetworkBuilder create(const Shape& shape);
  NeuralNetworkBuilder& add(LayerConfigVariant config);
  NeuralNetworkBuilder& setOptimizer(std::unique_ptr<core::Optimizer> opt);
  NeuralNetworkBuilder& setLossFunction(std::unique_ptr<loss::Loss> loss);
  NeuralNetworkBuilder& inject(
      std::function<std::pair<std::unique_ptr<ILayer>, Shape>(const Shape&)>
          layer_creator);
  std::unique_ptr<NeuralNetwork> build(float learning_rate = 0.1f);

 private:
  Shape input_shape;
  std::vector<LayerConfigVariant> configs;
  std::unique_ptr<core::Optimizer> optimizer = std::make_unique<core::SGD>();
  std::unique_ptr<loss::Loss> loss_fn =
      std::make_unique<loss::MeanSquaredError>();
  NeuralNetworkBuilder() = default;  // Private constructor
};
class NeuralNetwork : public rl::agent::ILearnable {
  friend class NeuralNetworkBuilder;

 private:
  int _totalParameters = 0;

 public:
  NeuralNetwork(const NeuralNetwork& other) {
    // Deep copy layers
    layers.clear();
    for (const auto& layer : other.layers) {
      layers.push_back(layer->clone());
    }
    // Deep copy optimizer and loss function
    optimizer = other.optimizer->clone();
    loss_fn = other.loss_fn->clone();

    // Copy configs and input shape
    configs = other.configs;
    input_shape = other.input_shape;
    m_optimized_act = other.m_optimized_act;
    _totalParameters = other._totalParameters;
  }
  const std::vector<std::unique_ptr<ILayer>>& getLayers() const {
    return layers;
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

  void set_learning_rate(float lr) override {
    rl::agent::ILearnable::set_learning_rate(lr);
    if (optimizer) {
      optimizer->set_learning_rate(lr);  // Propagate to optimizer
    }
  }

  void print() const {
    std::cout << "NeuralNetwork: \n";
    std::cout << "Input Shape: (" << input_shape.height << ", "
              << input_shape.width << ", " << input_shape.depth << ")\n";
    std::cout << "Total Parameters: " << _totalParameters << "\n";
    std::cout << "Layers:\n";
    for (size_t i = 0; i < layers.size(); ++i) {
      std::cout << " Layer " << i + 1 << ": " << layers[i]->info() << "\n";
      for (const auto& param : layers[i]->getParameters()) {
        std::cout << "  Param Shape: (" << param->rows << ", " << param->cols
                  << ")\n";
        param->print();
      }
    }
  }

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
