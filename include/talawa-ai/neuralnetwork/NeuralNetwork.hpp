#pragma once
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "talawa-ai/core/Matrix.hpp"
#include "talawa-ai/core/Optimizer.hpp"
#include "talawa-ai/neuralnetwork/Conv2DLayer.hpp"
#include "talawa-ai/neuralnetwork/DenseLayer.hpp"
#include "talawa-ai/neuralnetwork/Layer.hpp"
#include "talawa-ai/neuralnetwork/Loss.hpp"
#include "talawa-ai/neuralnetwork/Pooling2DLayer.hpp"
namespace talawa_ai {
namespace nn {

using LayerConfigVariant =
    std::variant<DenseLayerConfig, Conv2DLayerConfig, Pooling2DLayerConfig>;
class NeuralNetwork;  // Forward declaration
class NeuralNetworkBuilder {
 public:
  static NeuralNetworkBuilder create(const Shape &shape);
  NeuralNetworkBuilder &add(LayerConfigVariant config);
  NeuralNetworkBuilder &setOptimizer(std::unique_ptr<core::Optimizer> opt);
  NeuralNetworkBuilder &setLossFunction(std::unique_ptr<loss::Loss> loss);
  std::unique_ptr<NeuralNetwork> build();

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

 public:
  core::Matrix predict(const core::Matrix &input);
  float train(const core::Matrix &input, const core::Matrix &target);

  std::vector<std::unique_ptr<Layer>> layers;

  std::unique_ptr<core::Optimizer> optimizer;
  std::unique_ptr<loss::Loss> loss_fn;

  std::unique_ptr<NeuralNetwork> clone() const;

 private:
  Shape get_input_shape() const;
  NeuralNetwork() = default;  // Private constructor (use Builder)
  std::vector<LayerConfigVariant> configs;

  // If set, this activation is applied *after* the last layer during
  // prediction. This allows us to train on Linear (stable) but output Softmax
  // (user-friendly).
  std::optional<core::Activation::Type> m_optimized_act;
  Shape input_shape = {0, 0, 0};
};

}  // namespace nn
}  // namespace talawa_ai
