#include "talawa-ai/neuralnetwork/NeuralNetwork.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <variant>

#include "talawa-ai/core/Optimizer.hpp"
#include "talawa-ai/neuralnetwork/Conv2DLayer.hpp"
#include "talawa-ai/neuralnetwork/DenseLayer.hpp"

using namespace talawa_ai::nn;
namespace talawa_ai {
namespace nn {
class LayerFactory {
 public:
  static std::pair<std::unique_ptr<Layer>, Shape> create(
      const Conv2DLayerConfig &cfg, const Shape &input_shape) {
    auto layer = std::make_unique<Conv2DLayer>(
        input_shape.depth, input_shape.height, input_shape.width, cfg.filters,
        cfg.kernel_size, cfg.stride, cfg.padding, cfg.init, cfg.act);
    Shape next = layer->getOutputShape();
    return {std::move(layer), next};
  }

  static std::pair<std::unique_ptr<Layer>, Shape> create(
      const DenseLayerConfig &cfg, const Shape &input_shape) {
    auto layer = std::make_unique<DenseLayer>(input_shape.flat(), cfg.neurons,
                                              cfg.act, cfg.init);
    Shape next = layer->getOutputShape();
    return {std::move(layer), next};
  }
  static std::pair<std::unique_ptr<Layer>, Shape> create(
      const Pooling2DLayerConfig &cfg, const Shape &input_shape) {
    auto layer = std::make_unique<Pooling2DLayer>(
        input_shape.depth, input_shape.height, input_shape.width, cfg.type,
        cfg.pool_size, cfg.stride);
    Shape next = layer->getOutputShape();
    return {std::move(layer), next};
  }
};

NeuralNetworkBuilder NeuralNetworkBuilder::create(const Shape &shape) {
  NeuralNetworkBuilder builder;
  builder.input_shape = shape;
  return builder;
}

NeuralNetworkBuilder &NeuralNetworkBuilder::add(LayerConfigVariant config) {
  configs.push_back(std::move(config));
  return *this;
}
NeuralNetworkBuilder &NeuralNetworkBuilder::setOptimizer(
    std::unique_ptr<Optimizer> opt) {
  optimizer = std::move(opt);
  return *this;
}
NeuralNetworkBuilder &NeuralNetworkBuilder::setLossFunction(
    std::unique_ptr<loss::Loss> loss) {
  loss_fn = std::move(loss);
  return *this;
}
std::unique_ptr<NeuralNetwork> NeuralNetworkBuilder::build() {
  std::cout << "Building Neural Network with " << configs.size()
            << " layers.\n";

  std::cout << "Checking for optimization opportunities...\n";

  std::unique_ptr<NeuralNetwork> network(new NeuralNetwork());

  network->configs = std::move(configs);
  network->optimizer = std::move(optimizer);
  network->loss_fn = std::move(loss_fn);

  Shape current_shape = input_shape;
  for (const auto &config_variant : network->configs) {
    std::visit(
        [&](auto &&config) {
          auto result = LayerFactory::create(config, current_shape);
          network->layers.push_back(std::move(result.first));
          current_shape = result.second;
        },
        config_variant);
  }

  std::cout << "Neural Network built successfully.\n";
  return network;
}

core::Matrix NeuralNetwork::predict(const core::Matrix &input) {
  auto output = input;

  for (const auto &layer : layers) {
    output = layer->forward(output, false);
  }
  return output;
}
float NeuralNetwork::train(const core::Matrix &input,
                           const core::Matrix &target) {
  // 1. Forward Pass (Record operations for Backprop)
  auto output = input;
  for (auto &layer : layers) {
    output = layer->forward(output, true);
  }

  // 2. Calculate Loss & Gradient
  float loss_val = loss_fn->calculate(output, target);
  core::Matrix error = loss_fn->gradient(output, target);  // dL/dY

  // 3. Backward Pass
  core::Matrix gradient = error;
  for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
    gradient = (*it)->backward(gradient);
  }

  // 4. Update Weights
  std::vector<core::Matrix *> all_params;
  std::vector<core::Matrix *> all_grads;

  for (auto &layer : layers) {
    auto p = layer->getParameters();
    auto g = layer->getParameterGradients();
    all_params.insert(all_params.end(), p.begin(), p.end());
    all_grads.insert(all_grads.end(), g.begin(), g.end());
  }

  optimizer->update(all_params, all_grads);

  return loss_val;
}

}  // namespace nn
}  // namespace talawa_ai
