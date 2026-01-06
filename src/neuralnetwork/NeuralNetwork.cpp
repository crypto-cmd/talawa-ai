#include "talawa/neuralnetwork/NeuralNetwork.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <variant>

#include "talawa/core/Optimizer.hpp"
#include "talawa/neuralnetwork/Conv2DLayer.hpp"
#include "talawa/neuralnetwork/DenseLayer.hpp"

namespace talawa::nn {
class LayerFactory {
 public:
  static std::pair<std::unique_ptr<ILayer>, Shape> create(
      const Conv2DLayerConfig& cfg, const Shape& input_shape) {
    auto layer = std::make_unique<Conv2DLayer>(
        input_shape.depth, input_shape.height, input_shape.width, cfg.filters,
        cfg.kernel_size, cfg.stride, cfg.padding, cfg.init, cfg.act);
    Shape next = layer->getOutputShape();
    return {std::move(layer), next};
  }

  static std::pair<std::unique_ptr<ILayer>, Shape> create(
      const DenseLayerConfig& cfg, const Shape& input_shape) {
    auto layer = std::make_unique<DenseLayer>(input_shape.flat(), cfg.neurons,
                                              cfg.act, cfg.init);
    Shape next = layer->getOutputShape();
    return {std::move(layer), next};
  }
  static std::pair<std::unique_ptr<ILayer>, Shape> create(
      const Pooling2DLayerConfig& cfg, const Shape& input_shape) {
    auto layer = std::make_unique<Pooling2DLayer>(
        input_shape.depth, input_shape.height, input_shape.width, cfg.type,
        cfg.pool_size, cfg.stride);
    Shape next = layer->getOutputShape();
    return {std::move(layer), next};
  }
};

NeuralNetworkBuilder NeuralNetworkBuilder::create(const Shape& shape) {
  NeuralNetworkBuilder builder;
  builder.input_shape = shape;
  return builder;
}

NeuralNetworkBuilder& NeuralNetworkBuilder::add(LayerConfigVariant config) {
  configs.push_back(std::move(config));
  return *this;
}
NeuralNetworkBuilder& NeuralNetworkBuilder::setOptimizer(
    std::unique_ptr<Optimizer> opt) {
  optimizer = std::move(opt);
  return *this;
}
NeuralNetworkBuilder& NeuralNetworkBuilder::setLossFunction(
    std::unique_ptr<loss::Loss> loss) {
  loss_fn = std::move(loss);
  return *this;
}
std::unique_ptr<NeuralNetwork> NeuralNetworkBuilder::build() {
  std::unique_ptr<NeuralNetwork> network(new NeuralNetwork());

  network->configs = std::move(configs);
  network->optimizer = std::move(optimizer);
  network->loss_fn = std::move(loss_fn);
  network->input_shape = this->input_shape;
  Shape current_shape = this->input_shape;
  for (const auto& config_variant : network->configs) {
    std::visit(
        [&](auto&& config) {
          auto result = LayerFactory::create(config, current_shape);
          network->layers.push_back(std::move(result.first));
          current_shape = result.second;
        },
        config_variant);
  }

  // Recompute total parameters after all layers are created
  network->recalcTotalParameters();
  return network;
}

core::Matrix NeuralNetwork::predict(const core::Matrix& input) const {
  auto output = input;

  for (const auto& layer : layers) {
    output = layer->forward(output, false);
  }
  return output;
}
float NeuralNetwork::train(const core::Matrix& input,
                           const core::Matrix& target) {
  // 1. Forward Pass (Record operations for Backprop)
  auto output = input;
  for (auto& layer : layers) {
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
  std::vector<core::Matrix*> all_params;
  std::vector<core::Matrix*> all_grads;

  for (auto& layer : layers) {
    auto p = layer->getParameters();
    auto g = layer->getParameterGradients();
    all_params.insert(all_params.end(), p.begin(), p.end());
    all_grads.insert(all_grads.end(), g.begin(), g.end());
  }

  optimizer->update(all_params, all_grads);

  return loss_val;
}

Shape NeuralNetwork::get_input_shape() const { return input_shape; }

std::unique_ptr<NeuralNetwork> NeuralNetwork::clone() const {
  return std::make_unique<NeuralNetwork>(*this);
}

bool NeuralNetwork::save(std::ostream& out) const {
  // Save input shape
  out.write(reinterpret_cast<const char*>(&input_shape), sizeof(Shape));

  // Save number of layers
  size_t layer_count = layers.size();
  out.write(reinterpret_cast<const char*>(&layer_count), sizeof(size_t));

  // Save each layer (first write an int identifying type)
  for (const auto& layer : layers) {
    int layer_type = -1;
    if (dynamic_cast<DenseLayer*>(layer.get())) {
      layer_type = 0;
    } else if (dynamic_cast<Conv2DLayer*>(layer.get())) {
      layer_type = 1;
    } else if (dynamic_cast<Pooling2DLayer*>(layer.get())) {
      layer_type = 2;
    } else {
      throw std::runtime_error("Unknown layer type during saving.");
    }

    out.write(reinterpret_cast<const char*>(&layer_type), sizeof(int));
    layer->save(out);
  }

  return out.good();
}

// Public helpers: save/load from filename
bool NeuralNetwork::saveToFile(const std::string& filename) const {
  std::ofstream out(filename, std::ios::binary);
  if (!out) return false;

  // Write binary network first
  bool ok = save(out);
  out.close();
  if (!ok) return false;

  // Write human-readable YAML metadata alongside the binary
  std::string yaml_file = filename + std::string(".yaml");
  std::ofstream yout(yaml_file);
  if (!yout) return false;

  // Input shape
  yout << "input_shape:\n";
  yout << "  depth: " << input_shape.depth << "\n";
  yout << "  height: " << input_shape.height << "\n";
  yout << "  width: " << input_shape.width << "\n";

  // Layers
  yout << "layers:\n";
  for (const auto& layer : layers) {
    std::string type = "Unknown";
    if (dynamic_cast<DenseLayer*>(layer.get()))
      type = "Dense";
    else if (dynamic_cast<Conv2DLayer*>(layer.get()))
      type = "Conv2D";
    else if (dynamic_cast<Pooling2DLayer*>(layer.get()))
      type = "Pooling2D";

    yout << "- type: " << type << "\n";
    yout << "  activation: " << layer->activation.getName() << "\n";

    auto params = layer->getParameters();
    if (!params.empty()) {
      yout << "  parameters:\n";
      for (size_t i = 0; i < params.size(); ++i) {
        yout << "    - name: param" << i << "\n";
        yout << "      rows: " << params[i]->rows << "\n";
        yout << "      cols: " << params[i]->cols << "\n";
      }
    }

    auto outshape = layer->getOutputShape();
    yout << "  output_shape:\n";
    yout << "    depth: " << outshape.depth << "\n";
    yout << "    height: " << outshape.height << "\n";
    yout << "    width: " << outshape.width << "\n";
  }

  yout.close();
  return yout.good();
}

std::unique_ptr<NeuralNetwork> NeuralNetwork::loadFromFile(
    const std::string& filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) return nullptr;
  return NeuralNetwork::load(in);
}

std::unique_ptr<NeuralNetwork> NeuralNetwork::load(std::istream& in) {
  auto network = std::unique_ptr<NeuralNetwork>(new NeuralNetwork());

  // Load input shape
  in.read(reinterpret_cast<char*>(&network->input_shape), sizeof(Shape));

  // Load number of layers
  size_t layer_count;
  in.read(reinterpret_cast<char*>(&layer_count), sizeof(size_t));

  // Load each layer
  for (size_t i = 0; i < layer_count; ++i) {
    // Read layer type identifier
    int layer_type;
    in.read(reinterpret_cast<char*>(&layer_type), sizeof(int));

    std::unique_ptr<ILayer> layer;

    if (layer_type == 0) {  // DenseLayer
      layer = std::make_unique<DenseLayer>();
    } else if (layer_type == 1) {  // Conv2DLayer
      layer = std::make_unique<Conv2DLayer>();
    } else if (layer_type == 2) {  // Pooling2DLayer
      layer = std::make_unique<Pooling2DLayer>();
    } else {
      throw std::runtime_error("Unknown layer type during loading.");
    }

    layer->load(in);
    network->layers.push_back(std::move(layer));
  }

  // Recompute parameter count for the newly-loaded network
  network->recalcTotalParameters();

  return network;
}

// Recompute total parameters helper
void NeuralNetwork::recalcTotalParameters() {
  int total = 0;
  for (const auto& layer : layers) {
    auto params = layer->getParameters();
    for (const auto& p : params) {
      total += static_cast<int>(p->rows) * static_cast<int>(p->cols);
    }
  }
  this->_totalParameters = total;
}
}  // namespace talawa::nn
