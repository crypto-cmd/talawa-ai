#pragma once
#include <stdexcept>
#include <vector>

#include "talawa/core/Activation.hpp"
#include "talawa/neuralnetwork/Layer.hpp"

namespace talawa {
namespace nn {

struct DenseLayerConfig {
  int neurons;
  Activation::Type act = Activation::RELU;
  Initializer init = Initializer::GLOROT_UNIFORM;
};

class DenseLayer : public ILayer {
 private:
  // Layer dimensions
  size_t in;
  size_t out;

  // Parameters
  core::Matrix weights;  // Shape: (input_size, output_size)
  core::Matrix biases;   // Shape: (1, output_size)

  // Cache for backpropagation
  core::Matrix input_cache;            // X from forward pass
  core::Matrix z_cache;                // WX+B from forward pass
  core::Matrix a_cache;                // Activation from forward pass
  core::Matrix input_gradients_cache;  // Pre-allocated for backward pass

  // Gradients
  core::Matrix weights_grad;  // Same shape as weights
  core::Matrix biases_grad;

  core::Matrix input_T_cache;    // To store input_cache.transpose()
  core::Matrix weights_T_cache;  // To store weights.transpose()
  core::Matrix dZ;               // To store activation gradient

 public:
  DenseLayer();
  DenseLayer(size_t input_dim, size_t neurons,
             Activation act = Activation::RELU,
             Initializer init = Initializer::GLOROT_UNIFORM);

  // Core Operations
  core::Matrix forward(const core::Matrix& input,
                       bool is_training = true) override;
  core::Matrix backward(const core::Matrix& outputGradients) override;

  // Optimizer

  std::vector<core::Matrix*> getParameters() override;
  std::vector<core::Matrix*> getParameterGradients() override;

  // Utilities
  std::string info() const override;
  void save(std::ostream& out) const override;
  void load(std::istream& in) override;

  Shape getOutputShape() const override;
};
};  // namespace nn
};  // namespace talawa
