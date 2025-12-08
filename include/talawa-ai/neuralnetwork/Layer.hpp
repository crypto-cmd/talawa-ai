#pragma once
#include "talawa-ai/core/Activation.hpp"
#include "talawa-ai/core/Initializer.hpp"
#include "talawa-ai/core/Matrix.hpp"

#define THROW_LAYER_ERROR(msg) THROW_TALAWA_AI_ERROR("Layer", msg)
namespace talawa_ai {
namespace nn {
using namespace talawa_ai::core;

enum class LayerType { DENSE, CONV2D };

struct Shape {
  int depth;
  int height;
  int width;

  int flat() const { return depth * height * width; }
};

class Layer {
 public:
  virtual ~Layer() = default;
  virtual Matrix forward(const Matrix &input, bool is_training = true) = 0;
  virtual Matrix backward(const Matrix &outputGradients) = 0;
  virtual std::vector<Matrix *> getParameters() = 0;
  virtual std::vector<Matrix *> getParameterGradients() = 0;

  // Hyperparameters
  Activation activation;
  Initializer initializer;

  // Debugging utility to print layer info
  virtual std::string info() const = 0;

  // Saving and loading
  virtual void save(std::ostream &out) const = 0;
  virtual void load(std::istream &in) = 0;

  virtual Shape getOutputShape() const = 0;
};
}  // namespace nn
}  // namespace talawa_ai
