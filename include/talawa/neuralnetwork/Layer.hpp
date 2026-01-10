#pragma once
#include "talawa/core/Activation.hpp"
#include "talawa/core/Initializer.hpp"
#include "talawa/core/Matrix.hpp"

#include <memory>

#define THROW_LAYER_ERROR(msg) THROW_talawa_ERROR("Layer", msg)
namespace talawa {
namespace nn {
using namespace talawa::core;

enum class LayerType { DENSE, CONV2D };

struct Shape {
  size_t depth;
  size_t height;
  size_t width;

  size_t flat() const { return depth * height * width; }
};

class ILayer {
 public:
  virtual ~ILayer() = default;
  virtual Matrix forward(const Matrix& input, bool is_training = true) = 0;
  virtual Matrix backward(const Matrix& outputGradients) = 0;
  virtual std::vector<Matrix*> getParameters() = 0;
  virtual std::vector<Matrix*> getParameterGradients() = 0;

  // Hyperparameters
  Activation activation;
  Initializer initializer;

  // Debugging utility to print layer info
  virtual std::string info() const = 0;

  // Saving and loading
  virtual void save(std::ostream& out) const = 0;
  virtual void load(std::istream& in) = 0;

  virtual Shape getOutputShape() const = 0;

  // Deep copy support
  virtual std::unique_ptr<ILayer> clone() const = 0;
};
}  // namespace nn
}  // namespace talawa
