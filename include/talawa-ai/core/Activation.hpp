#pragma once
#include <string>

#include "talawa-ai/core/Matrix.hpp"

namespace talawa_ai {
namespace core {

class Activation {
 public:
  enum Type {
    LINEAR,      // No activation (Identity)
    RELU,        // Rectified Linear Unit (Common for hidden layers)
    SIGMOID,     // S-curve (0 to 1)
    TANH,        // Hyperbolic Tangent (-1 to 1)
    SOFTMAX,     // Probability distribution (Output layer)
    LOG_SOFTMAX  // Log of Softmax (Output layer)
  };

  Activation(Type type = RELU);

  // Forward Pass: Returns A = f(Z)
  Matrix apply(const Matrix& z) const;

  // Backward Pass: Returns f'(Z)
  Matrix derivative(const Matrix& z) const;

  void backprop(const Matrix& a, const Matrix& outputGradients,
                  Matrix& dZ) const;
  // Utility
  std::string getName() const;

  Type type;
};

}  // namespace core
}  // namespace talawa_ai
