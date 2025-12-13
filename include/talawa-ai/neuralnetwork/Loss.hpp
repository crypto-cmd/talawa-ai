#pragma once
#include <string>

#include "talawa-ai/core/Matrix.hpp"

namespace talawa_ai {
namespace nn {
namespace loss {

using namespace talawa_ai::core;

enum class LossInputType {
  PROBABILITIES,  // Expects 0.0 to 1.0 (e.g., from Softmax)
  LOGITS,         // Expects raw scores (e.g., from Linear)
  RAW_VALUES      // Expects any value (e.g., Regression)
};

class Loss {
 public:
  virtual ~Loss() = default;

  // Calculates the scalar loss value (for tracking progress)
  virtual float calculate(const Matrix& prediction, const Matrix& target) = 0;

  // Calculates the gradient dL/dY (to start Backpropagation)
  virtual Matrix gradient(const Matrix& prediction, const Matrix& target) = 0;

  virtual std::string getName() const = 0;
  virtual LossInputType getInputType() const = 0;
};

// include/talawa-ai/neuralnetwork/Loss.hpp

class HuberLoss : public Loss {
 public:
  float calculate(const Matrix& prediction, const Matrix& target) override;
  Matrix gradient(const Matrix& prediction, const Matrix& target) override;
  std::string getName() const override { return "Huber Loss"; }
  LossInputType getInputType() const override {
    return LossInputType::RAW_VALUES;
  }
};
// --- Mean Squared Error (MSE) ---
// Best for: Regression (predicting house prices, coordinates, etc.)
class MeanSquaredError : public Loss {
 public:
  float calculate(const Matrix& prediction, const Matrix& target) override;
  Matrix gradient(const Matrix& prediction, const Matrix& target) override;
  std::string getName() const override { return "Mean Squared Error"; }
  LossInputType getInputType() const override {
    return LossInputType::RAW_VALUES;
  }
};

class CrossEntropyLoss : public Loss {
 public:
  float calculate(const Matrix& prediction, const Matrix& target) override;
  Matrix gradient(const Matrix& prediction, const Matrix& target) override;
  std::string getName() const override { return "Cross Entropy Loss"; }
  LossInputType getInputType() const override { return LossInputType::LOGITS; }
};

class CategoricalCrossEntropyLoss : public Loss {
 public:
  float calculate(const Matrix& prediction, const Matrix& target) override;
  Matrix gradient(const Matrix& prediction, const Matrix& target) override;
  std::string getName() const override {
    return "Categorical Cross Entropy Loss";
  }
  LossInputType getInputType() const override {
    return LossInputType::PROBABILITIES;
  }
};
class CrossEntropyWithLogitsLoss : public Loss {
 public:
  float calculate(const Matrix& prediction, const Matrix& target) override;
  Matrix gradient(const Matrix& prediction, const Matrix& target) override;
  std::string getName() const override {
    return "Cross Entropy With Logits Loss";
  }
  LossInputType getInputType() const override { return LossInputType::LOGITS; }
};

class EmptyLoss : public Loss {
 public:
  float calculate(const Matrix& prediction, const Matrix& target) override {
    return 0.0f;
  }
  Matrix gradient(const Matrix& prediction, const Matrix& target) override {
    return Matrix::zeros(prediction.rows, prediction.cols);
  }
  std::string getName() const override { return "Empty Loss"; }
  LossInputType getInputType() const override {
    return LossInputType::RAW_VALUES;
  }
};
}  // namespace loss
}  // namespace nn
}  // namespace talawa_ai
