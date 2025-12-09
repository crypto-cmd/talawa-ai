#pragma once
#include <vector>

#include "talawa-ai/core/Matrix.hpp"

namespace talawa_ai {
namespace core {

// --- Base Optimizer Interface ---
class Optimizer {
 public:
  virtual ~Optimizer() = default;

  // The core function: updates parameters using their gradients
  // We pass pointers so we can modify the actual weights in memory.
  virtual void update(const std::vector<Matrix*>& params,
                      const std::vector<Matrix*>& grads) = 0;
  virtual std::string getName() const = 0;
};

// --- Stochastic Gradient Descent (SGD) ---
class SGD : public Optimizer {
 private:
  float learning_rate;

 public:
  explicit SGD(float learning_rate = 0.01f);

  void update(const std::vector<Matrix*>& params,
              const std::vector<Matrix*>& grads) override;
  std::string getName() const override { return "Stochastic Gradient Descent"; }

  void setLearningRate(float lr) { learning_rate = lr; }
};

// --- Adam ----
class Adam : public Optimizer {
 private:
  float learning_rate;
  float beta1;
  float beta2;
  float epsilon;

  int t;  // Time step

  // State caches (Momentum and Velocity)
  // These must align 1:1 with the params vector
  std::vector<Matrix> m_cache;
  std::vector<Matrix> v_cache;

 public:
  explicit Adam(float learning_rate = 0.001f, float beta1 = 0.9f,
                float beta2 = 0.999f, float epsilon = 1e-8f);

  void update(const std::vector<Matrix*>& params,
              const std::vector<Matrix*>& grads) override;
  std::string getName() const override { return "Adam"; }
};

}  // namespace core
}  // namespace talawa_ai
