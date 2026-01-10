#pragma once
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "talawa/core/Matrix.hpp"
#include "talawa/rl/IAgent.hpp"

namespace talawa {
namespace core {

// --- Base Optimizer Interface ---
class Optimizer : public rl::agent::ILearnable {
 public:
  virtual ~Optimizer() = default;

  // The core function: updates parameters using their gradients
  // We pass pointers so we can modify the actual weights in memory.
  virtual void update(const std::vector<Matrix*>& params,
                      const std::vector<Matrix*>& grads) = 0;
  virtual std::string getName() const = 0;

  // Deep copy support
  virtual std::unique_ptr<Optimizer> clone() const = 0;
};

// --- Stochastic Gradient Descent (SGD) ---
class SGD : public Optimizer {
 private:

 public:
  explicit SGD();

  void update(const std::vector<Matrix*>& params,
              const std::vector<Matrix*>& grads) override;
  std::string getName() const override { return "Stochastic Gradient Descent"; }

  std::unique_ptr<Optimizer> clone() const override {
    return std::make_unique<SGD>();
  }
};

// --- Adam ----
class Adam : public Optimizer {
 private:
  float beta1;
  float beta2;
  float epsilon;

  int t;  // Time step

  // State caches (Momentum and Velocity)
  // These must align 1:1 with the params vector
  std::vector<Matrix> m_cache;
  std::vector<Matrix> v_cache;

 public:
  explicit Adam(float beta1 = 0.9f, float beta2 = 0.999f,
                float epsilon = 1e-8f);

  void update(const std::vector<Matrix*>& params,
              const std::vector<Matrix*>& grads) override;
  std::string getName() const override { return "Adam"; }
  std::unique_ptr<Optimizer> clone() const override {
    return std::make_unique<Adam>();
  }
};

}  // namespace core
}  // namespace talawa
