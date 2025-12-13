#include "talawa-ai/core/Optimizer.hpp"

#include <immintrin.h>

#include <cstring>
#include <stdexcept>

namespace talawa_ai {
namespace core {

SGD::SGD(float learning_rate) : learning_rate(learning_rate) {}

void SGD::update(const std::vector<Matrix*>& params,
                 const std::vector<Matrix*>& grads) {
  // Sanity check: We must have exactly one gradient matrix for every parameter
  // matrix
  if (params.size() != grads.size()) {
    throw std::runtime_error("Optimizer Mismatch: Parameter count (" +
                             std::to_string(params.size()) +
                             ") does not match Gradient count (" +
                             std::to_string(grads.size()) + ").");
  }

  float lr = this->learning_rate;
  __m256 lr_vec = _mm256_set1_ps(lr);

  // Iterate over all layers' parameters (weights and biases)
  for (size_t i = 0; i < params.size(); ++i) {
    Matrix* param = params[i];
    Matrix* grad = grads[i];

    float* P = param->rawData();
    const float* G = grad->rawData();
    int size = param->rows * param->cols;
    int main_loop_limit = (size / 8) * 8;

    // AVX2 vectorized: W = W - lr * dW
    for (int j = 0; j < main_loop_limit; j += 8) {
      __m256 p_vec = _mm256_loadu_ps(&P[j]);
      __m256 g_vec = _mm256_loadu_ps(&G[j]);

      // Clamp gradients to [-1, 1] for stability
      __m256 thresh_vec = _mm256_set1_ps(1);
      __m256 neg_thresh_vec = _mm256_set1_ps(-1);
      g_vec = _mm256_min_ps(thresh_vec, _mm256_max_ps(neg_thresh_vec, g_vec));

      __m256 update = _mm256_fnmadd_ps(lr_vec, g_vec, p_vec);  // p - lr*g
      _mm256_storeu_ps(&P[j], update);
    }
    // Scalar cleanup
    for (int j = main_loop_limit; j < size; ++j) {
      P[j] -= lr * std::clamp(G[j], -1.0f, 1.0f);
    }
  }
}

Adam::Adam(float lr, float beta1, float beta2, float eps)
    : learning_rate(lr), beta1(beta1), beta2(beta2), epsilon(eps), t(0) {}

void Adam::update(const std::vector<Matrix*>& params,
                  const std::vector<Matrix*>& grads) {
  if (params.size() != grads.size()) {
    throw std::runtime_error(
        "Optimizer Mismatch: Params/Grads count mismatch.");
  }

  // 1. Initialize State Caches on first run
  if (m_cache.empty()) {
    m_cache.reserve(params.size());
    v_cache.reserve(params.size());
    for (const auto* p : params) {
      // Create zero matrices matching parameter shapes
      m_cache.push_back(Matrix::zeros(p->rows, p->cols));
      v_cache.push_back(Matrix::zeros(p->rows, p->cols));
    }
  }

  // 2. Increment Time Step
  t++;

  // 3. Calculate Bias Corrections
  float correction_m = 1.0f / (1.0f - std::pow(beta1, t));
  float correction_v = 1.0f / (1.0f - std::pow(beta2, t));

  // Pre-compute constants for AVX
  __m256 beta1_vec = _mm256_set1_ps(beta1);
  __m256 beta2_vec = _mm256_set1_ps(beta2);
  __m256 one_minus_beta1_vec = _mm256_set1_ps(1.0f - beta1);
  __m256 one_minus_beta2_vec = _mm256_set1_ps(1.0f - beta2);
  __m256 correction_m_vec = _mm256_set1_ps(correction_m);
  __m256 correction_v_vec = _mm256_set1_ps(correction_v);
  __m256 lr_vec = _mm256_set1_ps(learning_rate);
  __m256 eps_vec = _mm256_set1_ps(epsilon);

  // 4. Update Parameters
  for (size_t i = 0; i < params.size(); ++i) {
    float* P = params[i]->rawData();
    const float* G = grads[i]->rawData();
    float* M = m_cache[i].rawData();
    float* V = v_cache[i].rawData();

    int size = params[i]->rows * params[i]->cols;
    int main_loop_limit = (size / 8) * 8;

    // AVX2 vectorized Adam update
    for (int j = 0; j < main_loop_limit; j += 8) {
      __m256 g_vec = _mm256_loadu_ps(&G[j]);
      __m256 m_vec = _mm256_loadu_ps(&M[j]);
      __m256 v_vec = _mm256_loadu_ps(&V[j]);
      __m256 p_vec = _mm256_loadu_ps(&P[j]);

      __m256 thresh_vec = _mm256_set1_ps(1);
      __m256 neg_thresh_vec = _mm256_set1_ps(-1);
      g_vec = _mm256_min_ps(thresh_vec, _mm256_max_ps(neg_thresh_vec, g_vec));

      // m = beta1 * m + (1 - beta1) * g
      m_vec = _mm256_fmadd_ps(beta1_vec, m_vec,
                              _mm256_mul_ps(one_minus_beta1_vec, g_vec));

      // v = beta2 * v + (1 - beta2) * g^2
      __m256 g_sq = _mm256_mul_ps(g_vec, g_vec);
      v_vec = _mm256_fmadd_ps(beta2_vec, v_vec,
                              _mm256_mul_ps(one_minus_beta2_vec, g_sq));

      // Store updated m and v
      _mm256_storeu_ps(&M[j], m_vec);
      _mm256_storeu_ps(&V[j], v_vec);

      // m_hat = m * correction_m, v_hat = v * correction_v
      __m256 m_hat = _mm256_mul_ps(m_vec, correction_m_vec);
      __m256 v_hat = _mm256_mul_ps(v_vec, correction_v_vec);

      // theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
      __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(v_hat), eps_vec);
      __m256 update = _mm256_div_ps(_mm256_mul_ps(lr_vec, m_hat), denom);
      p_vec = _mm256_sub_ps(p_vec, update);

      _mm256_storeu_ps(&P[j], p_vec);
    }

    // Scalar cleanup for remaining elements
    for (int j = main_loop_limit; j < size; ++j) {
      float g = std::clamp(G[j], -1.0f, 1.0f);

      // Update biased first moment (Momentum)
      M[j] = beta1 * M[j] + (1.0f - beta1) * g;

      // Update biased second raw moment (Velocity)
      V[j] = beta2 * V[j] + (1.0f - beta2) * g * g;

      // Compute bias-corrected moments
      float m_hat = M[j] * correction_m;
      float v_hat = V[j] * correction_v;

      // Update Parameter
      P[j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
  }
}

}  // namespace core
}  // namespace talawa_ai
