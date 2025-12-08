#include "talawa-ai/core/Activation.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace talawa_ai {
namespace core {
const float EPSILON = 1e-7f;

Activation::Activation(Type type) : type(type) {}

// CONSTANT: Minimum probability to ensure gradients keep flowing.
// If this is 0.0, the gradient (y * error) becomes 0.0, killing the neuron.
const float MIN_PROB = 1e-7f;

// --- Forward Pass (Apply) ---
Matrix Activation::apply(const Matrix& z) const {
  Matrix result = z;  // Copy dimensions
  float* data = &result(0, 0);
  size_t size = result.rows * result.cols;

  switch (type) {
    case LINEAR:
      return result;  // Identity

    case RELU: {
      // f(x) = max(0, x) - Vectorized
      __m256 zero_vec = _mm256_setzero_ps();
      int main_limit = (size / 8) * 8;
      for (int i = 0; i < main_limit; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(&data[i]);
        __m256 result_vec = _mm256_max_ps(zero_vec, x_vec);
        _mm256_storeu_ps(&data[i], result_vec);
      }
      for (int i = main_limit; i < size; ++i) {
        data[i] = data[i] > 0.0f ? data[i] : 0.0f;
      }
      break;
    }

    case SIGMOID:
      // f(x) = 1 / (1 + e^-x)
      for (int i = 0; i < size; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
      }
      break;

    case TANH:
      // f(x) = tanh(x)
      for (int i = 0; i < size; ++i) {
        data[i] = std::tanh(data[i]);
      }
      break;

    case SOFTMAX: {
      // Formula: exp(x_i) / sum(exp(x_j))
      // Stabilization: exp(x_i - max_val) / sum(exp(x_j - max_val))

      // Iterate Row by Row (Batch)
      for (int r = 0; r < result.rows; ++r) {
        float* row = &data[r * result.cols];
        int cols = result.cols;

        // 1. Find Max (for stability)
        float max_val = row[0];
        for (int c = 1; c < cols; ++c) {
          if (row[c] > max_val) max_val = row[c];
        }

        // 2. Exponentiate and Sum
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
          float val = std::exp(row[c] - max_val);
          row[c] = val;
          sum += val;
        }

        // 3. Normalize
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; ++c) {
          float prob = row[c] * inv_sum;
          // Clip to EPSILON
          if (prob < EPSILON) prob = EPSILON;
          if (prob > 1.0f - EPSILON) prob = 1.0f - EPSILON;
          row[c] = prob;
        }
      }
      break;
      case LOG_SOFTMAX: {
        throw std::runtime_error("Log-Softmax not implemented.");
      }
    }
  }
  return result;
}

void Activation::backprop(const Matrix& a, const Matrix& outputGradients,
                          Matrix& dZ) const {
  // a: Activated outputs from forward pass (A)
  // outputGradients: dL/dA (Gradient from the layer ahead)

  // Ensure output buffer is correct size
  if (dZ.rows != a.rows || dZ.cols != a.cols) {
    dZ = Matrix(a.rows, a.cols);
  }

  float* dZ_data = dZ.rawData();
  const float* a_data = a.rawData();
  const float* grad_data = outputGradients.rawData();
  int size = a.rows * a.cols;

  switch (type) {
    case LINEAR:
      // dZ = dL/dA * 1 (pass through unchanged)
      std::memcpy(dZ_data, grad_data, size * sizeof(float));
      break;
    case RELU: {
      // f'(x) = 1 if x > 0 else 0
      __m256 zero_vec = _mm256_setzero_ps();
      int main_limit = (size / 8) * 8;
      for (int i = 0; i < main_limit; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a_data[i]);
        __m256 g_vec = _mm256_loadu_ps(&grad_data[i]);
        // Create mask: a > 0 ? all 1s : all 0s
        __m256 mask = _mm256_cmp_ps(a_vec, zero_vec, _CMP_GT_OQ);
        // Apply mask to gradient
        __m256 result = _mm256_and_ps(mask, g_vec);
        _mm256_storeu_ps(&dZ_data[i], result);
      }
      for (int i = main_limit; i < size; ++i) {
        dZ_data[i] = a_data[i] > 0.0f ? grad_data[i] : 0.0f;
      }
      break;
    }

    case SIGMOID: {
      // f'(x) = f(x) * (1 - f(x))
      __m256 one_vec = _mm256_set1_ps(1.0f);
      int main_limit = (size / 8) * 8;
      for (int i = 0; i < main_limit; i += 8) {
        __m256 s_vec = _mm256_loadu_ps(&a_data[i]);
        __m256 g_vec = _mm256_loadu_ps(&grad_data[i]);
        // s * (1 - s) * grad
        __m256 deriv = _mm256_mul_ps(s_vec, _mm256_sub_ps(one_vec, s_vec));
        __m256 result = _mm256_mul_ps(deriv, g_vec);
        _mm256_storeu_ps(&dZ_data[i], result);
      }
      for (int i = main_limit; i < size; ++i) {
        float s = a_data[i];
        dZ_data[i] = s * (1.0f - s) * grad_data[i];
      }
      break;
    }

    case TANH: {
      // f'(x) = 1 - tanh^2(x)
      __m256 one_vec = _mm256_set1_ps(1.0f);
      int main_limit = (size / 8) * 8;
      for (int i = 0; i < main_limit; i += 8) {
        __m256 t_vec = _mm256_loadu_ps(&a_data[i]);
        __m256 g_vec = _mm256_loadu_ps(&grad_data[i]);
        // (1 - t^2) * grad
        __m256 t_sq = _mm256_mul_ps(t_vec, t_vec);
        __m256 deriv = _mm256_sub_ps(one_vec, t_sq);
        __m256 result = _mm256_mul_ps(deriv, g_vec);
        _mm256_storeu_ps(&dZ_data[i], result);
      }
      for (int i = main_limit; i < size; ++i) {
        float t = a_data[i];
        dZ_data[i] = (1.0f - t * t) * grad_data[i];
      }
      break;
    }

    case SOFTMAX:
      // --- Softmax Vector-Jacobian Product ---
      // Formula: dx_i = y_i * (grad_i - sum(y_k * grad_k))
      for (int r = 0; r < outputGradients.rows; ++r) {
        const float* a_row = &a_data[r * a.cols];
        const float* g_row = &grad_data[r * outputGradients.cols];
        float* dz_row = &dZ_data[r * dZ.cols];
        int cols = outputGradients.cols;

        // 1. Calculate Dot Product: sum(y_k * grad_k)
        float dot = 0.0f;
        int main_limit = (cols / 8) * 8;
        __m256 dot_vec = _mm256_setzero_ps();
        for (int c = 0; c < main_limit; c += 8) {
          __m256 a_vec = _mm256_loadu_ps(&a_row[c]);
          __m256 g_vec = _mm256_loadu_ps(&g_row[c]);
          dot_vec = _mm256_fmadd_ps(a_vec, g_vec, dot_vec);
        }
        // Horizontal sum
        float temp[8];
        _mm256_storeu_ps(temp, dot_vec);
        for (int x = 0; x < 8; ++x) dot += temp[x];
        // Scalar cleanup
        for (int c = main_limit; c < cols; ++c) {
          dot += a_row[c] * g_row[c];
        }

        // 2. Apply formula: y * (g - dot)
        __m256 dot_broadcast = _mm256_set1_ps(dot);
        for (int c = 0; c < main_limit; c += 8) {
          __m256 y_vec = _mm256_loadu_ps(&a_row[c]);
          __m256 g_vec = _mm256_loadu_ps(&g_row[c]);
          __m256 result =
              _mm256_mul_ps(y_vec, _mm256_sub_ps(g_vec, dot_broadcast));
          _mm256_storeu_ps(&dz_row[c], result);
        }
        for (int c = main_limit; c < cols; ++c) {
          dz_row[c] = a_row[c] * (g_row[c] - dot);
        }
      }
      break;
    default:
      throw std::runtime_error(
          "Backprop not defined for this activation type.");
  }
}
// --- Backward Pass (Derivative) ---
Matrix Activation::derivative(const Matrix& z) const {
  Matrix result = z;

  switch (type) {
    case LINEAR:
      result.fill(1.0f);  // Derivative of x is 1
      break;

    case RELU:
      // f'(x) = 1 if x > 0 else 0
      result.apply([&](int, int, float x) { return x > 0.0f ? 1.0f : 0.0f; });
      break;

    case SIGMOID:
      // f'(x) = f(x) * (1 - f(x))
      // Note: We re-calculate sigmoid here. Ideally, passed 'z' is cached
      // activation.
      result.apply([&](int, int, float x) {
        float s = 1.0f / (1.0f + std::exp(-x));
        return s * (1.0f - s);
      });
      break;

    case TANH:
      // f'(x) = 1 - tanh^2(x)
      result.apply([&](int, int, float x) {
        float t = std::tanh(x);
        return 1.0f - (t * t);
      });
      break;

    default:
      throw std::runtime_error(
          "Derivative not defined for this activation type.");
  }
  return result;
}

std::string Activation::getName() const {
  switch (type) {
    case LINEAR:
      return "Linear";
    case RELU:
      return "ReLU";
    case SIGMOID:
      return "Sigmoid";
    case TANH:
      return "Tanh";
    case SOFTMAX:
      return "Softmax";
    default:
      return "Unknown";
  }
}

}  // namespace core
}  // namespace talawa_ai
