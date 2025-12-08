#include "talawa-ai/neuralnetwork/Loss.hpp"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace talawa_ai {
namespace nn {
namespace loss {
using namespace talawa_ai::core;

// Small epsilon to prevent division by zero or log(0)
const float EPSILON = 1e-7f;

// ==========================================
// Mean Squared Error (MSE)
// Formula: L = (1/N) * Σ (y_pred - y_true)^2
// ==========================================

float MeanSquaredError::calculate(const Matrix& prediction,
                                  const Matrix& target) {
  float total_loss = prediction.reduce<float>(
      [&](float acc, int row, int col, float pred_val) {
        float true_val = target(row, col);
        float diff = pred_val - true_val;
        return acc + diff * diff;
      },
      0.0f);
  int size = prediction.rows * prediction.cols;

  return total_loss / static_cast<float>(size);
}

Matrix MeanSquaredError::gradient(const Matrix& prediction,
                                  const Matrix& target) {
  // Gradient: dL/dY = (2/N) * (y_pred - y_true)

  float n = static_cast<float>(prediction.rows * prediction.cols);
  float factor = 2.0f / n;

  return prediction.map([&](int row, int col, float pred_val) {
    float true_val = target(row, col);
    return factor * (pred_val - true_val);
  });
}

// ==========================================
// Cross Entropy Loss
// Formula: L = -Σ y_true * log(y_pred)
// ==========================================

float CrossEntropyLoss::calculate(const Matrix& prediction,
                                  const Matrix& target) {
  float total_loss = prediction.reduce<float>(
      [&](float acc, int row, int col, float pred_val) {
        float true_val = target(row, col);
        float clipped_pred = std::clamp(pred_val, EPSILON, 1.0f - EPSILON);
        return acc - true_val * std::log(clipped_pred);
      },
      0.0f);
  return total_loss / static_cast<float>(prediction.rows);
}

Matrix CrossEntropyLoss::gradient(const Matrix& prediction,
                                  const Matrix& target) {
  // Gradient: dL/dY = - (y_true / y_pred) / N
  float n = static_cast<float>(prediction.rows);
  return prediction.map([&](int row, int col, float pred_val) {
    float true_val = target(row, col);
    float clipped_pred = std::clamp(pred_val, EPSILON, 1.0f - EPSILON);
    return -(true_val / clipped_pred) / n;
  });
}

// ==========================================
// Categorical Cross Entropy
// ==========================================

float CategoricalCrossEntropyLoss::calculate(const Matrix& prediction,
                                             const Matrix& target) {
  float total_loss = prediction.reduce<float>(
      [&target](float acc, int row, int col, float value) {
        float p = std::clamp(value, EPSILON, 1.0f - EPSILON);
        acc += -target(row, col) * std::log(p);
        return acc;
      },
      0.0f);

  size_t size = prediction.rows * prediction.cols;

  return total_loss / static_cast<float>(prediction.rows);
}

Matrix CategoricalCrossEntropyLoss::gradient(const Matrix& prediction,
                                             const Matrix& target) {
  // Gradient w.r.t Prediction: dL/dp = - (target / prediction)
  float batch_scale = 1.0f / static_cast<float>(prediction.rows);

  return prediction.map([&](int row, int col, float p_val) {
    auto p = std::clamp(p_val, EPSILON, 1.0f - EPSILON);
    auto grad_val = -(target(row, col) / p);

    grad_val *= batch_scale;

    return grad_val;
  });
}

// ==========================================
// Cross Entropy With Logits (Stable)
// ==========================================
float CrossEntropyWithLogitsLoss::calculate(const Matrix& prediction,
                                            const Matrix& target) {
  // Prediction contains LOGITS (Raw Z).
  // Formula: - sum( target * log_softmax(z) )
  float total_loss = 0.0f;

  for (size_t r = 0; r < prediction.rows; ++r) {
    // 1. Find Max for numerical stability (Log-Sum-Exp trick)
    float max_val = prediction(r, 0);
    for (size_t c = 1; c < prediction.cols; ++c) {
      if (prediction(r, c) > max_val) max_val = prediction(r, c);
    }

    // 2. Calculate Sum of Exponentials
    float sum_exp = 0.0f;
    for (size_t c = 0; c < prediction.cols; ++c) {
      sum_exp += std::exp(prediction(r, c) - max_val);
    }
    float log_sum_exp = std::log(sum_exp);

    // 3. Calculate Loss for this row
    for (size_t c = 0; c < prediction.cols; ++c) {
      float t = target(r, c);
      if (t > 0.0f) {  // Optimization for sparse targets
        // log_softmax = (z - max) - log(sum_exp)
        float log_softmax = (prediction(r, c) - max_val) - log_sum_exp;
        total_loss += -t * log_softmax;
      }
    }
  }

  return total_loss / static_cast<float>(prediction.rows);
}

Matrix CrossEntropyWithLogitsLoss::gradient(const Matrix& prediction,
                                            const Matrix& target) {
  // dL/dZ = Softmax(Z) - Target
  // This is ALWAYS bounded between -1 and 1. No explosions.

  Matrix grad(prediction.rows, prediction.cols);
  float batch_scale = 1.0f / static_cast<float>(prediction.rows);
  int n_rows = prediction.rows;
  int n_cols = prediction.cols;

  float* G = grad.rawData();
  const float* P = prediction.rawData();
  const float* T = target.rawData();

#pragma omp parallel for
  for (int r = 0; r < n_rows; ++r) {
    const float* pred_row = &P[r * n_cols];
    const float* targ_row = &T[r * n_cols];
    float* grad_row = &G[r * n_cols];

    // 1. Softmax Calculation
    float max_val = pred_row[0];
    for (int c = 1; c < n_cols; ++c) {
      if (pred_row[c] > max_val) max_val = pred_row[c];
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < n_cols; ++c) {
      sum_exp += std::exp(pred_row[c] - max_val);
    }

    // 2. Gradient Calculation
    float inv_sum = 1.0f / sum_exp;
    for (int c = 0; c < n_cols; ++c) {
      float softmax_p = std::exp(pred_row[c] - max_val) * inv_sum;
      float t = targ_row[c];
      // Result = (P - T) / BatchSize
      grad_row[c] = (softmax_p - t) * batch_scale;
    }
  }
  return grad;
}

}  // namespace loss
}  // namespace nn
}  // namespace talawa_ai
