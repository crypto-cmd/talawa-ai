#include "talawa-ai/neuralnetwork/Pooling2DLayer.hpp"

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>

namespace talawa_ai {
namespace nn {

using namespace core;

Pooling2DLayer::Pooling2DLayer(int d, int h, int w, PoolingType type, int ps,
                               int s)
    : depth(d),
      input_height(h),
      input_width(w),
      type(type),
      pool_size(ps),
      stride(s) {
  output_height = (input_height - pool_size) / stride + 1;
  output_width = (input_width - pool_size) / stride + 1;
  this->activation = Activation::LINEAR;
}

core::Matrix Pooling2DLayer::forward(const core::Matrix& input,
                                     bool is_training) {
  int batch_size = input.rows;
  int output_pixels = output_height * output_width;
  int output_cols = depth * output_pixels;

  Matrix output(batch_size, output_cols);

  if (is_training && type == PoolingType::MAX) {
    max_indices_cache.assign(batch_size, std::vector<int>(output_cols, -1));
  }

  // Pre-calculate scaling factor for Average pooling
  float avg_scale = 1.0f / (pool_size * pool_size);

#pragma omp parallel for
  for (int b = 0; b < batch_size; ++b) {
    for (int d = 0; d < depth; ++d) {
      for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
          int start_y = y * stride;
          int start_x = x * stride;
          int end_y = std::min(start_y + pool_size, input_height);
          int end_x = std::min(start_x + pool_size, input_width);

          // --- LOGIC SPLIT ---
          if (type == PoolingType::MAX) {
            float max_val = -std::numeric_limits<float>::infinity();
            int max_idx = -1;

            for (int wy = start_y; wy < end_y; ++wy) {
              for (int wx = start_x; wx < end_x; ++wx) {
                int flat_idx =
                    (d * input_height * input_width) + (wy * input_width) + wx;
                float val = input(b, flat_idx);
                if (val > max_val) {
                  max_val = val;
                  max_idx = flat_idx;
                }
              }
            }
            // Write Output
            int out_idx =
                (d * output_height * output_width) + (y * output_width) + x;
            output(b, out_idx) = max_val;
            if (is_training) max_indices_cache[b][out_idx] = max_idx;

          } else if (type == PoolingType::AVERAGE) {
            float sum = 0.0f;
            for (int wy = start_y; wy < end_y; ++wy) {
              for (int wx = start_x; wx < end_x; ++wx) {
                int flat_idx =
                    (d * input_height * input_width) + (wy * input_width) + wx;
                sum += input(b, flat_idx);
              }
            }
            // Write Output
            int out_idx =
                (d * output_height * output_width) + (y * output_width) + x;
            output(b, out_idx) = sum * avg_scale;
          }
        }
      }
    }
  }
  return output;
}

core::Matrix Pooling2DLayer::backward(const core::Matrix& outputGradients) {
  int batch_size = outputGradients.rows;
  int input_cols = depth * input_height * input_width;
  Matrix dX = Matrix::zeros(batch_size, input_cols);

  // Pre-calculate scaling factor for Average pooling gradients
  float avg_grad_scale = 1.0f / (pool_size * pool_size);

#pragma omp parallel for
  for (int b = 0; b < batch_size; ++b) {
    // Iterate over the OUTPUT gradient (since it maps to a window in input)
    for (int d = 0; d < depth; ++d) {
      for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
          int out_idx =
              (d * output_height * output_width) + (y * output_width) + x;
          float grad = outputGradients(b, out_idx);

          if (type == PoolingType::MAX) {
            // Route gradient ONLY to the max pixel
            int max_idx = max_indices_cache[b][out_idx];
            if (max_idx != -1) {
// We use atomic add or accumulation if windows overlap
// (though typically they don't in pooling).
// Since this loops over 'out_idx', we can write directly if stride >=
// pool_size. For safety with generic stride, we accumulate.
#pragma omp atomic
              dX(b, max_idx) += grad;
            }

          } else if (type == PoolingType::AVERAGE) {
            // Distribute gradient EQUALLY to all pixels in window
            int start_y = y * stride;
            int start_x = x * stride;
            int end_y = std::min(start_y + pool_size, input_height);
            int end_x = std::min(start_x + pool_size, input_width);

            float distributed_grad = grad * avg_grad_scale;

            for (int wy = start_y; wy < end_y; ++wy) {
              for (int wx = start_x; wx < end_x; ++wx) {
                int flat_idx =
                    (d * input_height * input_width) + (wy * input_width) + wx;

#pragma omp atomic
                dX(b, flat_idx) += distributed_grad;
              }
            }
          }
        }
      }
    }
  }
  return dX;
}

Shape Pooling2DLayer::getOutputShape() const {
  return {depth, output_height, output_width};
}

std::string Pooling2DLayer::info() const {
  std::stringstream ss;
  std::string typeStr = (type == PoolingType::MAX) ? "MAX" : "AVG";
  ss << "Pooling Layer [" << typeStr << "] " << input_height << "x"
     << input_width << " -> " << output_height << "x" << output_width;
  return ss.str();
}

}  // namespace nn
}  // namespace talawa_ai
