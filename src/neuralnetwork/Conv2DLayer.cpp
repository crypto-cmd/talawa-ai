#include "talawa-ai/neuralnetwork/Conv2DLayer.hpp"

#include <cmath>
#include <iostream>
#include <sstream>

namespace talawa_ai::nn {

using namespace core;

Conv2DLayer::Conv2DLayer(int d, int h, int w, int f, int k, int s, int p,
                         Initializer init, Activation act)
    : depth(d),
      input_height(h),
      input_width(w),
      filters(f),
      kernel_size(k),
      stride(s),
      padding(p) {
  this->initializer = init;
  this->activation = act;

  // 1. Calculate Output Dimensions
  output_height = (input_height - kernel_size + 2 * padding) / stride + 1;
  output_width = (input_width - kernel_size + 2 * padding) / stride + 1;

  // 2. Initialize Kernels
  // Shape: (Fan-In, Fan-Out) -> (K*K*D, Filters)
  int fan_in = kernel_size * kernel_size * depth;
  kernels = Matrix(fan_in, filters);
  initializer.apply(kernels);

  // 3. Initialize Biases (1, Filters) - Broadcasted later
  biases = Matrix(1, filters);
  Initializer(Initializer::ZEROS).apply(biases);

  // 4. Pre-allocate Gradients
  kernels_grad = Matrix(fan_in, filters);
  biases_grad = Matrix(1, filters);
}

// --- IM2COL: The Heart of High-Performance Conv2D ---
// Transforms input (Batch, D*H*W) -> (Batch*OutH*OutW, K*K*D)
Matrix Conv2DLayer::im2col(const Matrix& input) {
  int batch_size = input.rows;
  int col_rows = batch_size * output_height * output_width;
  int col_cols = kernel_size * kernel_size * depth;

  Matrix result(col_rows, col_cols);

// Note: This loop structure is complex. In a production library,
// you might parallelize the batch loop with OpenMP.
#pragma omp parallel for
  for (int b = 0; b < batch_size; ++b) {
    for (int y = 0; y < output_height; ++y) {
      for (int x = 0; x < output_width; ++x) {
        // Calculate the row index in the Result Matrix
        int row_idx = b * (output_height * output_width) + y * output_width + x;

        // Calculate the starting pixel in the Input Image
        int in_y_origin = y * stride - padding;
        int in_x_origin = x * stride - padding;

        int col_idx = 0;

        // Loop over the kernel window (Depth, KernelY, KernelX)
        for (int c = 0; c < depth; ++c) {
          for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
              int in_y = in_y_origin + ky;
              int in_x = in_x_origin + kx;

              float val = 0.0f;
              // Boundary Check (Padding)
              if (in_y >= 0 && in_y < input_height && in_x >= 0 &&
                  in_x < input_width) {
                // Flattened Index: D*H*W
                int flat_idx = (c * input_height * input_width) +
                               (in_y * input_width) + in_x;
                val = input(b, flat_idx);
              }

              result(row_idx, col_idx) = val;
              col_idx++;
            }
          }
        }
      }
    }
  }
  return result;
}

// --- COL2IM: For Backpropagation ---
// Transforms Gradient Cols -> Gradient Input Image
Matrix Conv2DLayer::col2im(const Matrix& col_matrix) {
  int batch_size = input_cache.rows;
  Matrix result = Matrix::zeros(batch_size, depth * input_height * input_width);

// Similar logic to im2col but accumulating gradients
#pragma omp parallel for
  for (int b = 0; b < batch_size; ++b) {
    for (int y = 0; y < output_height; ++y) {
      for (int x = 0; x < output_width; ++x) {
        int row_idx = b * (output_height * output_width) + y * output_width + x;
        int in_y_origin = y * stride - padding;
        int in_x_origin = x * stride - padding;
        int col_idx = 0;

        for (int c = 0; c < depth; ++c) {
          for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
              int in_y = in_y_origin + ky;
              int in_x = in_x_origin + kx;

              if (in_y >= 0 && in_y < input_height && in_x >= 0 &&
                  in_x < input_width) {
                int flat_idx = (c * input_height * input_width) +
                               (in_y * input_width) + in_x;

                // Atomic add needed if multiple threads map to same pixel
                // But simplified here: im2col overlaps, so we accumulate
                float grad_val = col_matrix(row_idx, col_idx);
#pragma omp atomic
                result(b, flat_idx) += grad_val;
              }
              col_idx++;
            }
          }
        }
      }
    }
  }
  return result;
}

Matrix Conv2DLayer::forward(const Matrix& input, bool is_training) {
  if (is_training) {
    input_cache = input;
  }

  // 1. Im2Col: Reshape input into columns
  Matrix cols = im2col(input);
  if (is_training) {
    col_cache = cols;
  }

  // 2. Convolution via GEMM (Cols * Kernels)
  // (Batch*OH*OW, K*K*D) . (K*K*D, Filters) -> (Batch*OH*OW, Filters)
  Matrix output_flat = cols.dot(kernels);

  // 3. Reshape Output to (Batch, Filters*OH*OW) and Add Bias
  // Since we don't have a 3D Tensor class, we keep it flattened (Batch,
  // F*OH*OW) However, the biases need to be added per-filter. Our flat output
  // is organized as pixel-major, then batch. We need to carefully add biases.

  // Simple approach: Iterate and add bias
  int pixels = output_height * output_width;
  output_flat.apply([&](int r, int c, float v) {
    return v + biases(0, c);  // Broadcast bias to every pixel
  });

  // 4. Activation
  z_cache = output_flat;  // Store Z for backprop
  Matrix a = activation.apply(z_cache);

  if (is_training) a_cache = a;

  // Note: The output is strictly 2D (Rows=Batch*Pix, Cols=Filters)
  // If the next layer expects (Batch, Flattened), we need to reshape.
  // But our 'im2col' flattened the batch dimension too.
  // We must reshape back to (Batch, TotalFlatSize).

  Matrix final_output(input.rows, filters * pixels);
  int flat_row = 0;
  for (int b = 0; b < input.rows; ++b) {
    for (int p = 0; p < pixels; ++p) {
      for (int f = 0; f < filters; ++f) {
        // Logic to map (Batch*Pixels, Filters) back to (Batch, Filters*Pixels)
        // This can be optimized with a copy kernel
        final_output(b, f * pixels + p) = a(flat_row, f);
      }
      flat_row++;
    }
  }

  return final_output;
}

Matrix Conv2DLayer::backward(const Matrix& outputGradients) {
  // outputGradients comes in as (Batch, Filters*Pixels).
  // We need to reshape it back to (Batch*Pixels, Filters) for im2col compatible
  // math.
  int pixels = output_height * output_width;
  Matrix dZ_flat(outputGradients.rows * pixels, filters);

  int flat_row = 0;
  for (int b = 0; b < outputGradients.rows; ++b) {
    for (int p = 0; p < pixels; ++p) {
      for (int f = 0; f < filters; ++f) {
        dZ_flat(flat_row, f) = outputGradients(b, f * pixels + p);
      }
      flat_row++;
    }
  }

  // 1. Activation Derivative
  // Reconstruct Z_cache if needed, or use a_cache depending on activation
  Matrix dZ;
  activation.backprop(a_cache, dZ_flat,
                      dZ);  // dZ is now (Batch*Pixels, Filters)

  // 2. Gradients w.r.t Weights (Kernels)
  // dW = Col_X^T * dZ
  // (K*K*D, Batch*Pixels) . (Batch*Pixels, Filters) -> (K*K*D, Filters)
  Matrix col_T = col_cache.transpose();
  col_T.dot(dZ, kernels_grad);

  // 3. Gradients w.r.t Biases
  // Sum dZ across all batches and pixels
  biases_grad.fill(0.0f);
  dZ.sumRows(biases_grad);

  // 4. Gradients w.r.t Input (dX)
  // dCol = dZ * W^T
  Matrix kernels_T = kernels.transpose();
  Matrix dCol = dZ.dot(kernels_T);

  // 5. Col2Im (Reshape back to image)
  return col2im(dCol);
}

std::vector<Matrix*> Conv2DLayer::getParameters() {
  return {&kernels, &biases};
}

std::vector<Matrix*> Conv2DLayer::getParameterGradients() {
  return {&kernels_grad, &biases_grad};
}

Shape Conv2DLayer::getOutputShape() const {
  return {filters, output_height, output_width};
}

std::string Conv2DLayer::info() const {
  std::stringstream ss;
  ss << "Conv2D Layer [" << input_height << "x" << input_width << "x" << depth
     << "] -> [" << output_height << "x" << output_width << "x" << filters
     << "] k=" << kernel_size << " s=" << stride << " p=" << padding;
  return ss.str();
}

}  // namespace talawa_ai::nn
