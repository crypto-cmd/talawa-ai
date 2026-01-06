#include "talawa/neuralnetwork/Conv2DLayer.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>

namespace talawa::nn {

using namespace core;

// Default constructor for load-time construction
Conv2DLayer::Conv2DLayer()
    : depth(0),
      input_height(0),
      input_width(0),
      filters(0),
      kernel_size(0),
      stride(0),
      padding(0),
      output_height(0),
      output_width(0) {}

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

void Conv2DLayer::save(std::ostream& out) const {
  // Save configuration
  out.write(reinterpret_cast<const char*>(&depth), sizeof(int));
  out.write(reinterpret_cast<const char*>(&input_height), sizeof(int));
  out.write(reinterpret_cast<const char*>(&input_width), sizeof(int));
  out.write(reinterpret_cast<const char*>(&filters), sizeof(int));
  out.write(reinterpret_cast<const char*>(&kernel_size), sizeof(int));
  out.write(reinterpret_cast<const char*>(&stride), sizeof(int));
  out.write(reinterpret_cast<const char*>(&padding), sizeof(int));

  int act = static_cast<int>(activation.type);
  out.write(reinterpret_cast<const char*>(&act), sizeof(int));

  // Save kernels
  size_t k_rows = kernels.rows;
  size_t k_cols = kernels.cols;
  out.write(reinterpret_cast<const char*>(&k_rows), sizeof(size_t));
  out.write(reinterpret_cast<const char*>(&k_cols), sizeof(size_t));
  size_t k_count = k_rows * k_cols;
  out.write(reinterpret_cast<const char*>(kernels.rawData()),
            k_count * sizeof(float));

  // Save biases
  size_t b_rows = biases.rows;
  size_t b_cols = biases.cols;
  out.write(reinterpret_cast<const char*>(&b_rows), sizeof(size_t));
  out.write(reinterpret_cast<const char*>(&b_cols), sizeof(size_t));
  size_t b_count = b_rows * b_cols;
  out.write(reinterpret_cast<const char*>(biases.rawData()),
            b_count * sizeof(float));
}

void Conv2DLayer::load(std::istream& in_stream) {
  in_stream.read(reinterpret_cast<char*>(&depth), sizeof(int));
  in_stream.read(reinterpret_cast<char*>(&input_height), sizeof(int));
  in_stream.read(reinterpret_cast<char*>(&input_width), sizeof(int));
  in_stream.read(reinterpret_cast<char*>(&filters), sizeof(int));
  in_stream.read(reinterpret_cast<char*>(&kernel_size), sizeof(int));
  in_stream.read(reinterpret_cast<char*>(&stride), sizeof(int));
  in_stream.read(reinterpret_cast<char*>(&padding), sizeof(int));

  // Activation
  int act;
  in_stream.read(reinterpret_cast<char*>(&act), sizeof(int));
  this->activation = Activation(static_cast<Activation::Type>(act));

  // Recompute output sizes
  output_height = (input_height - kernel_size + 2 * padding) / stride + 1;
  output_width = (input_width - kernel_size + 2 * padding) / stride + 1;

  // Read kernels
  size_t k_rows, k_cols;
  in_stream.read(reinterpret_cast<char*>(&k_rows), sizeof(size_t));
  in_stream.read(reinterpret_cast<char*>(&k_cols), sizeof(size_t));
  this->kernels = Matrix(static_cast<int>(k_rows), static_cast<int>(k_cols));
  size_t k_count = k_rows * k_cols;
  in_stream.read(reinterpret_cast<char*>(this->kernels.rawData()),
                 k_count * sizeof(float));

  // Read biases
  size_t b_rows, b_cols;
  in_stream.read(reinterpret_cast<char*>(&b_rows), sizeof(size_t));
  in_stream.read(reinterpret_cast<char*>(&b_cols), sizeof(size_t));
  this->biases = Matrix(static_cast<int>(b_rows), static_cast<int>(b_cols));
  size_t b_count = b_rows * b_cols;
  in_stream.read(reinterpret_cast<char*>(this->biases.rawData()),
                 b_count * sizeof(float));

  // Recreate gradients
  int fan_in = kernel_size * kernel_size * depth;
  this->kernels_grad = Matrix(fan_in, filters);
  this->biases_grad = Matrix(1, filters);
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
    // Per-thread pointers to avoid per-access overhead
    float* res_row =
        result.rawData() +
        static_cast<long long>(b) * (depth * input_height * input_width);
    const float* col_buf = col_matrix.rawData();
    int col_cols = col_matrix.cols;
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

                // Removed atomic: parallelized over batch so writes are
                // thread-local
                float grad_val =
                    col_buf[static_cast<long long>(row_idx) * col_cols +
                            col_idx];
                res_row[flat_idx] += grad_val;
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
  auto t_gemm = std::chrono::steady_clock::now();
  // Transpose kernels into a reusable buffer, then multiply using B^T path
  kernels.transpose(kernels_T);
  Matrix output_flat;
  cols.dotWithBTransposed(kernels_T, output_flat);
  profiling_gemm += std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::steady_clock::now() - t_gemm)
                        .count();

  // 3. Reshape Output to (Batch, Filters*OH*OW) and Add Bias
  // Since we don't have a 3D Tensor class, we keep it flattened (Batch,
  // F*OH*OW) However, the biases need to be added per-filter. Our flat output
  // is organized as pixel-major, then batch. We need to carefully add biases.

  // Simple approach: Iterate and add bias
  int pixels = output_height * output_width;
  auto t_bias = std::chrono::steady_clock::now();
  // Use optimized broadcast add (vectorized + parallel)
  output_flat += biases;
  profiling_bias += std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::steady_clock::now() - t_bias)
                        .count();

  // 4. Activation
  z_cache = output_flat;  // Store Z for backprop
  auto t_act = std::chrono::steady_clock::now();
  Matrix a = activation.apply(z_cache);
  profiling_activation +=
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t_act)
          .count();

  if (is_training) a_cache = a;

  // Note: The output is strictly 2D (Rows=Batch*Pix, Cols=Filters)
  // If the next layer expects (Batch, Flattened), we need to reshape.
  // But our 'im2col' flattened the batch dimension too.
  // We must reshape back to (Batch, TotalFlatSize).

  auto t_reshape = std::chrono::steady_clock::now();
  Matrix final_output(input.rows, filters * pixels);

  // Parallelize over batch and filter to improve cache behaviour
#pragma omp parallel for collapse(2)
  for (int b = 0; b < input.rows; ++b) {
    for (int f = 0; f < filters; ++f) {
      float* dest =
          final_output.rawData() + (b * (filters * pixels) + f * pixels);
      const float* src = a.rawData() + ((b * pixels) * filters + f);
      for (int p = 0; p < pixels; ++p) dest[p] = src[p * filters];
    }
  }

  profiling_reshape +=
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t_reshape)
          .count();

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
  auto t_actback = std::chrono::steady_clock::now();
  activation.backprop(a_cache, dZ_flat,
                      dZ);  // dZ is now (Batch*Pixels, Filters)
  profiling_act_backprop +=
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t_actback)
          .count();

  // 2. Gradients w.r.t Weights (Kernels)
  // dW = Col_X^T * dZ
  // (K*K*D, Batch*Pixels) . (Batch*Pixels, Filters) -> (K*K*D, Filters)
  // Reuse pre-allocated transpose buffer
  col_cache.transpose(this->col_T);
  auto t_kgrad = std::chrono::steady_clock::now();
  this->col_T.dot(dZ, kernels_grad);
  profiling_kernels_grad +=
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t_kgrad)
          .count();

  // 3. Gradients w.r.t Biases
  // Sum dZ across all batches and pixels
  biases_grad.fill(0.0f);
  auto t_bgrad = std::chrono::steady_clock::now();
  dZ.sumRows(biases_grad);
  profiling_bias_grad +=
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t_bgrad)
          .count();

  // 4. Gradients w.r.t Input (dX)
  // dCol = dZ * W^T
  auto t_dcol = std::chrono::steady_clock::now();
  Matrix dCol;
  // Use kernels directly as B_T here (rows = fan_in, cols = filters) so that
  // dotWithBTransposed sees B_T.cols == dZ.cols
  dZ.dotWithBTransposed(kernels, dCol);
  profiling_dcol += std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::steady_clock::now() - t_dcol)
                        .count();

  // 5. Col2Im (Reshape back to image)
  auto t_col2im = std::chrono::steady_clock::now();
  Matrix res = col2im(dCol);
  profiling_col2im += std::chrono::duration_cast<std::chrono::duration<double>>(
                          std::chrono::steady_clock::now() - t_col2im)
                          .count();
  return res;
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

}  // namespace talawa::nn
