#pragma once
#include <vector>

#include "talawa/neuralnetwork/Layer.hpp"

namespace talawa {
namespace nn {

struct Conv2DLayerConfig {
  int filters;
  int kernel_size;
  int stride = 1;
  int padding = 0;  // 0 = Valid, >0 = Zero Padding
  Initializer init = Initializer::GLOROT_UNIFORM;
  Activation::Type act = Activation::RELU;
};

class Conv2DLayer : public ILayer {
 private:
  // Dimensions
  int depth, input_height, input_width;
  int filters, kernel_size, stride, padding;
  int output_height, output_width;

  // Parameters
  core::Matrix kernels;  // Shape: (kernel_size * kernel_size * depth, filters)
  core::Matrix biases;   // Shape: (1, filters)

  // Cache for Backprop
  core::Matrix input_cache;  // Original Input
  core::Matrix col_cache;    // Input transformed via Im2Col
  core::Matrix z_cache;      // Pre-activation output
  core::Matrix a_cache;      // Activated output

  // Gradients
  core::Matrix kernels_grad;
  core::Matrix biases_grad;
  core::Matrix input_grad;  // dL/dX

  // Cached transposed buffers to avoid frequent allocations
  core::Matrix kernels_T;  // Transposed kernels (filters x K*K*D)
  core::Matrix col_T;      // Transposed columns when needed

  // Helpers
  core::Matrix im2col(const core::Matrix& input);
  core::Matrix col2im(const core::Matrix& col_matrix);

 public:
  Conv2DLayer();

  // Profiling accumulators (seconds) to help narrow hotspots
  double profiling_im2col = 0.0;
  double profiling_gemm = 0.0;
  double profiling_bias = 0.0;
  double profiling_activation = 0.0;
  double profiling_reshape = 0.0;

  double profiling_col2im = 0.0;
  double profiling_kernels_grad = 0.0;
  double profiling_bias_grad = 0.0;
  double profiling_dcol = 0.0;
  double profiling_act_backprop = 0.0;
  Conv2DLayer(int depth, int height, int width, int filters, int kernel_size,
              int stride = 1, int padding = 0,
              Initializer init = Initializer::GLOROT_UNIFORM,
              Activation act = Activation::RELU);

  core::Matrix forward(const core::Matrix& input,
                       bool is_training = true) override;
  core::Matrix backward(const core::Matrix& outputGradients) override;

  std::vector<core::Matrix*> getParameters() override;
  std::vector<core::Matrix*> getParameterGradients() override;

  Shape getOutputShape() const override;
  std::string info() const override;

  void save(std::ostream&) const override;
  void load(std::istream&) override;
};

}  // namespace nn
}  // namespace talawa
