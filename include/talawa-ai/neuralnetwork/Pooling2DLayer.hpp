#pragma once
#include "talawa-ai/neuralnetwork/Layer.hpp"

namespace talawa_ai {
namespace nn {

enum class PoolingType { MAX, AVERAGE };

struct Pooling2DLayerConfig {
  PoolingType type = PoolingType::MAX;
  int pool_size = 2;
  int stride = 2;
};

class Pooling2DLayer : public Layer {
 private:
  // Configuration
  PoolingType type;
  int depth, input_height, input_width;
  int pool_size, stride;
  int output_height, output_width;

  // Cache for Backprop
  // MAX: Stores the flat index of the "winning" pixel
  std::vector<std::vector<int>> max_indices_cache;

  // AVERAGE: We don't need a complex cache, just the input shape
  // (We use input dimensions to allocate dX)

 public:
  Pooling2DLayer(int depth, int height, int width,
                 PoolingType type = PoolingType::MAX, int pool_size = 2,
                 int stride = 2);

  core::Matrix forward(const core::Matrix& input,
                       bool is_training = true) override;
  core::Matrix backward(const core::Matrix& outputGradients) override;

  std::vector<core::Matrix*> getParameters() override { return {}; }
  std::vector<core::Matrix*> getParameterGradients() override { return {}; }

  Shape getOutputShape() const override;
  std::string info() const override;

  void save(std::ostream& out) const override {}
  void load(std::istream& in) override {}
};

}  // namespace nn
}  // namespace talawa_ai
