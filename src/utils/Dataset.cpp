#include "talawa-ai/utils/Dataset.hpp"

namespace talawa_ai {
namespace utils {
using namespace talawa_ai::core;
void Dataset::shuffle() {
  // Simple Fisher-Yates shuffle
  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(indices.begin(), indices.end(), g);
}

void Dataset::splice(size_t start, size_t end, Matrix& feature_batch,
                     Matrix& label_batch) {
  if (start < 0 || end > size() || start >= end) {
    throw std::out_of_range("Dataset::splice indices out of bounds");
  }
  size_t batch_size = end - start;
  feature_batch = Matrix(batch_size, features.cols);
  label_batch = Matrix(batch_size, labels.cols);

  for (size_t i = 0; i < batch_size; ++i) {
    size_t idx = indices[start + i];
    for (size_t j = 0; j < features.cols; ++j) {
      feature_batch(i, j) = features(idx, j);
    }
    for (size_t j = 0; j < labels.cols; ++j) {
      label_batch(i, j) = labels(idx, j);
    }
  }
}

}  // namespace utils
}  // namespace talawa_ai
