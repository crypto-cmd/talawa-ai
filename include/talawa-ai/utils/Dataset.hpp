#pragma once
#include <vector>

#include "talawa-ai/core/Matrix.hpp"

namespace talawa_ai {
namespace utils {
using namespace talawa_ai::core;
class Dataset {
 public:
  std::vector<int> indices;
  Matrix features;  // Feature matrix (All samples)
  Matrix labels;    // Label matrix (One-Hot Encoded)

  Dataset() = default;
  void shuffle();
  void splice(size_t start, size_t end, Matrix& features, Matrix& labels);
  size_t size() const { return features.rows; }
};

}  // namespace utils
}  // namespace talawa_ai
