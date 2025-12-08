#pragma once
#include <string>
#include <vector>

#include "talawa-ai/core/Matrix.hpp"
#include "talawa-ai/utils/Dataset.hpp"

namespace talawa_ai {
namespace utils {

class DataLoader {
 public:
  /**
   * Loads a CSV file into a Dataset.
   * * @param path Path to the CSV file.
   * @param label_index Column index of the label (e.g., 0 for MNIST, -1 for
   * last column).
   * @param num_classes Number of output classes (for One-Hot Encoding).
   * @param scale Factor to divide input values by (e.g., 255.0f for images).
   * @param skip_header Whether to skip the first row.
   */
  static Dataset loadCSV(const std::string& path, int label_index,
                         int num_classes, float scale = 1.0f,
                         bool skip_header = true);
};

}  // namespace utils
}  // namespace talawa_ai
