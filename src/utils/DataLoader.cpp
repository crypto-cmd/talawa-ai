#include "talawa/utils/DataLoader.hpp"

#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "talawa/utils/Dataset.hpp"

namespace talawa {
namespace utils {

using namespace talawa::core;

Dataset DataLoader::loadCSV(const std::string& path, int label_index,
                            int num_classes, float scale, bool skip_header) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("DataLoader: Could not open file " + path);
  }

  std::vector<std::vector<float>> x_data;
  std::vector<std::vector<float>> y_data;

  std::string line;
  if (skip_header) {
    std::getline(file, line);
  }

  std::cout << "[DataLoader] Loading " << path << "..." << std::endl;

  int dropped_rows = 0;
  size_t expected_cols = 0;

  while (std::getline(file, line)) {
    if (line.empty()) continue;

    std::stringstream ss(line);
    std::string cell;
    std::vector<float> row_x;
    std::vector<float> row_y(num_classes, 0.0f);  // Init One-Hot vector

    int col_idx = 0;
    while (std::getline(ss, cell, ',')) {
      try {
        // Handle potential carriage returns in Linux/Windows file mix
        if (!cell.empty() && cell.back() == '\r') {
          cell.pop_back();
        }

        float val = std::stof(cell);

        if (col_idx == label_index) {
          // Process Label (One-Hot Encode)
          int label = static_cast<int>(val);
          if (label >= 0 && label < num_classes) {
            row_y[label] = 1.0f;
          }
        } else {
          // Process Feature (Normalize)
          row_x.push_back(val / scale);
        }
        col_idx++;
      } catch (...) {
        // If parsing fails (e.g. empty string ",,"), we just skip this cell.
        // This causes the row size to shrink, which we catch below.
      }
    }

    // --- VALIDATION LOGIC ---
    if (!row_x.empty()) {
      if (x_data.empty()) {
        // First valid row sets the standard
        expected_cols = row_x.size();
        x_data.push_back(row_x);
        y_data.push_back(row_y);
      } else {
        // Subsequent rows must match the first row's width
        if (row_x.size() == expected_cols) {
          x_data.push_back(row_x);
          y_data.push_back(row_y);
        } else {
          // Drop jagged rows
          dropped_rows++;
        }
      }
    }
  }

  std::cout << "[DataLoader] Loaded " << x_data.size() << " samples.\n";
  if (dropped_rows > 0) {
    std::cout << "[DataLoader] WARNING: Dropped " << dropped_rows
              << " rows due to inconsistent column counts (jagged data).\n";
  }

  if (x_data.empty()) {
    throw std::runtime_error("DataLoader: No valid data loaded.");
  }

  // Convert vectors to Matrix
  Dataset dataset;
  // We rely on the implicit conversion or assignment we defined earlier
  dataset.features = x_data;
  dataset.labels = y_data;

  dataset.indices.resize(dataset.features.rows);
  std::iota(dataset.indices.begin(), dataset.indices.end(), 0);

  return dataset;
}

}  // namespace utils
}  // namespace talawa
