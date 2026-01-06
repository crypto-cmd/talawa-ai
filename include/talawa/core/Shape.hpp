#pragma once
#include <array>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace talawa::core {

constexpr int MAX_STACK_DIMS = 6;  // Covers almost all RL/DL cases
/**
 * @brief Represents the shape of a multi-dimensional array/tensor
 * @note Defined from Nth dimension to 0th dimension in order
 * to match typical tensor shape conventions (e.g., [Depth, Height, Width])
 */
struct Shape {
 private:
  // Tiny buffer for stack storage
  size_t stack_dims_[MAX_STACK_DIMS];
  // Fallback for crazy shapes (like >6 dimensions)
  std::vector<size_t> heap_dims_;

  int rank_ = 0;
  bool use_heap_ = false;

 public:
  // Constructors
  Shape() = default;

  Shape(std::initializer_list<int> dims) { setup(dims.begin(), dims.size()); }

  Shape(const std::vector<int>& dims) { setup(dims.data(), dims.size()); }

  void setup(const int* data, size_t size) {
    rank_ = static_cast<int>(size);
    if (rank_ <= MAX_STACK_DIMS) {
      use_heap_ = false;
      for (int i = 0; i < rank_; ++i) stack_dims_[i] = data[i];
    } else {
      use_heap_ = true;
      heap_dims_.assign(data, data + size);
    }
  }

  // Accessors
  int rank() const { return rank_; }

  int operator[](int index) const {
    if (use_heap_) return heap_dims_[index];
    return stack_dims_[index];
  }

  // Utility: Total number of elements (Volume)
  int size() const {
    if (rank_ == 0)
      return 0;  // Or 1 for scalar, depends on your math preference
    int product = 1;
    for (int i = 0; i < rank_; ++i) product *= (*this)[i];
    return product;
  }

  // Equality check
  bool operator==(const Shape& other) const {
    if (rank_ != other.rank_) return false;
    for (int i = 0; i < rank_; ++i) {
      if ((*this)[i] != other[i]) return false;
    }
    return true;
  }
};

}  // namespace talawa::core
