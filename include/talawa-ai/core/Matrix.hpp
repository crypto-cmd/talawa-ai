#pragma once
#include <functional>
#include <initializer_list>
#include <vector>

#include "talawa-ai/core/Error.hpp"

namespace talawa_ai {
namespace core {
#define THROW_MATRIX_ERROR(msg) THROW_TALAWA_AI_ERROR("Matrix", msg)

class Matrix {
 public:
  Matrix(int rows, int cols);
  Matrix() : rows(0), cols(0) {}  // Initialize to zero instead of undefined
  Matrix(std::initializer_list<std::initializer_list<float>> values);
  Matrix(const Matrix &) = default;
  Matrix(Matrix &&) = default;
  void print(int decimals = 2) const;

  // Operation overloads
  Matrix operator+(const Matrix &other) const;
  Matrix operator-(const Matrix &other) const;
  Matrix operator*(const float scalar) const;
  Matrix &operator=(const Matrix &other);
  Matrix &operator=(Matrix &&) = default;
  bool operator==(const Matrix &other) const;

  Matrix &operator+=(const Matrix &other);

  // Element access
  float operator()(int row, int col) const;
  float &operator()(int row, int col);

  // Dot and hadamard products
  Matrix dot(const Matrix &other) const;
  void dot(const Matrix &other, Matrix &out) const;
  Matrix hadamard(const Matrix &other) const;
  Matrix addVector(const Matrix &vector) const;
  void sumRows(Matrix &out) const;
  // Returns a new Matrix containing rows from start_row (inclusive) to end_row
  // (exclusive)
  Matrix slice(int start_row, int end_row) const;

  // Other utilities
  Matrix transpose() const;
  void transpose(Matrix &out) const;
  void fill(float value);

  static Matrix identity(int size);
  static Matrix zeros(int rows, int cols);
  static Matrix ones(int rows, int cols);
  static Matrix random(int rows, int cols);

  Matrix &operator=(const std::vector<std::vector<float>> &inputData);

  size_t rows;
  size_t cols;

  // Higher Order Functions
  // Applies a function to each element of the matrix
  void apply(std::function<float(int row, int col, float)> func) {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        (*this)(i, j) = func(i, j, (*this)(i, j));
      }
    }
  }

  Matrix map(std::function<float(int row, int col, float)> func) const {
    Matrix result(rows, cols);
    result.apply([&](int i, int j, float val) { return func(i, j, val); });
    return result;
  }

  // Reduce function to a single value
  template <typename T>
  T reduce(std::function<T(T accumulator, int row, int col, float value)> func,
           T initial) const {
    T accumulator = initial;
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        accumulator = func(accumulator, i, j, (*this)(i, j));
      }
    }
    return accumulator;
  }
  // Access to raw data for performance-critical operations
  float *rawData() { return data.data(); }
  const float *rawData() const { return data.data(); }

 private:
  std::vector<float> data;
};

}  // namespace core
}  // namespace talawa_ai
