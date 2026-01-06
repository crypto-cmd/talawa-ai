#include "talawa/core/Matrix.hpp"

#include <immintrin.h>  // Required for AVX2
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

#define PARALLEL_FOR _Pragma("omp parallel for")
using namespace talawa::core;

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
  data.resize(rows * cols, 0.0f);
}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> values)
    : rows(values.size()), cols(values.begin()->size()) {
  if (rows == 0 || cols == 0) {
    throw std::invalid_argument("Matrix dimensions must be greater than 0");
  }

  // Check that every row has the same number of columns
  for (const auto& row : values) {
    if (row.size() != cols) {
      throw std::invalid_argument(
          "All rows must have the same number of columns");
    }
  }

  data.resize(rows * cols, 0.0f);

  int r = 0;
  for (const auto& row : values) {
    int c = 0;
    for (float value : row) {
      (*this)(r, c++) = value;
    }
    ++r;
  }
}

void Matrix::print(int decimals) const {
  // 1. Calculate the maximum width needed for any number in the matrix
  size_t max_width = 0;
  for (const auto& val : data) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(decimals) << val;
    std::string s = ss.str();
    if (s.length() > max_width) {
      max_width = s.length();
    }
  }

  // Add a little padding (1 space)
  int width = max_width + 1;

  std::cout << "Matrix (" << rows << "x" << cols << "):" << std::endl;
  std::cout << "[" << std::endl;

  for (size_t i = 0; i < rows; ++i) {
    std::cout << "  [";  // Indent for the row
    for (size_t j = 0; j < cols; ++j) {
      // Print the number with the calculated dynamic width
      std::cout << std::setw(width) << std::fixed << std::setprecision(decimals)
                << (*this)(i, j);

      // Print a comma unless it's the last column
      if (j < cols - 1) std::cout << ",";
    }
    std::cout << "  ]";

    // Print a comma unless it's the last row
    if (i < rows - 1) std::cout << ",";

    std::cout << std::endl;
  }
  std::cout << "]" << std::endl;
}

// Operation overloads
Matrix Matrix::operator+(const Matrix& other) const {
  Matrix result = *this;  // Copy current matrix
  result += other;        // Use the += operator
  return result;
}
Matrix Matrix::operator-(const Matrix& other) const {
  Matrix result(rows, cols);
  const float* A = this->data.data();
  const float* B = other.data.data();
  float* C = result.data.data();
  int size = rows * cols;
  int main_loop_limit = (size / 8) * 8;

  PARALLEL_FOR
  for (int i = 0; i < main_loop_limit; i += 8) {
    __m256 a_vec = _mm256_loadu_ps(&A[i]);
    __m256 b_vec = _mm256_loadu_ps(&B[i]);
    __m256 result_vec = _mm256_sub_ps(a_vec, b_vec);
    _mm256_storeu_ps(&C[i], result_vec);
  }
  for (int i = main_loop_limit; i < size; ++i) {
    C[i] = A[i] - B[i];
  }
  return result;
}
Matrix Matrix::operator*(const float scalar) const {
  Matrix result(rows, cols);
  const float* A = this->data.data();
  float* C = result.data.data();
  int size = rows * cols;
  int main_loop_limit = (size / 8) * 8;
  __m256 scalar_vec = _mm256_set1_ps(scalar);

  PARALLEL_FOR
  for (int i = 0; i < main_loop_limit; i += 8) {
    __m256 a_vec = _mm256_loadu_ps(&A[i]);
    __m256 result_vec = _mm256_mul_ps(a_vec, scalar_vec);
    _mm256_storeu_ps(&C[i], result_vec);
  }
  for (int i = main_loop_limit; i < size; ++i) {
    C[i] = A[i] * scalar;
  }
  return result;
}
Matrix& Matrix::operator=(const Matrix& other) {
  if (this == &other) return *this;  // Self-assignment check

  // 1. Check if we need to resize
  if (rows * cols != other.rows * other.cols) {
    // Only re-allocate if total size is different
    data.resize(other.rows * other.cols);
  }

  // 2. Update dimensions
  rows = other.rows;
  cols = other.cols;

  // 3. Copy data
  for (size_t i = 0; i < rows * cols; ++i) {
    data[i] = other.data[i];
  }

  return *this;
}

float Matrix::operator()(int row, int col) const {
  if (row < 0 || row >= static_cast<int>(rows) || col < 0 ||
      col >= static_cast<int>(cols)) {
    THROW_MATRIX_ERROR("Matrix indices out of bounds: (" + std::to_string(row) +
                       ", " + std::to_string(col) + ") for matrix of size (" +
                       std::to_string(rows) + "x" + std::to_string(cols) + ")");
  }
  return data[row * cols + col];
}
float& Matrix::operator()(int row, int col) {
  if (row < 0 || row >= static_cast<int>(rows) || col < 0 ||
      col >= static_cast<int>(cols)) {
    THROW_MATRIX_ERROR("Matrix indices out of bounds: (" + std::to_string(row) +
                       ", " + std::to_string(col) + ") for matrix of size (" +
                       std::to_string(rows) + "x" + std::to_string(cols) + ")");
  }
  return data[row * cols + col];
}

void Matrix::fill(float value) { std::fill(data.begin(), data.end(), value); }

Matrix Matrix::identity(int size) {
  Matrix result(size, size);
  for (int i = 0; i < size; ++i) {
    result(i, i) = 1.0f;
  }
  return result;
}

Matrix Matrix::zeros(int rows, int cols) {
  Matrix result(rows, cols);
  result.fill(0.0f);
  return result;
}
Matrix Matrix::ones(int rows, int cols) {
  Matrix result(rows, cols);
  result.fill(1.0f);
  return result;
}
Matrix Matrix::slice(int start_row, int end_row) const {
  if (start_row < 0 || end_row > rows || start_row >= end_row) {
    throw std::out_of_range("Matrix::slice indices out of bounds");
  }

  int new_rows = end_row - start_row;
  Matrix result(new_rows, cols);

  // Calculate memory offsets
  // data is std::vector<float>, stored row-major
  long long start_idx = (long long)start_row * cols;
  long long total_elements = (long long)new_rows * cols;

  // Copy the contiguous block of memory
  // This is much faster than looping element-by-element
  std::copy(data.begin() + start_idx, data.begin() + start_idx + total_elements,
            result.data.begin());

  return result;
}
Matrix Matrix::random(int rows, int cols) {
  Matrix result(rows, cols);
  for (int i = 0; i < rows * cols; ++i) {
    result.data[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  return result;
}
Matrix& Matrix::operator=(const std::vector<std::vector<float>>& inputData) {
  if (inputData.empty()) {
    rows = 0;
    cols = 0;
    data.clear();
    return *this;
  }

  const size_t target_rows = inputData.size();
  const size_t target_cols = inputData[0].size();

  for (const auto& row : inputData) {
    if (row.size() != target_cols) {
      THROW_MATRIX_ERROR("Cannot assign jagged array with row size" +
                         std::to_string(row.size()) +
                         " to Matrix with column size " +
                         std::to_string(target_cols));
    }
  }

  rows = static_cast<int>(target_rows);
  cols = static_cast<int>(target_cols);
  data.assign(rows * cols, 0.0f);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      (*this)(i, j) = inputData[i][j];
    }
  }

  return *this;
}

Matrix Matrix::transpose() const {
  // Allocate the result matrix (Swap rows and cols)
  Matrix result(cols, rows);

  const float* src = this->data.data();
  float* dst = result.data.data();

  int n = rows;
  int m = cols;

  // TUNING: 32 is often better than 64 for Transpose to avoid
  // "Cache Associativity Conflict" (evicting your own data).
  const int BLOCK_SIZE = 32;

// Parallelize the breakdown of blocks
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; i += BLOCK_SIZE) {
    for (int j = 0; j < m; j += BLOCK_SIZE) {
      // Define boundaries for this block (handle edges)
      int i_max = std::min(i + BLOCK_SIZE, n);
      int j_max = std::min(j + BLOCK_SIZE, m);

      // Transpose the block
      // By working on a small square, data stays in L1 Cache
      for (int ii = i; ii < i_max; ++ii) {
        // Pre-calculate row offset for source
        int src_row_offset = ii * m;

        for (int jj = j; jj < j_max; ++jj) {
          // dst[row j, col i] = src[row i, col j]
          dst[jj * n + ii] = src[src_row_offset + jj];
        }
      }
    }
  }

  return result;
}

const inline Matrix Matrix::Empty = Matrix(0, 0);
Matrix& Matrix::operator+=(const Matrix& other) {
  // CASE 1: Standard Element-wise Addition
  if (rows == other.rows && cols == other.cols) {
    float* A = data.data();
    const float* B = other.data.data();

    // Total number of elements (e.g., 3000 * 3000 = 9,000,000)
    int size = rows * cols;

    // Calculate how many blocks of 8 we can process
    int main_loop_limit = (size / 8) * 8;

    // 1. Parallel AVX Loop
    // We step by 8. OpenMP handles distributing these chunks to threads.
    PARALLEL_FOR
    for (int i = 0; i < main_loop_limit; i += 8) {
      // Load 8 floats from A
      __m256 a_vec = _mm256_loadu_ps(&A[i]);

      // Load 8 floats from B
      __m256 b_vec = _mm256_loadu_ps(&B[i]);

      // Add them: result = a + b
      __m256 sum_vec = _mm256_add_ps(a_vec, b_vec);

      // Store result back into A
      _mm256_storeu_ps(&A[i], sum_vec);
    }

    // 2. Cleanup Loop (Scalar)
    // Handle the remaining 1-7 elements that didn't fit in a block of 8
    // This is so fast it doesn't need to be parallelized.
    for (int i = main_loop_limit; i < size; ++i) {
      A[i] += B[i];
    }
  }
  // CASE 2: Broadcasting
  else if (other.rows == 1 && other.cols == cols) {
    float* A = data.data();
    const float* b = other.data.data();
    int n = rows;
    int m = cols;

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      float* row_ptr = &A[i * m];
      int j = 0;

      // AVX2 Broadcast Add
      for (; j <= m - 8; j += 8) {
        __m256 a_vec = _mm256_loadu_ps(&row_ptr[j]);
        __m256 b_vec = _mm256_loadu_ps(&b[j]);  // Re-uses bias vector
        _mm256_storeu_ps(&row_ptr[j], _mm256_add_ps(a_vec, b_vec));
      }
      // Cleanup
      for (; j < m; ++j) {
        row_ptr[j] += b[j];
      }
    }
  } else {
    THROW_MATRIX_ERROR("Dimension mismatch for += operation: (" +
                       std::to_string(rows) + "x" + std::to_string(cols) +
                       ") += (" + std::to_string(other.rows) + "x" +
                       std::to_string(other.cols) + ")");
  }

  return *this;
}

bool Matrix::operator==(const Matrix& other) const {
  if (rows != other.rows || cols != other.cols) {
    return false;
  }

  for (size_t i = 0; i < rows * cols; ++i) {
    if (data[i] != other.data[i]) {
      return false;
    }
  }

  return true;
}

// Optimized: Transpose directly into an existing buffer
void Matrix::transpose(Matrix& out) const {
  // 1. Resize if necessary (only happens once if reused)
  if (out.rows != cols || out.cols != rows) {
    out = Matrix(cols, rows);
  }

  const float* src = this->data.data();
  float* dst = out.data.data();
  int n = rows;
  int m = cols;
  const int BLOCK_SIZE = 32;

#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; i += BLOCK_SIZE) {
    for (int j = 0; j < m; j += BLOCK_SIZE) {
      int i_max = std::min(i + BLOCK_SIZE, n);
      int j_max = std::min(j + BLOCK_SIZE, m);

      for (int ii = i; ii < i_max; ++ii) {
        int src_row_offset = ii * m;
        for (int jj = j; jj < j_max; ++jj) {
          dst[jj * n + ii] = src[src_row_offset + jj];
        }
      }
    }
  }
}

// Profiling: cumulative time spent in Matrix::dot
double Matrix::profiling_dot_time = 0.0;

Matrix Matrix::dot(const Matrix& other) const {
  auto t_start = std::chrono::steady_clock::now();
  if (cols != other.rows) {
    THROW_MATRIX_ERROR("Dimension mismatch for dot product: (" +
                       std::to_string(rows) + "x" + std::to_string(cols) +
                       ") . (" + std::to_string(other.rows) + "x" +
                       std::to_string(other.cols) + ")");
  }

  long long ops = (long long)rows * cols * other.cols;
  const long long THREADING_THRESHOLD = 10'000;  // 300 Million Ops

  int num_threads = 1;

  int n = rows;
  int m = other.cols;
  int k_dim = cols;  // Shared dimension

  Matrix result = Matrix::zeros(n, m);
  float* C = result.data.data();
  const float* A = this->data.data();

  // --- FAST PATH (Small Matrices) ---
  if (ops < THREADING_THRESHOLD) {
    const float* B = other.data.data();  // Read B directly (No Transpose!)

    // The "ikj" loop order (Cache Friendly)
    for (int i = 0; i < n; ++i) {
      const float* row_A = &A[i * k_dim];
      float* row_C = &C[i * m];

      for (int k = 0; k < k_dim; ++k) {
        float val_A = row_A[k];
        const float* row_B = &B[k * m];

        // Compiler will auto-vectorize this simple inner loop
        for (int j = 0; j < m; ++j) {
          row_C[j] += val_A * row_B[j];
        }
      }
    }
    Matrix::profiling_dot_time +=
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - t_start)
            .count();
    return result;
  }

  // --- LARGE MATRIX PATH (Transposed + AVX2 + Multithreaded) ---

  // Only pay the 'bureaucracy' cost if the job is big enough
  int rows_per_thread_min = 16;  // Don't spawn a thread for less than 16 rows
  int max_threads_needed =
      (rows + rows_per_thread_min - 1) / rows_per_thread_min;
  int hardware_threads = std::thread::hardware_concurrency();
  int available = (hardware_threads > 2) ? (hardware_threads - 2) : 1;
  num_threads = std::max(1, std::min(available, max_threads_needed));
  // Transpose B for linear memory access (Critical for SIMD)
  auto other_T = other.transpose();
  const float* B_T = other_T.data.data();

  // TUNING PARAMETER: Fits in L1/L2 Cache.
  // 64 floats * 64 floats * 4 bytes = 16KB (Fits easily in L1)
  const int BLOCK_SIZE = std::max(16, rows_per_thread_min);

  // Micro-kernel blocked multiplication
#pragma omp parallel for num_threads(num_threads)
  for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
    // Loop 2: Blocks of j (Rows of B_T / Cols of B)
    for (int jj = 0; jj < m; jj += BLOCK_SIZE) {
      // Loop 3: Blocks of k (The shared dimension)
      // We iterate through the "depth" of the matrices in chunks
      for (int kk = 0; kk < k_dim; kk += BLOCK_SIZE) {
        // Calculate boundary for the current blocks (handle edges)
        int i_max = std::min(ii + BLOCK_SIZE, n);
        int j_max = std::min(jj + BLOCK_SIZE, m);
        int k_max = std::min(kk + BLOCK_SIZE, k_dim);

        // --- MICRO KERNEL (The actual math) ---
        // Everything inside here operates on data likely sitting in L1
        // Cache
        for (int i = ii; i < i_max; ++i) {
          // Pointer to the start of the current row in A
          // Offset by 'kk' because we are only processing a chunk of
          // the row
          const float* row_A_ptr = &A[i * k_dim + kk];

          float* row_C_ptr = &C[i * m];

          for (int j = jj; j < j_max; ++j) {
            // Pointer to the start of the current row in B_T
            const float* row_B_ptr = &B_T[j * k_dim + kk];

            // AVX2 Accumulator
            __m256 sum_vec = _mm256_setzero_ps();

            int k = 0;
            int range = k_max - kk;  // How many elements in this K-block?

            // Main SIMD Loop inside the block
            for (; k <= range - 8; k += 8) {
              __m256 a_vals = _mm256_loadu_ps(&row_A_ptr[k]);
              __m256 b_vals = _mm256_loadu_ps(&row_B_ptr[k]);
              sum_vec = _mm256_fmadd_ps(a_vals, b_vals, sum_vec);
            }

            // Horizontal sum of the vector
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            float partial_sum = 0.0f;
            for (int x = 0; x < 8; ++x) partial_sum += temp[x];

            // Scalar Cleanup for remaining k in this block
            for (; k < range; ++k) {
              partial_sum += row_A_ptr[k] * row_B_ptr[k];
            }

            // Add this partial sum to the result
            // Note: We use += because we are building the result
            // chunk by chunk
            row_C_ptr[j] += partial_sum;
          }
        }
      }
    }
  }

  return result;
}

// Optimized: Writes result into 'out' to avoid allocation
void Matrix::dot(const Matrix& other, Matrix& out) const {
  auto t_start = std::chrono::steady_clock::now();
  // 1. Safety Checks
  if (cols != other.rows) {
    THROW_MATRIX_ERROR("Dimension mismatch for dot product: (" +
                       std::to_string(rows) + "x" + std::to_string(cols) +
                       ") . (" + std::to_string(other.rows) + "x" +
                       std::to_string(other.cols) + ")");
  }
  if (out.rows != rows || out.cols != other.cols) {
    // Only resize if absolutely necessary (shouldn't happen in training loop)
    out = Matrix(rows, other.cols);
  }

  // 2. Reset Output
  out.fill(0.0f);

  const float* A = data.data();
  // We need B transposed for the AVX2 kernel to work efficiently
  // Optimization: In a real library, we'd have a 'dotNT' kernel to avoid this
  // transpose. For now, doing this transpose is still faster than a
  // cache-miss-heavy loop.
  Matrix B_T_Mat = other.transpose();
  const float* B_T = B_T_Mat.data.data();

  float* C = out.data.data();

  int n = rows;
  int m = other.cols;
  int k_dim = cols;

  const int BLOCK_SIZE = 64;  // Fits in L1 Cache

  // Reuse the exact same high-performance kernel logic from your other function
#pragma omp parallel for
  for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < m; jj += BLOCK_SIZE) {
      for (int kk = 0; kk < k_dim; kk += BLOCK_SIZE) {
        int i_max = std::min(ii + BLOCK_SIZE, n);
        int j_max = std::min(jj + BLOCK_SIZE, m);
        int k_max = std::min(kk + BLOCK_SIZE, k_dim);

        for (int i = ii; i < i_max; ++i) {
          const float* row_A_ptr = &A[i * k_dim + kk];
          float* row_C_ptr = &C[i * m];

          for (int j = jj; j < j_max; ++j) {
            const float* row_B_ptr = &B_T[j * k_dim + kk];

            __m256 sum_vec = _mm256_setzero_ps();
            int k = 0;
            int range = k_max - kk;

            // AVX2 Loop
            for (; k <= range - 8; k += 8) {
              __m256 a_vals = _mm256_loadu_ps(&row_A_ptr[k]);
              __m256 b_vals = _mm256_loadu_ps(&row_B_ptr[k]);
              sum_vec = _mm256_fmadd_ps(a_vals, b_vals, sum_vec);
            }

            // Horizontal Sum
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            float partial_sum = 0.0f;
            for (int x = 0; x < 8; ++x) partial_sum += temp[x];

            // Cleanup
            for (; k < range; ++k) {
              partial_sum += row_A_ptr[k] * row_B_ptr[k];
            }

            row_C_ptr[j] += partial_sum;
          }
        }
      }
    }
  }
  Matrix::profiling_dot_time +=
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t_start)
          .count();
}

Matrix Matrix::dotWithBTransposed(const Matrix& B_T) const {
  auto t_start = std::chrono::steady_clock::now();
  // Validate dimensions: this->cols == B_T.cols (since B_T is transpose of B)
  if (this->cols != B_T.cols) {
    THROW_MATRIX_ERROR("Dimension mismatch for dotWithBTransposed: (" +
                       std::to_string(rows) + "x" + std::to_string(cols) +
                       ") . B^T(" + std::to_string(B_T.rows) + "x" +
                       std::to_string(B_T.cols) + ")");
  }
  int n = this->rows;
  int m = B_T.rows;  // number of columns in original B
  int k_dim = this->cols;
  Matrix result = Matrix::zeros(n, m);
  float* C = result.data.data();
  const float* A = this->data.data();
  const float* BTR = B_T.data.data();

  const int BLOCK_SIZE = 64;
#pragma omp parallel for
  for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < m; jj += BLOCK_SIZE) {
      for (int kk = 0; kk < k_dim; kk += BLOCK_SIZE) {
        int i_max = std::min(ii + BLOCK_SIZE, n);
        int j_max = std::min(jj + BLOCK_SIZE, m);
        int k_max = std::min(kk + BLOCK_SIZE, k_dim);

        for (int i = ii; i < i_max; ++i) {
          const float* row_A_ptr = &A[i * k_dim + kk];
          float* row_C_ptr = &C[i * m];

          for (int j = jj; j < j_max; ++j) {
            const float* row_B_ptr = &BTR[j * k_dim + kk];

            __m256 sum_vec = _mm256_setzero_ps();
            int k = 0;
            int range = k_max - kk;
            for (; k <= range - 8; k += 8) {
              __m256 a_vals = _mm256_loadu_ps(&row_A_ptr[k]);
              __m256 b_vals = _mm256_loadu_ps(&row_B_ptr[k]);
              sum_vec = _mm256_fmadd_ps(a_vals, b_vals, sum_vec);
            }
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            float partial_sum = 0.0f;
            for (int x = 0; x < 8; ++x) partial_sum += temp[x];
            for (; k < range; ++k) {
              partial_sum += row_A_ptr[k] * row_B_ptr[k];
            }
            row_C_ptr[j] += partial_sum;
          }
        }
      }
    }
  }
  Matrix::profiling_dot_time +=
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t_start)
          .count();
  return result;
}

void Matrix::dotWithBTransposed(const Matrix& B_T, Matrix& out) const {
  auto t_start = std::chrono::steady_clock::now();
  if (this->cols != B_T.cols)
    throw std::runtime_error("Dimension mismatch for dotWithBTransposed");
  if (out.rows != this->rows || out.cols != B_T.rows) {
    out = Matrix(this->rows, B_T.rows);
  }
  out.fill(0.0f);

  const float* A = this->data.data();
  const float* BTR = B_T.data.data();
  float* C = out.data.data();

  int n = this->rows;
  int m = B_T.rows;
  int k_dim = this->cols;

  const int BLOCK_SIZE = 64;
#pragma omp parallel for
  for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < m; jj += BLOCK_SIZE) {
      for (int kk = 0; kk < k_dim; kk += BLOCK_SIZE) {
        int i_max = std::min(ii + BLOCK_SIZE, n);
        int j_max = std::min(jj + BLOCK_SIZE, m);
        int k_max = std::min(kk + BLOCK_SIZE, k_dim);

        for (int i = ii; i < i_max; ++i) {
          const float* row_A_ptr = &A[i * k_dim + kk];
          float* row_C_ptr = &C[i * m];

          for (int j = jj; j < j_max; ++j) {
            const float* row_B_ptr = &BTR[j * k_dim + kk];

            __m256 sum_vec = _mm256_setzero_ps();
            int k = 0;
            int range = k_max - kk;
            for (; k <= range - 8; k += 8) {
              __m256 a_vals = _mm256_loadu_ps(&row_A_ptr[k]);
              __m256 b_vals = _mm256_loadu_ps(&row_B_ptr[k]);
              sum_vec = _mm256_fmadd_ps(a_vals, b_vals, sum_vec);
            }
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            float partial_sum = 0.0f;
            for (int x = 0; x < 8; ++x) partial_sum += temp[x];
            for (; k < range; ++k) {
              partial_sum += row_A_ptr[k] * row_B_ptr[k];
            }
            row_C_ptr[j] += partial_sum;
          }
        }
      }
    }
  }
  Matrix::profiling_dot_time +=
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t_start)
          .count();
  return;
}

Matrix Matrix::addVector(const Matrix& vector) const {
  if (vector.rows != 1 || vector.cols != cols) {
    THROW_MATRIX_ERROR("Dimension mismatch for addVector operation.");
  }

  Matrix result = *this;  // Copy current matrix
  result += vector;       // Use the += operator with broadcasting
  return result;
}

void Matrix::sumRows(Matrix& out) const {
  float* R = out.data.data();
  const float* A = this->data.data();
  int n = rows;
  int m = cols;

  // Row-major iteration for cache-friendly access
  for (int i = 0; i < n; ++i) {
    const float* row_ptr = &A[i * m];
    int j = 0;
    // AVX2 vectorized accumulation
    for (; j <= m - 8; j += 8) {
      __m256 r_vec = _mm256_loadu_ps(&R[j]);
      __m256 a_vec = _mm256_loadu_ps(&row_ptr[j]);
      _mm256_storeu_ps(&R[j], _mm256_add_ps(r_vec, a_vec));
    }
    // Scalar cleanup
    for (; j < m; ++j) {
      R[j] += row_ptr[j];
    }
  }
}

Matrix Matrix::hadamard(const Matrix& other) const {
  if (rows != other.rows || cols != other.cols) {
    THROW_MATRIX_ERROR("Dimension mismatch for Hadamard product.");
  }

  Matrix result(rows, cols);
  const float* A = this->data.data();
  const float* B = other.data.data();
  float* C = result.data.data();
  int size = rows * cols;
  int main_loop_limit = (size / 8) * 8;

  PARALLEL_FOR
  for (int i = 0; i < main_loop_limit; i += 8) {
    __m256 a_vec = _mm256_loadu_ps(&A[i]);
    __m256 b_vec = _mm256_loadu_ps(&B[i]);
    __m256 result_vec = _mm256_mul_ps(a_vec, b_vec);
    _mm256_storeu_ps(&C[i], result_vec);
  }
  for (int i = main_loop_limit; i < size; ++i) {
    C[i] = A[i] * B[i];
  }
  return result;
}
