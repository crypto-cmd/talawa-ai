#include "talawa/core/Matrix.hpp"

#include <omp.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>

using namespace talawa;
// Helper: Floating point comparison (since 1.000001 != 1.0)
bool is_close(float a, float b, float epsilon = 1e-4f) {
  return std::abs(a - b) < epsilon;
}

void test_construction_and_access() {
  std::cout << "[Test] Construction & Access... ";

  core::Matrix m(3, 4);
  assert(m.rows == 3);
  assert(m.cols == 4);

  // Test writing
  m(0, 0) = 10.5f;
  m(2, 3) = -5.0f;

  // Test reading
  assert(is_close(m(0, 0), 10.5f));
  assert(is_close(m(2, 3), -5.0f));

  // Test initialization (should be 0)
  assert(is_close(m(1, 1), 0.0f));

  std::cout << "Passed. ✅" << std::endl;
}

void test_fill_zeros_ones() {
  std::cout << "[Test] Fill, Zeros, Ones... ";

  // Test Fill
  core::Matrix m(2, 2);
  m.fill(3.0f);
  assert(is_close(m(0, 0), 3.0f));
  assert(is_close(m(1, 1), 3.0f));

  // Test Zeros Static Method
  core::Matrix z = core::Matrix::zeros(2, 2);
  assert(is_close(z(0, 0), 0.0f));
  assert(is_close(z(1, 1), 0.0f));

  // Test Ones Static Method
  core::Matrix o = core::Matrix::ones(2, 2);
  assert(is_close(o(0, 0), 1.0f));
  assert(is_close(o(1, 1), 1.0f));

  std::cout << "Passed. ✅" << std::endl;
}

void test_scalar_operations() {
  std::cout << "[Test] Scalar Operations... ";

  core::Matrix m = core::Matrix::ones(2, 2);

  // 1.0 * 5.5 = 5.5
  m = m * 5.5f;

  assert(is_close(m(0, 0), 5.5f));
  assert(is_close(m(1, 1), 5.5f));

  std::cout << "Passed. ✅" << std::endl;
}

void test_transpose() {
  std::cout << "[Test] Transpose... ";

  // Create a 2x3 Matrix
  // [ 1, 2, 3 ]
  // [ 4, 5, 6 ]
  core::Matrix m(2, 3);
  int counter = 1;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      m(i, j) = (float)counter++;
    }
  }

  // Transpose it -> Should be 3x2
  core::Matrix t = m.transpose();

  assert(t.rows == 3);
  assert(t.cols == 2);

  // Check values (Rows become Columns)
  // [ 1, 4 ]
  // [ 2, 5 ]
  // [ 3, 6 ]
  assert(is_close(t(0, 0), 1.0f));
  assert(is_close(t(0, 1), 4.0f));
  assert(is_close(t(2, 0), 3.0f));  // Was (0, 2)

  std::cout << "Passed. ✅" << std::endl;
}

void test_dot_product() {
  std::cout << "[Test] Dot Product (MatMul)... ";

  // Matrix A (2x3)
  // [ 1, 2, 3 ]
  // [ 4, 5, 6 ]
  core::Matrix A(2, 3);
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(0, 2) = 3;
  A(1, 0) = 4;
  A(1, 1) = 5;
  A(1, 2) = 6;

  // Matrix B (3x2)
  // [ 7, 8 ]
  // [ 9, 1 ]
  // [ 2, 3 ]
  core::Matrix B(3, 2);
  B(0, 0) = 7;
  B(0, 1) = 8;
  B(1, 0) = 9;
  B(1, 1) = 1;
  B(2, 0) = 2;
  B(2, 1) = 3;

  // Compute C = A * B
  // Expected Result (2x2):
  // [ 31, 19 ]
  // [ 85, 55 ]
  core::Matrix C = A.dot(B);

  assert(C.rows == 2);
  assert(C.cols == 2);

  assert(is_close(C(0, 0), 31.0f));
  assert(is_close(C(0, 1), 19.0f));
  assert(is_close(C(1, 0), 85.0f));
  assert(is_close(C(1, 1), 55.0f));

  std::cout << "Passed. ✅" << std::endl;
}
double test_speed(int size = 1000) {
  std::cout << "[Test] Speed Test (" << size << "x" << size << " . " << size
            << "x" << size << " Dot Product)... ";

  core::Matrix A = core::Matrix::ones(size, size);
  core::Matrix B = core::Matrix::ones(size, size);
  // Just time the dot product but it might be too fast and we get a negative
  // time due to clock precision
  auto start = std::chrono::steady_clock::now();
  core::Matrix C = A.dot(B);
  auto end = std::chrono::steady_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "Completed in " << duration.count() << " seconds. ✅"
            << std::endl;
  return duration.count();
}

void test_equality() {
  std::cout << "[Test] Equality Operator... ";

  core::Matrix m1 = core::Matrix::identity(3);
  core::Matrix m3 = core::Matrix::zeros(3, 3);
  m3.apply([](int i, int j, float value) -> float {
    return random() % 3;  // Random 0 or 1
  });

  auto m4 = m1.dot(m3);
  assert(m4 == m3);  // Identity * Ones = Ones

  m4.print();

  std::cout << "Passed. ✅" << std::endl;
}

int main() {
  std::cout << "===========================" << std::endl;
  std::cout << "   RUNNING MATRIX TESTS    " << std::endl;
  std::cout << "===========================" << std::endl;

  test_construction_and_access();
  test_fill_zeros_ones();
  test_scalar_operations();
  test_transpose();
  test_dot_product();
  test_equality();

  double total_duration = 0.0;
  int iterations = 15;
  for (int i = 0; i < iterations; ++i) {
    double duration = test_speed(3000);
    total_duration += duration;
  }
  std::cout << "Average Time for 3000x3000 Dot Product: "
            << (total_duration / iterations) << " seconds." << std::endl;

  std::cout << "===========================" << std::endl;
  std::cout << "   ALL TESTS PASSED        " << std::endl;
  std::cout << "===========================" << std::endl;

  // Demonstrate Pretty Print at the end
  std::cout << "\nVisual Print Test:" << std::endl;
  core::Matrix m(3, 3);
  m.fill(1.5f);
  m(1, 1) = 100.55f;
  m.print();

  return 0;
}
