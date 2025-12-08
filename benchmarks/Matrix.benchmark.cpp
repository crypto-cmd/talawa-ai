#include "talawa-ai/core/Matrix.hpp"

#include <vector>

#include "talawa-ai/utils/Timer.hpp"

// A "sink" function that prevents the compiler from optimizing away the
// variable
void do_not_optimize(const talawa_ai::core::Matrix& m) {
  volatile float sum = 0;
  // We just need to read one value to force the calculation
  if (m.rows > 0 && m.cols > 0) {
    sum = m(0, 0);
  }
}

void benchmark_multiplication(int size) {
  using namespace talawa_ai::core;

  // Initialize with random data so compiler doesn't optimize it away
  Matrix a(size, size);
  Matrix b(size, size);
  a.apply(
      [](int, int, float) { return static_cast<float>(rand()) / RAND_MAX; });
  b.apply(
      [](int, int, float) { return static_cast<float>(rand()) / RAND_MAX; });

  std::cout << "Benchmarking " << size << "x" << size << "... ";

  // 3. Measure PURE Math
  {
    MEASURE_SCOPE("Matrix Dot Product");
    auto c = a.dot(b);
    do_not_optimize(c);  // Prevent optimization
  }
}

int main() {
  // Test small, medium, and "heavy" sizes
  benchmark_multiplication(128);  // Fits in L1/L2 Cache (Fastest) (<0.01 ms)
  benchmark_multiplication(512);  // Fits in L3 Cache (Still Fast) (~0.02 ms)
  benchmark_multiplication(
      1024);  // RAM Heavy (Will expose cache misses) (~0.20 ms)
  benchmark_multiplication(
      2048);  // RAM Heavy (Will expose cache misses) (~0.30 ms)
  benchmark_multiplication(
      4096);  // RAM Heavy (Will expose cache misses) (~1.83 s)
  return 0;
}
