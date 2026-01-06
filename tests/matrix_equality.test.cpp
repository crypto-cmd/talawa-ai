#include <iostream>

#include "talawa/core/Matrix.hpp"

using namespace talawa::core;
int main() {
  Matrix m1 = Matrix::identity(3);
  Matrix m2 = m1;
  std::vector<Matrix*> a = {&m1, &m2};

  auto m4 = new Matrix(3, 4);
  *m4 = *(a[0]);  // Copy assignment?

  (*m4)(0, 0) = 10.0f;

  std::cout << m4->rows << "x" << m4->cols << std::endl;

  if (*m4 == *(a[0])) {
    std::cout << "Error: Matrices should not be equal after modification!"
              << std::endl;
    return 1;
  } else {
    std::cout << "Success: Matrices are not equal after modification."
              << std::endl;
  }
}
