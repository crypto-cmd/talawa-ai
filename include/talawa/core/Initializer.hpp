#pragma once
#include <random>

#include "talawa/core/Matrix.hpp"

namespace talawa {
namespace core {

class Initializer {
 public:
  // Supported Initialization Strategies
  enum Type {
    ZEROS,           // Set all to 0 (Bad for weights, okay for biases)
    ONES,            // Set all to 1
    RANDOM_UNIFORM,  // Small random numbers between -1 and 1
    RANDOM_NORMAL,   // Gaussian distribution
    GLOROT_UNIFORM,  // Xavier Initialization (Good for Sigmoid/Tanh)
    HE_NORMAL        // He Initialization (Good for ReLU)
  };

  // Constructors
  Initializer(Type type = GLOROT_UNIFORM, unsigned int seed = 42);

  // The main worker function
  // Modifies the passed matrix in-place
  void apply(Matrix& weights);

 private:
  Type type;
  unsigned int seed;  // For reproducibility
  std::mt19937 gen;

  // Internal logic helpers
  void fillZeros(Matrix& m);
  void fillOnes(Matrix& m);
  void fillRandomUniform(Matrix& m);
  void fillRandomNormal(Matrix& m);
  void fillGlorotUniform(Matrix& m);
  void fillHeNormal(Matrix& m);
};

}  // namespace core
}  // namespace talawa
