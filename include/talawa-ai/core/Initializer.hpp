#pragma once
#include "talawa-ai/core/Matrix.hpp"
#include <random>

namespace talawa_ai {
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
  void apply(Matrix &weights) const;

 private:
  Type type;
  unsigned int seed; // For reproducibility

  // Internal logic helpers
  void fillZeros(Matrix &m) const;
  void fillOnes(Matrix &m) const;
  void fillRandomUniform(Matrix &m) const;
  void fillRandomNormal(Matrix &m) const;
  void fillGlorotUniform(Matrix &m) const;
  void fillHeNormal(Matrix &m) const;
};

}  // namespace core
}  // namespace talawa_ai
