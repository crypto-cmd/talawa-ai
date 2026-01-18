#include "talawa/core/Initializer.hpp"

#include <cmath>
#include <random>

namespace talawa {
namespace core {

Initializer::Initializer(Type type, unsigned int seed)
    : type(type), seed(seed), gen(seed) {}

void Initializer::apply(Matrix& weights) {
  switch (type) {
    case ZEROS:
      fillZeros(weights);
      break;
    case ONES:
      fillOnes(weights);
      break;
    case RANDOM_UNIFORM:
      fillRandomUniform(weights);
      break;
    case RANDOM_NORMAL:
      fillRandomNormal(weights);
      break;
    case GLOROT_UNIFORM:
      fillGlorotUniform(weights);
      break;
    case HE_NORMAL:
      fillHeNormal(weights);
      break;
  }
}

// --- Implementation of Strategies ---

void Initializer::fillZeros(Matrix& m) { m.fill(0.0f); }

void Initializer::fillOnes(Matrix& m) { m.fill(1.0f); }

void Initializer::fillRandomUniform(Matrix& m) {
  // Simple uniform distribution [-0.05, 0.05]
  std::uniform_real_distribution<float> dis(-0.05f, 0.05f);

  m.apply([&](int, int, float) { return dis(gen); });
}

void Initializer::fillRandomNormal(Matrix& m) {
  // Simple normal distribution (mean=0, std=0.05)
  std::normal_distribution<float> dis(0.0f, 0.05f);

  m.apply([&](int, int, float) { return dis(gen); });
}

void Initializer::fillGlorotUniform(Matrix& m) {
  // Limit = sqrt(6 / (fan_in + fan_out))
  float fan_in = static_cast<float>(m.rows);
  float fan_out = static_cast<float>(m.cols);
  float limit = std::sqrt(6.0f / (fan_in + fan_out));

  std::uniform_real_distribution<float> dis(-limit, limit);

  m.apply([&](int, int, float) { return dis(gen); });
}

void Initializer::fillHeNormal(Matrix& m) {
  // StdDev = sqrt(2 / fan_in)
  float fan_in = static_cast<float>(m.rows);
  float std_dev = std::sqrt(2.0f / fan_in);

  std::normal_distribution<float> dis(0.0f, std_dev);

  m.apply([&](int, int, float) { return dis(gen); });
}

}  // namespace core
}  // namespace talawa
