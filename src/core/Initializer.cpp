#include "talawa-ai/core/Initializer.hpp"

#include <cmath>
#include <random>

namespace talawa_ai {
namespace core {

Initializer::Initializer(Type type, unsigned int seed)
    : type(type), seed(seed) {}

void Initializer::apply(Matrix &weights) const {
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

void Initializer::fillZeros(Matrix &m) const { m.fill(0.0f); }

void Initializer::fillOnes(Matrix &m) const { m.fill(1.0f); }

void Initializer::fillRandomUniform(Matrix &m) const {
  // Simple uniform distribution [-0.05, 0.05]
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(-0.05f, 0.05f);

  m.apply([&](int, int, float) { return dis(gen); });
}

void Initializer::fillRandomNormal(Matrix &m) const {
  // Simple normal distribution (mean=0, std=0.05)
  std::mt19937 gen(seed);
  std::normal_distribution<float> dis(0.0f, 0.05f);

  m.apply([&](int, int, float) { return dis(gen); });
}

void Initializer::fillGlorotUniform(Matrix &m) const {
  // Limit = sqrt(6 / (fan_in + fan_out))
  float fan_in = static_cast<float>(m.rows);
  float fan_out = static_cast<float>(m.cols);
  float limit = std::sqrt(6.0f / (fan_in + fan_out));

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(-limit, limit);

  m.apply([&](int, int, float) { return dis(gen); });
}

void Initializer::fillHeNormal(Matrix &m) const {
  // StdDev = sqrt(2 / fan_in)
  float fan_in = static_cast<float>(m.rows);
  float std_dev = std::sqrt(2.0f / fan_in);

  std::mt19937 gen(seed);
  std::normal_distribution<float> dis(0.0f, std_dev);

  m.apply([&](int, int, float) { return dis(gen); });
}

}  // namespace core
}  // namespace talawa_ai
