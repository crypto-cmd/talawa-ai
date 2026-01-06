#pragma once

#include "talawa/evo/Genome.hpp"
namespace talawa::evo {
template <typename T>
class IFitnessStrategy {
 public:
  virtual ~IFitnessStrategy() = default;
  virtual double calculateFitness(const Genome<T>& ind) = 0;
};
}  // namespace talawa::evo
