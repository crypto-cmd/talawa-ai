#pragma once
#include <memory>

#include "talawa/evo/Genome.hpp"

namespace talawa::evo {

template <typename T>
class ICrossoverStrategy {
 public:
  virtual ~ICrossoverStrategy() = default;
  virtual std::unique_ptr<Genome<T>> crossover(const Genome<T>& parent1,
                                               const Genome<T>& parent2) = 0;
};
}  // namespace talawa::evo
