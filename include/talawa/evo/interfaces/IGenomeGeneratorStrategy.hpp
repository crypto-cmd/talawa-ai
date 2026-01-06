#pragma once

#include <memory>

#include "talawa/evo/Genome.hpp"

namespace talawa::evo {
template <typename T>
class IGenomeGeneratorStrategy {
 public:
  virtual ~IGenomeGeneratorStrategy() = default;
  // Returns a single random Genome<T> wrapped in a unique_ptr
  virtual std::unique_ptr<Genome<T>> generateGene() = 0;
};
}  // namespace talawa::evo
