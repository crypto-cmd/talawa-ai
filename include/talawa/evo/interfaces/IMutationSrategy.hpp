#pragma once
#include "talawa/evo/Genome.hpp"

namespace talawa::evo {

template <typename T>
class IMutationStrategy {
 public:
  virtual ~IMutationStrategy() = default;
  virtual void mutate(Genome<T>& ind) = 0;
};
}  // namespace talawa::evo
