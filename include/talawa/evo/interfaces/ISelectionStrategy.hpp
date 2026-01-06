#pragma once
#include <memory>

#include "talawa/evo/Genome.hpp"

namespace talawa::evo {
template <typename T>
class ISelectionStrategy {
 public:
  virtual ~ISelectionStrategy() = default;
  virtual const Genome<T>& select(
      const std::vector<std::unique_ptr<Genome<T>>>& pop) = 0;
};
}  // namespace talawa::evo
