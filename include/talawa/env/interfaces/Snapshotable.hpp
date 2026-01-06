#pragma once
#include <memory>

namespace talawa::env {
template <typename State>
class Snapshotable {
 public:
  virtual ~Snapshotable() = default;
  virtual std::unique_ptr<State> snapshot() const = 0;
  virtual void restore(const State& state) = 0;
};
}  // namespace talawa::env
