#pragma once
#include <memory>
namespace talawa_ai::env {
class GameState {
 public:
  virtual ~GameState() = default;
  virtual std::unique_ptr<GameState> clone() const = 0;
  virtual bool equals(const GameState& other) const = 0;
  virtual size_t hash() const = 0;
};
}  // namespace talawa_ai::env
