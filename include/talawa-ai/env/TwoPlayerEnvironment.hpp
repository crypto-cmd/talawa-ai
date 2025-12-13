#pragma once

#include "talawa-ai/env/Environment.hpp"

namespace talawa_ai::env {
constexpr int PLAYER_1 = 1;
constexpr int PLAYER_2 = -1;
// Outcome from a player's perspective
enum class GameOutcome { Win, Loss, Draw, Ongoing };

class TwoPlayerEnvironment : public Environment {
 public:
  virtual ~TwoPlayerEnvironment() = default;

  // Which player's turn
  virtual int current_player() const = 0;

  // Get outcome for a specific player
  virtual GameOutcome outcome_for(int player) const = 0;

  // Bring base class render into scope
  using Environment::render;

  // Render with player perspective (optional override)
  virtual void render(int perspective) const {
    (void)perspective;
    Environment::render();
  }
};

}  // namespace talawa_ai::env
