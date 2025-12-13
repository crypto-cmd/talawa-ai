#pragma once
#include <optional>
#include <random>

#include "talawa-ai/core/Matrix.hpp"
#include "talawa-ai/rl/Agent.hpp"

namespace talawa_ai::rl::agent {

class RandomAgent : public Agent {
 private:
  int action_size;
  std::mt19937 rng;
  std::uniform_int_distribution<int> dist;

 public:
  RandomAgent(int action_size)
      : action_size(action_size),
        rng(std::random_device{}()),
        dist(0, action_size - 1) {}

  std::string name() const override { return "RandomAgent"; }

  env::Action act(const env::GameState& state,
                  const std::optional<core::Matrix>& mask = std::nullopt,
                  bool training = false) override {
    std::vector<int> valid_moves;

    if (mask.has_value()) {
      // 1. Filter using the mask
      const core::Matrix& m = mask.value();
      for (int i = 0; i < action_size; ++i) {
        // Check if mask allows this move (1.0f)
        if (m(0, i) > 0.5f) {
          valid_moves.push_back(i);
        }
      }
    } else {
      // 2. No mask? Assume all allowed (or handle gracefully)
      for (int i = 0; i < action_size; ++i) valid_moves.push_back(i);
    }

    if (valid_moves.empty()) {
      throw std::runtime_error("No valid actions available for RandomAgent.");
    }

    // 3. Pick random valid move
    std::uniform_int_distribution<> dist(0, valid_moves.size() - 1);
    int choice = valid_moves[dist(rng)];

    // 4. Return as Matrix
    core::Matrix action = {{static_cast<float>(choice)}};
    return action;
  }
};

}  // namespace talawa_ai::rl::agent
