#pragma once
#include <optional>
#include <random>

#include "talawa/core/Matrix.hpp"
#include "talawa/env/types.hpp"
#include "talawa/rl/IAgent.hpp"

namespace talawa::rl::agent {

class RandomAgent : public IAgent {
 private:
  int action_size_;
  std::mt19937 rng_;

 public:
  RandomAgent(env::Space action_space)
      : action_size_(action_space.n()), rng_(std::random_device{}()) {
    if (action_space.type != env::SpaceType::DISCRETE) {
      throw std::runtime_error(
          "RandomAgent only supports discrete action spaces.");
    }
  }

  env::Action act(const env::Observation& observation,
                  const std::optional<core::Matrix>& mask = std::nullopt,
                  bool training = false) override {
    if (!mask.has_value()) {
      std::uniform_int_distribution<int> dist(0, action_size_ - 1);
      int choice = dist(rng_);
      return core::Matrix({{static_cast<float>(choice)}});
    }

    const auto& legal_mask = mask.value();
    std::vector<int> legal_actions;
    legal_mask.forEach([&](int row, int col, float val) {
      if (val > 0.5f) {
        legal_actions.push_back(col);
      }
      return val;
    });

    if (legal_actions.empty()) {
      throw std::runtime_error("No legal actions available for RandomAgent.");
    }
    std::uniform_int_distribution<int> legal_dist(0, legal_actions.size() - 1);
    int choice = legal_actions[legal_dist(rng_)];
    return core::Matrix({{static_cast<float>(choice)}});
  };
  void print() const override {
    printf("RandomAgent: Action Size = %d\n", action_size_);
  }
};

}  // namespace talawa::rl::agent
