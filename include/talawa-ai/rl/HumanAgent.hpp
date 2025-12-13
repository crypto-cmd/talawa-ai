#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "talawa-ai/rl/Agent.hpp"

namespace talawa_ai::rl::agent {

class HumanAgent : public Agent {
 public:
  explicit HumanAgent(int action_size, std::string prompt = "Your move: ")
      : action_size_(action_size), prompt_(std::move(prompt)) {}

  std::string name() const override { return "HumanAgent"; }

  env::Action act(const env::GameState& state,
                  const std::optional<core::Matrix>& mask = std::nullopt,
                  bool training = false) override {
    (void)state;     // unused
    (void)training;  // unused

    std::vector<int> valid_moves;

    if (mask.has_value()) {
      const auto& m = mask.value();
      for (int i = 0; i < action_size_; ++i) {
        if (m(0, i) > 0.5f) valid_moves.push_back(i);
      }
    } else {
      for (int i = 0; i < action_size_; ++i) valid_moves.push_back(i);
    }

    int choice = -1;
    while (true) {
      std::cout << prompt_;
      std::cin >> choice;

      if (std::cin.fail()) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid input. Try again.\n";
        continue;
      }

      bool valid = std::find(valid_moves.begin(), valid_moves.end(), choice) !=
                   valid_moves.end();
      if (valid) break;

      std::cout << "Invalid move. Valid moves: ";
      for (int m : valid_moves) std::cout << m << " ";
      std::cout << "\n";
    }

    core::Matrix action(1, 1);
    action(0, 0) = static_cast<float>(choice);
    return action;
  }

  // Human doesn't learn
  void observe(env::Transition transition) override { (void)transition; }
  void learn() override {}
  bool ready_to_learn() const override { return false; }

 private:
  int action_size_;
  std::string prompt_;
};

}  // namespace talawa_ai::rl::agent
