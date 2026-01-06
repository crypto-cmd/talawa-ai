#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "talawa/rl/IAgent.hpp"

namespace talawa::rl::agent {

class HumanAgent : public IAgent {
 public:
  HumanAgent(int action_size, std::string prompt = "Your move: ")
      : action_size_(action_size), prompt_(std::move(prompt)) {}

  env::Action act(const env::Observation& state,
                  const std::optional<core::Matrix>& mask = std::nullopt,
                  bool = false) override {
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
      std::cout << "Current Observation: ";
      state.print();
      std::cout << "\n";
      std::cout << "Valid moves: ";
      for (int m : valid_moves) std::cout << m << " ";
      std::cout << "\n";
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

      std::cout << "Invalid move\n";
    }

    return {{static_cast<float>(choice)}};
  }
  void print() const override {
    std::cout << "HumanAgent: Action Size = " << action_size_ << "\n";
  }

  void update(const env::Transition&) override {
    // HumanAgent does not learn from transitions
  }

 private:
  int action_size_;
  std::string prompt_;
};

}  // namespace talawa::rl::agent
