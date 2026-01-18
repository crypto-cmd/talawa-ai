#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "talawa/core/Matrix.hpp"
#include "talawa/env/interfaces/IEnvironment.hpp"
#include "talawa/env/interfaces/Snapshotable.hpp"

namespace talawa::env {

class StickGameEnv : public IEnvironment, public Snapshotable<uint8_t> {
 private:
  int _num_sticks_remaining;
  int _num_sticks_initial;

 public:
  enum Player : AgentID { PLAYER_1, PLAYER_2 };
  // Default to the classic "Game of 21"
  StickGameEnv(int num_sticks = 21)
      : IEnvironment({Player::PLAYER_1, Player::PLAYER_2}) {
    _num_sticks_initial = num_sticks;
    _num_sticks_remaining = num_sticks;
  }

  // 1. Reset the game
  void reset(size_t random_seed = 42) override {
    _num_sticks_remaining = 21;
    // Clear previous reports
    for (auto& [id, data] : agents_data_) {
      data.report = StepReport{};
    }
  }

  Observation observe(const AgentID&) const override {
    // Observation is simply the number of sticks remaining regardless of agent
    core::Matrix obs(1, 1);
    obs(0, 0) = static_cast<float>(_num_sticks_remaining);
    return obs;
  }

  // 2. The Core Logic
  void step(const Action& action) override {
    // Update the current info report for the active agent
    agents_data_.at(get_active_agent()).report.previous_state =
        observe(get_active_agent());
    agents_data_.at(get_active_agent()).report.action = action;

    // Validate action
    int sticks_taken = action.item<int>() + 1;  // Action 0,1,2 -> take 1,2,3
    if (sticks_taken < 1 || sticks_taken > 3) {
      throw std::runtime_error(
          "Invalid action: can only take 1, 2, or 3 sticks.");
    }
    if (sticks_taken > _num_sticks_remaining) {
      throw std::runtime_error(
          "Invalid action: cannot take more sticks than are remaining.");
    }

    // Apply action
    _num_sticks_remaining -= sticks_taken;

    // Update report after action
    agents_data_.at(get_active_agent()).report.resulting_state =
        observe(get_active_agent());

    // Determine reward for the reports
    if (_num_sticks_remaining <= 0) {
      // Current agent took the last stick and loses
      auto active_agent = get_active_agent();
      agents_data_.at(active_agent).report.reward = -1.0f;
      agents_data_.at(active_agent).report.episode_status =
          EpisodeStatus::Terminated;
      cumulative_rewards_[active_agent] += -1.0f;

      // Other agent wins
      auto other_agent_index = (active_agent_index_ + 1) % 2;
      AgentID other_agent = agent_order_[other_agent_index];  // Switch agent
      agents_data_.at(other_agent).report.reward = 1.0f;
      agents_data_.at(other_agent).report.episode_status =
          EpisodeStatus::Terminated;
      cumulative_rewards_[other_agent] += 1.0f;
    } else {
      // Game continues
      agents_data_.at(get_active_agent()).report.reward = 0.0f;
      agents_data_.at(get_active_agent()).report.episode_status =
          EpisodeStatus::Running;
    }

    // Switch active agent
    active_agent_index_ = (active_agent_index_ + 1) % 2;

    // Check for termination
    done_ = (_num_sticks_remaining <= 0);
  }

  // 3. Get Legal Moves (Critical for this game!)
  std::optional<ActionMask> get_legal_mask(const AgentID&) override {
    ActionMask mask(1, 3);
    // If 1 stick left, can only take index 0 (1 stick).
    // If 2 left, indices 0, 1. Etc.
    int max_take = std::min(3, _num_sticks_remaining);

    for (int i = 0; i < max_take; ++i) {
      mask(0, i) = 1;  // 1 = Legal
    }

    return mask;
  }

  std::unique_ptr<uint8_t> snapshot() const override {
    return std::make_unique<uint8_t>(
        static_cast<uint8_t>(_num_sticks_remaining));
  }
  void restore(const uint8_t& state) override {
    _num_sticks_remaining = static_cast<int>(state);
  }
  Space get_action_space(const AgentID&) const override {
    auto space = Space::Discrete(3);  // 0,1,2 represent taking 1,2,3 sticks
    return space;
  }

  Space get_observation_space(const AgentID&) const override {
    // 1. Define the space once
    Space shared_obs_space = Space::Continuous(
        {1},     // 1 number shown (the number of sticks remaining)
        {0.0f},  // Low: 0 sticks
        {21.0f}  // High: 21 sticks
    );
    return shared_obs_space;
  }
  std::unique_ptr<IEnvironment> clone() const override {
    return std::make_unique<StickGameEnv>(*this);
  }
};

}  // namespace talawa::env
