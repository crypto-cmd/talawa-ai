#include "talawa/env/FrozenLake.hpp"

#include <random>

namespace talawa::env {
// [A, 1, 2, 3, 4, 5(O), 6, 7(O), 8, 9, 10, 11(O), 12(O), 13, 14, G]

FrozenLake::FrozenLake()
    : IEnvironment({0}),
      agent_position_(0),
      grid_size_(16),
      hole_positions_{5, 7, 11, 12},
      goal_position_(15) {
  reset();
}
void FrozenLake::reset(size_t random_seed) {
  agent_position_ = 0;
  done_ = false;
  cumulative_rewards_[0] = 0.0f;
  // Clear previous reports
  for (auto& [id, data] : agents_data_) {
    data.report = StepReport{};
  }
}
Observation FrozenLake::observe(const AgentID&) const {
  core::Matrix obs(1, 1);
  obs(0, 0) = static_cast<float>(agent_position_);
  return obs;
}
void FrozenLake::step(const Action& action) {
  if (done_) {
    throw std::runtime_error("Episode has terminated. Please reset the env.");
  }
  int move = static_cast<int>(action(0, 0));
  if (move < 0 || move > 3) {
    throw std::runtime_error("Invalid action for FrozenLake.");
  }
  auto prev_obs = observe(0);
  auto reward = -0.01f;  // Assume normal
  // Update position
  agent_position_ += move;
  if (agent_position_ >= grid_size_) {
    agent_position_ = grid_size_ - 1;
  }
  auto new_obs = observe(0);
  // Check for holes
  if (std::find(hole_positions_.begin(), hole_positions_.end(),
                agent_position_) != hole_positions_.end()) {
    reward = -1.0f;
    // Fell into a hole
    done_ = true;
  }
  // Check for goal
  if (agent_position_ == goal_position_) {
    done_ = true;
    reward = 1.0f;
  }
  // Normal step
  cumulative_rewards_[0] += reward;
  agents_data_.at(0).report = {
      .previous_state = prev_obs,
      .action = action,
      .reward = reward,
      .resulting_state = new_obs,
      .episode_status =
          done_ ? EpisodeStatus::Terminated : EpisodeStatus::Running,
  };
}
StepReport FrozenLake::last(const AgentID&) const {
  return agents_data_.at(0).report;
}
Space FrozenLake::get_action_space(const AgentID&) const {
  return Space::Discrete(4);  // 4 possible moves: Stay, Walk, Hop, Jump
}
Space FrozenLake::get_observation_space(const AgentID&) const {
  return Space::Discrete(grid_size_);  // Positions 0 to grid_size_-1
}
std::unique_ptr<IEnvironment> FrozenLake::clone() const {
  return std::make_unique<FrozenLake>(*this);
}
std::unique_ptr<int> FrozenLake::snapshot() const {
  return std::make_unique<int>(agent_position_);
}
void FrozenLake::restore(const int& state) {
  agent_position_ = state;
  done_ = (agent_position_ == goal_position_) ||
          (std::find(hole_positions_.begin(), hole_positions_.end(),
                     agent_position_) != hole_positions_.end());
  // Clear previous reports
  for (auto& [id, data] : agents_data_) {
    data.report = StepReport{};
  }
}

}  // namespace talawa::env
