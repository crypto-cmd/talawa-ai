#include "talawa/env/Corridor.hpp"

namespace talawa::env {

Corridor::Corridor() : IEnvironment({0}), position(0) {
  reset();
}
Corridor::~Corridor() {}

void Corridor::reset(size_t random_seed) {
  position = rand() % (goal / 2);  // Start somewhere in the first half
  done_ = false;
  cumulative_rewards_[0] = 0.0f;  // Reset cumulative reward for the agent
}

talawa::env::Observation Corridor::observe(const AgentID&) const {
  core::Matrix obs(1, 1);
  obs(0, 0) =
      (static_cast<float>(position) / static_cast<float>(goal)) * 2.0f - 1.0f;
  return Observation(obs);
}

void Corridor::step(const Action& action) {
  if (done_) {
    throw std::runtime_error(
        "Episode has terminated. Please reset the environment.");
  }
  // Update the current info report for the agent
  auto active_agent = get_active_agent();
  agents_data_.at(active_agent).report.previous_state = observe(active_agent);
  agents_data_.at(active_agent).report.action = action;

  int move = static_cast<int>(action.item<int>());
  if (move == 0) {
    position = std::max(0, position - 1);  // Move left
  } else if (move == 1) {
    position = std::min(goal, position + 1);  // Move right
  } else {
    throw std::runtime_error("Invalid action for Corridor environment.");
  }

  if (position >= goal) {
    done_ = true;
  }
  // Update report after action
  agents_data_.at(active_agent).report.resulting_state = observe(active_agent);
  agents_data_.at(active_agent).report.reward =
      done_ ? 1.0f : -0.01f;  // Small penalty each step to encourage efficiency
  agents_data_.at(active_agent).report.episode_status =
      done_ ? EpisodeStatus::Terminated : EpisodeStatus::Running;
  cumulative_rewards_[active_agent] +=
      agents_data_.at(active_agent).report.reward;
}

Space Corridor::get_action_space(const AgentID&) const {
  return Space::Discrete(2);  // Two actions: left (0), right (1)
}
Space Corridor::get_observation_space(const AgentID&) const {
  return Space::Discrete(1);  // a single continuous value [0.0, 1.0]
}

std::unique_ptr<IEnvironment> Corridor::clone() const {
  return std::make_unique<Corridor>(*this);
}
}  // namespace talawa::env
