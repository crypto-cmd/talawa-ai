#include "talawa/env/TicTacToe.hpp"

namespace talawa::env {
TicTacToe::TicTacToe() : IEnvironment({0, 1}), board_(3, 3), active_agent_index_(0) {
  reset();
}
void TicTacToe::reset(size_t random_seed) {
  board_.fill(0.0f);
  active_agent_index_ = 0;
  done_ = false;
  cumulative_rewards_[0] = 0.0f;
  cumulative_rewards_[1] = 0.0f;
  // Clear previous reports
  for (auto& [id, data] : agents_data_) {
    data.report = StepReport{};
  }
}
Observation TicTacToe::observe(const AgentID&) const {
  // Flatten the 3x3 board into a 1x9 observation
  core::Matrix obs(1, 9);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      obs(0, r * 3 + c) = board_(r, c);
    }
  }
  return Observation(obs);
}

std::optional<ActionMask> TicTacToe::get_legal_mask(const AgentID&) {
  ActionMask mask(1, 9);  // 9 possible positions
  for (int i = 0; i < 9; ++i) {
    int row = i / 3;
    int col = i % 3;
    if (board_(row, col) == 0.0f) {
      mask(0, i) = 1;  // Legal move
    } else {
      mask(0, i) = 0;  // Illegal move
    }
  }
  return mask;
}

void TicTacToe::step(const Action& action) {
  if (done_) {
    throw std::runtime_error("Game is already over.");
  }
  // Update the current info report for the active agent
  auto active_agent = get_active_agent();
  agents_data_.at(active_agent).report.previous_state = observe(active_agent);
  agents_data_.at(active_agent).report.action = action;

  // Validate action
  int index = action.item<int>();
  int row = index / 3;
  int col = index % 3;
  if (row < 0 || row >= 3 || col < 0 || col >= 3) {
    throw std::runtime_error("Invalid action: position out of bounds.");
  }
  if (board_(row, col) != 0.0f) {
    throw std::runtime_error("Invalid action: position already taken.");
  }
  // Apply action
  float mark = (active_agent_index_ == 0) ? 1.0f : -1.0f;
  board_(row, col) = mark;

  // Update report after action
  agents_data_.at(active_agent).report.resulting_state = observe(active_agent);

  // Determine rewards
  bool win = false;

  // Check rows, columns, diagonals
  for (int i = 0; i < 3; ++i) {
    if (board_(i, 0) == mark && board_(i, 1) == mark && board_(i, 2) == mark)
      win = true;
    if (board_(0, i) == mark && board_(1, i) == mark && board_(2, i) == mark)
      win = true;
  }
  if (board_(0, 0) == mark && board_(1, 1) == mark && board_(2, 2) == mark)
    win = true;
  if (board_(0, 2) == mark && board_(1, 1) == mark && board_(2, 0) == mark)
    win = true;

  if (win) {
    done_ = true;
    agents_data_.at(active_agent).report.reward = 1.0f;
    agents_data_.at(active_agent).report.episode_status =
        EpisodeStatus::Terminated;
    cumulative_rewards_[active_agent] += 1.0f;

    // Other agent loses
    auto other_agent_index = (active_agent_index_ + 1) % 2;
    AgentID other_agent = agent_order_[other_agent_index];
    agents_data_.at(other_agent).report.reward = -1.0f;
    agents_data_.at(other_agent).report.episode_status =
        EpisodeStatus::Terminated;
    cumulative_rewards_[other_agent] += -1.0f;
    return;
  }
  // Check for draw
  bool draw = true;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      if (board_(r, c) == 0.0f) {
        draw = false;
        break;
      }
    }
    if (!draw) break;
  }
  if (draw) {
    done_ = true;
    // Both agents get 0 reward
    for (const auto& agent_id : agent_order_) {
      agents_data_.at(agent_id).report.reward = 0.0f;
      agents_data_.at(agent_id).report.episode_status =
          EpisodeStatus::Terminated;
    }
    return;
  }
  // Game continues
  agents_data_.at(get_active_agent()).report.reward = 0.0f;
  agents_data_.at(get_active_agent()).report.episode_status =
      EpisodeStatus::Running;

  // Switch active agent
  active_agent_index_ = (active_agent_index_ + 1) % 2;
}

Space TicTacToe::get_action_space(const AgentID&) const {
  return Space::Discrete(9);  // 9 possible positions (0-8)
}
Space TicTacToe::get_observation_space(const AgentID&) const {
  return Space::Discrete(9);  // 3x3 board flattened
}
std::unique_ptr<IEnvironment> TicTacToe::clone() const {
  return std::make_unique<TicTacToe>(*this);
}
}  // namespace talawa::env
