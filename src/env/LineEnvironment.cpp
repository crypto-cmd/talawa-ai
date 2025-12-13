#include "talawa-ai/env/LineEnvironment.hpp"

#include <iostream>
#include <random>
#include <stdexcept>

#include "talawa-ai/core/Matrix.hpp"
#include "talawa-ai/env/LineEnvironment.hpp"

namespace talawa_ai::env {
LineEnvironment::LineEnvironment(int length)
    : state_(0, false), length_(length), rng_(std::random_device{}()) {}
void LineEnvironment::reset() {
  // Start at a random position (excluding the goal at length_-1)
  std::uniform_int_distribution<int> dist(0, length_ - 2);
  int start_pos = dist(rng_);
  state_ = LineEnvironmentGameState(start_pos, false);
}
Observation LineEnvironment::observe() {
  core::Matrix obs(1, length_);
  obs.fill(0.0f);
  if (state_.position_ >= 0 && state_.position_ < length_) {
    obs(0, state_.position_) = 1.0f;
  }
  return obs;
}
Transition LineEnvironment::step(const Action& action) {
  if (state_.done_) {
    throw std::runtime_error(
        "Cannot step in a finished environment. Please reset.");
  }
  auto state = snapshot();

  int move = static_cast<int>(action(0, 0));
  // Operate action: 0 = move left, 1 = move right
  state_.position_ += (move == 0) ? -1 : 1;
  state_.done_ = (state_.position_ < 0) || (state_.position_ == length_ - 1);
  float reward = -0.01f;
  if (state_.done_) {
    reward = (state_.position_ == length_ - 1) ? 1.0f : -1.0f;
  }

  auto next_state = snapshot();  // Copy this new state
  return Transition{.state = std::move(state),
                    .action = action,
                    .reward = reward,
                    .next_state = std::move(next_state),
                    .terminated = state_.done_};
}
std::unique_ptr<GameState> LineEnvironment::snapshot() const {
  return std::make_unique<LineEnvironmentGameState>(state_.position_,
                                                    state_.done_);
}
void LineEnvironment::restore(const GameState& state) {
  const auto& s = dynamic_cast<const LineEnvironmentGameState&>(state);
  state_.position_ = s.position_;
  state_.done_ = s.done_;
}

bool LineEnvironment::is_done() const { return state_.done_; }
ActionType LineEnvironment::get_action_type() const {
  return ActionType::DISCRETE;
}
std::string LineEnvironment::name() const {
  return "LineEnvironment [" + std::to_string(length_) + "]";
}
std::vector<int> LineEnvironment::get_observation_shape() const {
  return {1, length_};
}
int LineEnvironment::get_action_space_size() const { return 2; }
LineEnvironment* LineEnvironment::clone() const {
  return new LineEnvironment(*this);
}

void LineEnvironment::render() const {
  std::string line(length_, '-');
  if (state_.position_ >= 0 && state_.position_ < length_) {
    line[state_.position_] = 'A';  // Represent the agent with 'A'
  }
  std::cout << line << std::endl;
}
}  // namespace talawa_ai::env
