#pragma once

#include <random>

#include "talawa-ai/env/Environment.hpp"
namespace talawa_ai::env {

class LineEnvironmentGameState : public GameState {
 public:
  LineEnvironmentGameState(int position, bool done)
      : position_(position), done_(done) {}
  std::unique_ptr<GameState> clone() const override {
    return std::make_unique<LineEnvironmentGameState>(position_, done_);
  }
  bool equals(const GameState& other) const override {
    const auto* o = dynamic_cast<const LineEnvironmentGameState*>(&other);
    if (!o) return false;
    return position_ == o->position_ && done_ == o->done_;
  }
  size_t hash() const override { return std::hash<int>()(position_); }

  int position_;
  bool done_;
};

class LineEnvironment : public Environment {
 public:
  LineEnvironment(int length = 5);
  ~LineEnvironment() override = default;

  void reset() override;
  Observation observe() override;
  Transition step(const Action& action) override;

  std::unique_ptr<GameState> snapshot() const override;
  void restore(const GameState& state) override;

  bool is_done() const override;

  ActionType get_action_type() const override;
  std::string name() const override;
  std::vector<int> get_observation_shape() const override;
  int get_action_space_size() const override;

  LineEnvironment* clone() const override;

  void render() const override;

 private:
  LineEnvironmentGameState state_;
  int length_;
  mutable std::mt19937 rng_;
};
}  // namespace talawa_ai::env
