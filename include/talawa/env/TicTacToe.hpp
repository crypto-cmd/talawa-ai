#pragma once
#include "talawa/env/interfaces/IEnvironment.hpp"
namespace talawa::env {
class TicTacToe : public IEnvironment {
 public:
  TicTacToe();
  ~TicTacToe() override = default;
  void reset(size_t random_seed = 42) override;
  AgentID get_active_agent() const override;
  Observation observe(const AgentID&) const override;
  void step(const Action& action) override;
  Space get_action_space(const AgentID&) const override;
  Space get_observation_space(const AgentID&) const override;
  std::unique_ptr<IEnvironment> clone() const override;
  bool is_done() const override { return done_; }
  std::optional<ActionMask> get_legal_mask(const AgentID&) override;

 private:
  core::Matrix board_;  // 3x3 board
  int active_agent_index_;
  bool done_;
};
}  // namespace talawa::env
