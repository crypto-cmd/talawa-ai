#pragma once

#include "talawa/env/interfaces/IEnvironment.hpp"
#include "talawa/env/interfaces/Snapshotable.hpp"

namespace talawa::env {
class FrozenLake : public IEnvironment, public Snapshotable<int> {
 public:
  FrozenLake();
  ~FrozenLake() override = default;

  bool is_done() const override { return done_; }
  void reset(size_t random_seed = 42) override;
  AgentID get_active_agent() const override;

  Observation observe(const AgentID&) const override;
  void step(const Action& action) override;
  StepReport last(const AgentID&) const override;

  Space get_action_space(const AgentID&) const override;
  Space get_observation_space(const AgentID&) const override;

  std::unique_ptr<IEnvironment> clone() const override;

  std::unique_ptr<int> snapshot() const override;
  void restore(const int& state) override;

 private:
  int agent_position_;
  int grid_size_;
  std::vector<int> hole_positions_;
  int goal_position_;
  bool done_;
};
}  // namespace talawa::env
