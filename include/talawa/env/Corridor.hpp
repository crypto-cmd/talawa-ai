#pragma once
#include "talawa/env/interfaces/IEnvironment.hpp"
namespace talawa::env {
class Corridor : public IEnvironment {
 private:
  int position;
  int goal = 20;
  bool done_ = false;

 public:
  Corridor(/* args */);
  ~Corridor();

  void reset(size_t random_seed = 42) override;

  Observation observe(const AgentID&) const override;
  void step(const Action& action) override;
  StepReport last(const AgentID&) const override;

  Space get_action_space(const AgentID&) const override;
  Space get_observation_space(const AgentID&) const override;

  std::unique_ptr<IEnvironment> clone() const override;
};

}  // namespace talawa::env
