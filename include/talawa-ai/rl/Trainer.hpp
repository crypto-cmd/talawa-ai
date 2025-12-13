#pragma once
#include "talawa-ai/env/Environment.hpp"
#include "talawa-ai/rl/Agent.hpp"
#include "talawa-ai/rl/Scheduler.hpp"
namespace talawa_ai::rl {
struct TrainConfig {
  int episodes = 1000;
  int max_steps = 10000;
  int log_interval = 0;
  std::optional<unsigned int> seed = std::nullopt;
  std::optional<float> stop_on_reward = std::nullopt;

  scheduler::SchedulerSet schedulers;

  std::function<void(const scheduler::ScheduleContext&)> on_step_end = nullptr;
  std::function<void(const scheduler::ScheduleContext&)> on_episode_end =
      nullptr;
};
struct TrainResult {
  // Future: add metrics like total rewards, steps per episode, etc.
  float total_reward = 0.0f;
};
class Trainer {
 public:
  Trainer(env::Environment& env, agent::Agent& agent);
  virtual ~Trainer() = default;

  TrainResult train(TrainConfig config);

 protected:
  env::Environment& env_;
  agent::Agent& agent_;

 private:
  struct StepResult {
    float reward;
  };
  struct EpisodeResult {
    int steps;
    float reward;
  };

  scheduler::ScheduleContext make_context(int episode, int step,
                                          int total_steps, float episode_reward,
                                          float last_reward);

  StepResult run_step();
  EpisodeResult run_episode(int episode, int& total_steps, float last_rewards,
                            TrainConfig& config);
  bool should_stop(int episode, float episode_reward,
                   const TrainConfig& config);
};
}  // namespace talawa_ai::rl
