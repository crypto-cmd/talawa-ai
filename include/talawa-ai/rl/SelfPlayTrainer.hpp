#pragma once

#include <functional>

#include "talawa-ai/env/TwoPlayerEnvironment.hpp"
#include "talawa-ai/rl/Agent.hpp"
#include "talawa-ai/rl/Scheduler.hpp"

namespace talawa_ai::rl {

struct SelfPlayConfig {
  int episodes = 10000;
  int max_steps_per_game = 100;

  scheduler::SchedulerSet schedulers;

  std::function<void(int episode, env::GameOutcome outcome)> on_game_end =
      nullptr;
  std::function<void(int episode)> on_episode_start = nullptr;
};

struct SelfPlayResult {
  int p1_wins = 0;
  int p2_wins = 0;
  int draws = 0;
  int total_episodes = 0;
};

class SelfPlayTrainer {
 public:
  // Same agent plays both sides (true self-play)
  SelfPlayTrainer(env::TwoPlayerEnvironment& env, agent::Agent& agent);

  // Two different agents (e.g., AI vs Human, or two different AIs)
  SelfPlayTrainer(env::TwoPlayerEnvironment& env, agent::Agent& agent1,
                  agent::Agent& agent2);

  SelfPlayResult train(SelfPlayConfig config);

  // Play a single game (for human play after training)
  env::GameOutcome play_game(bool render = false);

 private:
  env::TwoPlayerEnvironment& env_;
  agent::Agent& agent1_;
  agent::Agent& agent2_;
  bool same_agent_;

  agent::Agent& current_agent();
};

}  // namespace talawa_ai::rl
