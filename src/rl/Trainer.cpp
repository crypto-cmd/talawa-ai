#include "talawa-ai/rl/Trainer.hpp"

namespace talawa_ai::rl {
Trainer::Trainer(env::Environment& env, agent::Agent& agent)
    : env_(env), agent_(agent) {}

scheduler::ScheduleContext Trainer::make_context(int episode, int step,
                                                 int total_steps,
                                                 float episode_reward,
                                                 float last_reward) {
  return scheduler::ScheduleContext{
      .episode = episode,
      .step = step,
      .total_steps = total_steps,
      .episode_reward = episode_reward,
      .last_reward = last_reward,
  };
}

Trainer::StepResult Trainer::run_step() {
  auto state = env_.snapshot();
  auto legal_mask = env_.get_legal_mask();
  auto action = agent_.act(*state, legal_mask, true);
  auto transition = env_.step(action);

  agent_.observe(std::move(transition));
  if (agent_.ready_to_learn()) {
    agent_.learn();
  }
  return StepResult{.reward = transition.reward};
}

Trainer::EpisodeResult Trainer::run_episode(int episode, int& total_steps,
                                            float last_rewards,
                                            TrainConfig& config) {
  env_.reset();
  int step = 0;
  float episode_reward = 0.0f;

  while (!env_.is_done() && step < config.max_steps) {
    auto step_result = run_step();
    episode_reward += step_result.reward;

    ++step;
    ++total_steps;
    auto ctx =
        make_context(episode, step, total_steps, episode_reward, last_rewards);
    config.schedulers.on_step(ctx);
    if (config.on_step_end) {
      config.on_step_end(ctx);
    }
  }
  return EpisodeResult{.steps = step, .reward = episode_reward};
}
bool Trainer::should_stop(int episode, float episode_reward,
                          const TrainConfig& config) {
  if (config.stop_on_reward.has_value() &&
      episode_reward >= config.stop_on_reward.value()) {
    return true;
  }
  return false;
}

TrainResult Trainer::train(TrainConfig config) {
  config.schedulers.initialize();

  auto total_rewards = 0.0f;
  float last_rewards = 0.0f;
  int total_steps = 0;

  for (int episode = 0; episode < config.episodes; ++episode) {
    auto result = run_episode(episode, total_steps, last_rewards, config);

    last_rewards = result.reward;
    total_rewards += result.reward;

    auto ctx = make_context(episode, result.steps, total_steps, result.reward,
                            last_rewards);
    config.schedulers.on_episode_end(ctx);
    if (config.on_episode_end) {
      config.on_episode_end(ctx);
    }
    if (should_stop(episode, result.reward, config)) {
      printf("Stopping training at episode %d due to reward threshold.\n",
             episode);
      break;
    }
  }
  return TrainResult{.total_reward = total_rewards};
}

}  // namespace talawa_ai::rl
