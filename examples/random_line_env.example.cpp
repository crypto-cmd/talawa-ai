#include <iostream>
#include <memory>
#include <talawa-ai/env/LineEnvironment.hpp>
#include <talawa-ai/rl/QTable.hpp>
#include <talawa-ai/rl/Scheduler.hpp>
#include <talawa-ai/rl/Trainer.hpp>
using namespace talawa_ai;
int main() {
  auto env = env::LineEnvironment(5);
  rl::agent::QTable agent(env.get_action_space_size(), 0.1f, 0.9f, 1.0f);
  try {
    agent.load("line_env_qtable");
    std::cout << "Loaded existing Q-table from file.\n";
  } catch (const std::exception& e) {
    std::cout << "No existing Q-table found, starting fresh.\n";
  }

  agent.print();
  std::cout << "=== RANDOM LINE ENV TRAINER ===\n\n";
  rl::Trainer trainer(env, agent);

  auto config = rl::TrainConfig{
      .episodes = 10000,
      .max_steps = 100,
  };
  config.schedulers.add(
      rl::scheduler::schedule("epsilon")
          .use(
              rl::scheduler::chain()
                  .add(std::make_unique<rl::scheduler::ConstantScheduler>(1.0f),
                       5000)
                  .add(std::make_unique<rl::scheduler::ExponentialDecay>(
                           1.0f, 0.05f, 0.9995f),
                       rl::scheduler::UNTIL_END)
                  .build())
          .bind_to([&agent](float value) { agent.set_epsilon(value); })
          .on(rl::scheduler::ScheduleEvent::OnEpisodeEnd)
          .build());
  config.on_episode_end = [&agent](const rl::scheduler::ScheduleContext& ctx) {
    if (ctx.episode % 1000 == 0) {
      std::cout << "Episode " << ctx.episode
                << ": Reward = " << ctx.episode_reward
                << " Steps = " << ctx.step
                << " Epsilon = " << agent.get_epsilon() << "\n";
    }
  };
  auto result = trainer.train(std::move(config));

  std::cout << "Training complete.\n";
  std::cout << "Final Q-Table: " << agent.name() << "\n";
  agent.print();
  agent.save("line_env_qtable");

  return 0;
}
