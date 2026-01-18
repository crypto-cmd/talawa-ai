#include <cmath>

#include "talawa/env/MountainCar.hpp"
#include "talawa/rl/Arena.hpp"
#include "talawa/rl/DQNAgent.hpp"
#include "talawa/rl/HumanAgent.hpp"
using namespace talawa;
int main() {
  env::MountainCar env(env::MountainCar::Friction::LOW);
  env.reset();

  auto as = env.get_action_space(0);
  auto obs_space = env.get_observation_space(0);

  // Create a Q-learning agent
  rl::agent::DQNConfig config{.num_actions = static_cast<int>(as.n())};
  // 1. Learning Dynamics
  config.learning_rate = 0.01f;  // Standard for Adam..
  config.gamma = 0.99f;          // Look ~100 steps into the future
  config.epsilon = 1.0f;         // Start full random

  // 2. Stability (The "Soft Update" Fix)
  config.target_update_type = rl::agent::TargetNetworkUpdateType::SOFT;
  config.tau = 0.005f;                // Move target 0.5%
  config.target_update_interval = 1;  // Update target every 1 step
  config.train_frequency = 4;         // Train every 4 steps

  // 3. Architecture
  config.use_double_dqn = true;  // Prevents overestimation
  config.use_dueling = true;     // Faster convergence

  // 4. Memory
  config.memory_size = 20000;       // Remember the "bad old days"
  config.memory_warmup_size = 100;  // Start training sooner
  config.sample_batch_size = 64;    // Robust gradient estimation

  auto builder = nn::NeuralNetworkBuilder::create({1, 1, 2});
  builder
      .add(nn::DenseLayerConfig{.neurons = 4,
                                .act = talawa::core::Activation::TANH,
                                .init = talawa::core::Initializer::HE_NORMAL})
      .setLossFunction(std::make_unique<nn::loss::HuberLoss>())
      .setOptimizer(std::make_unique<core::Adam>());
  rl::agent::DQNAgent ai(builder, config);

  env.register_agent(0, ai, "AIPlayer");
  talawa::arena::Arena arena(env);

  ai.getQNetwork().print();
  auto tournamentConfig = talawa::arena::Arena::TournamentConfig{
      .rounds = 10,
      .max_steps = 400,
  };
  auto matchConfig = talawa::arena::Arena::MatchConfig{
      .max_steps = 400,
      .training = true,
  };
  arena.tournament(tournamentConfig).print();

  for (int episode = 0; episode < 1000; ++episode) {
    arena.match(matchConfig);  // Train for 2000 episodes
    std::cout << "Completed episode " << episode + 1 << "/2000\r";
    if (episode > 400) {
      ai.set_epsilon(std::max(0.0015f, ai.get_epsilon() * 0.995f));
      // ai.set_learning_rate(std::max(0.001f, ai.get_learning_rate() *
      // 0.995f));
    }

    // if (episode % 100 == 0 && episode != 0) {
    //   std::cout << "\nIntermediate Tournament Results after " << episode + 1
    //             << " episodes:\n";
    //   auto tournamentResults = arena.tournament(tournamentConfig);
    //   if (tournamentResults.agents[0].avg_reward() >= -190) {
    //     std::cout << "Environment solved in " << episode + 1 << "
    //     episodes!\n"; tournamentResults.print(); break;
    //   }
    //   tournamentResults.print();
    // }
  }
  ai.getQNetwork().print();
  std::cout << "\nTraining complete! ( EPS:" << ai.get_epsilon()
            << ", LR:" << ai.get_learning_rate() << " )\n";
  arena.tournament(tournamentConfig).print();

  std::cout << "Done!\n";
  ai.print();
  ai.getQNetwork().print();

  // Final evaluation with rendering
  visualizer::IRenderer* renderer = &env;
  renderer->initRendering();
  while (renderer->is_active()) {
    arena.match(matchConfig, renderer);
  }
  return 0;
}
