#include "talawa/env/Corridor.hpp"

#include <cmath>

#include "talawa/rl/Arena.hpp"
#include "talawa/rl/DQNAgent.hpp"
#include "talawa/rl/HumanAgent.hpp"
using namespace talawa;
int main() {
  env::Corridor env;

  auto as = env.get_action_space(0);
  auto obs_space = env.get_observation_space(0);

  // Create a Q-learning agent
  rl::agent::DQNConfig config{.num_actions = static_cast<int>(as.n())};
  config.sample_batch_size = 32;
  config.memory_warmup_size = 100;
  config.memory_size = 5000;
  config.target_update_interval = 1000;  // Update target very infrequently
  config.learning_rate = 0.001f;
  config.epsilon = 0.5f;
  config.use_double_dqn = true;

  auto builder =
      nn::NeuralNetworkBuilder::create({1, 1, (size_t)obs_space.n()});
  builder
      .add(nn::DenseLayerConfig{
          .neurons = 64,
          .act = talawa::core::Activation::RELU,
          .init = talawa::core::Initializer::HE_NORMAL,
      })
      .add(nn::DenseLayerConfig{
          .neurons = 64,
          .act = talawa::core::Activation::RELU,
          .init = talawa::core::Initializer::HE_NORMAL,
      })
      .setLossFunction(std::make_unique<nn::loss::MeanSquaredError>())
      .setOptimizer(std::make_unique<core::Adam>(0.001f));
  rl::agent::DQNAgent ai(builder, config);

  env.register_agent(0, ai, "AIPlayer");
  talawa::arena::Arena arena(env);

  auto tournamentConfig = talawa::arena::Arena::TournamentConfig{
      .rounds = 10,
      .max_steps = 50,  // More steps to reach goal=20 from start 0-9
  };
  arena.tournament(tournamentConfig).print();

  for (int episode = 0; episode < 2000; ++episode) {
    arena.match(tournamentConfig.max_steps, true);
    // Decay epsilon gradually
    if (episode > 50) {
      ai.set_epsilon(std::max(0.05f, ai.get_epsilon() * 0.995f));
    }
    // Only decay learning rate after good performance is achieved
    if (episode > 150) {
      ai.set_learning_rate(std::max(0.0001f, ai.get_learning_rate() * 0.99f));
    }
    if (episode % 100 == 0 && episode != 0) {
      // Set low epsilon for evaluation
      float saved_eps = ai.get_epsilon();
      ai.set_epsilon(0.01f);

      std::cout << "\nIntermediate Tournament Results after " << episode + 1
                << " episodes:\n";
      auto results = arena.tournament(tournamentConfig);
      results.print();

      ai.set_epsilon(saved_eps);  // Restore training epsilon

      // if (results.agents[0].avg_reward() >= 0.8f) {
      //   std::cout << "Environment solved!\n";
      //   break;
      // }
    }
    std::cout << "Completed episode " << episode + 1 << "/2000\r";
  }
  std::cout << "\nTraining complete! ( EPS:" << ai.get_epsilon()
            << ", LR:" << ai.get_learning_rate() << " )\n";
  arena.tournament(tournamentConfig).print();

  std::cout << "Done!\n";
  ai.print();
  return 0;
}
