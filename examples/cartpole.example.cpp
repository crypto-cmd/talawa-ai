#include "talawa/env/CartPole.hpp"

#include <cmath>

#include "talawa/rl/Arena.hpp"
#include "talawa/rl/DQNAgent.hpp"
#include "talawa/rl/HumanAgent.hpp"
using namespace talawa;
int main() {
  env::CartPole cartpole_env;
  cartpole_env.reset();

  auto as = cartpole_env.get_action_space(0);
  auto obs_space = cartpole_env.get_observation_space(0);

  // Create a Q-learning agent
  rl::agent::DQNConfig config{.num_actions = static_cast<int>(as.n())};
  config.sample_batch_size = 64;
  config.memory_warmup_size = 5000;
  config.memory_size = 15000;
  config.target_update_interval = 150;
  config.learning_rate = 0.1f;

    config.use_double_dqn = false;

  auto builder = nn::NeuralNetworkBuilder::create({1, 1, 4});
  builder
      .add(nn::DenseLayerConfig{
          .neurons = 64,
          .act = talawa::core::Activation::TANH,
          .init = talawa::core::Initializer::HE_NORMAL,
      })
      .setLossFunction(std::make_unique<nn::loss::MeanSquaredError>())
      .setOptimizer(std::make_unique<core::Adam>());
  rl::agent::DQNAgent ai(builder, config);

  cartpole_env.register_agent(0, ai, "AIPlayer");
  talawa::arena::Arena arena(cartpole_env);

  auto tournamentConfig = talawa::arena::Arena::TournamentConfig{
      .rounds = 10,
      .max_steps = 30,
  };
  arena.tournament(tournamentConfig).print();

  for (int episode = 0; episode < 2000; ++episode) {
    arena.match(tournamentConfig.max_steps, true);  // Train for 10000 episodes
    std::cout << "Completed episode " << episode + 1 << "/5000\r";
    if (episode > 400) {
      ai.set_epsilon(std::max(0.0015f, ai.get_epsilon() * 0.995f));
      ai.set_learning_rate(std::max(0.001f, ai.get_learning_rate() * 0.995f));
    }

    if (episode % 100 == 0 && episode != 0) {
      std::cout << "\nIntermediate Tournament Results after " << episode + 1
                << " episodes:\n";
      auto tournamentResults = arena.tournament(tournamentConfig);
      if (tournamentResults.agents[0].avg_reward() >= 500.0f) {
        std::cout << "Environment solved in " << episode + 1 << " episodes!\n";
        tournamentResults.print();
        break;
      }
      tournamentResults.print();
    }
  }
  std::cout << "\nTraining complete! ( EPS:" << ai.get_epsilon()
            << ", LR:" << ai.get_learning_rate() << " )\n";
  arena.tournament(tournamentConfig).print();

  std::cout << "Done!\n";
  ai.print();
  return 0;
}
