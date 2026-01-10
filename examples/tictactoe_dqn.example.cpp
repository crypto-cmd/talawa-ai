#include "talawa/env/SticksGameEnvironment.hpp"
#include "talawa/env/TicTacToe.hpp"
#include "talawa/neuralnetwork/NeuralNetwork.hpp"
#include "talawa/rl/Arena.hpp"
#include "talawa/rl/DQNAgent.hpp"
#include "talawa/rl/HumanAgent.hpp"
#include "talawa/rl/QTable.hpp"
using namespace talawa;

int main() {
  auto env = env::TicTacToe();

  auto as = env.get_action_space(0);
  auto obs_space = env.get_observation_space(0);

  // Create a Q-learning agent
  rl::agent::DQNConfig config{.num_actions = static_cast<int>(as.n())};
  config.sample_batch_size = 32;
  config.memory_warmup_size = 2000;
  config.memory_size = 10000;
  config.update_rule = rl::agent::UpdateRule::ZeroSum;
  config.target_update_interval = 1000;

  std::cout << "Observation space:" << obs_space.n() << "\n";
  std::cout << "Action space:" << as.n() << "\n";

  auto builder = nn::NeuralNetworkBuilder::create({1, 3, 3});
  builder
      .add(nn::Conv2DLayerConfig{
          .filters = 64,
          .kernel_size = 3,
          .stride = 1,
          .padding = 1,
          .init = talawa::core::Initializer::GLOROT_UNIFORM,
          .act = talawa::core::Activation::LINEAR,
      })
      .add(nn::DenseLayerConfig{
          .neurons = 64,
          .act = talawa::core::Activation::LINEAR,
          .init = talawa::core::Initializer::GLOROT_UNIFORM,
      })
      .setLossFunction(std::make_unique<nn::loss::MeanSquaredError>())
      .setOptimizer(std::make_unique<core::Adam>(0.01f));
  rl::agent::DQNAgent ai(builder, config);

  ai.print();

  // Create a random agent for opponent
  rl::agent::QTable random(
      as, {
              .learning_rate = 0.0f,  // No learning
              .discount_factor = 0.0f,
              .epsilon = 1.0f,  // Always explore
              .starting_q_value = 0.0f,
              .update_rule = rl::agent::QTable::UpdateRule::ZeroSum,
          });
  // Q-learning agent is doing self-play
  env.register_agent(0, ai, "QAgent1");
  env.register_agent(1, random, "Random");
  auto tournamentConfig = talawa::arena::Arena::TournamentConfig{
      .rounds = 50,
      .max_steps = 9,
  };
  talawa::arena::Arena arena(env);
  arena.tournament(tournamentConfig).print();
  for (int episode = 0; episode < 8000; ++episode) {
    arena.match(9);  // Train for 2000 episodes
    std::cout << "Completed episode " << episode + 1 << "/8000\r";
    if (episode > 500) {
      ai.set_epsilon(std::max(0.0015f, ai.get_epsilon() * 0.999f));
      ai.set_learning_rate(std::max(0.05f, ai.get_learning_rate() * 0.999f));
    }
  }
  arena.tournament(tournamentConfig).print();

  ai.print();

  rl::agent::HumanAgent human(as.n());

  std::cout << "Starting a match against the trained Q-agent!\n";
  env.register_agent(0, human, "HumanPlayer");
  env.register_agent(1, ai, "TrainedQAgent");
  arena.match(50, false);  // One match with human
}
