#include "talawa/neuralnetwork/DenseLayer.hpp"
#include "talawa/neuralnetwork/Layer.hpp"
#include "talawa/neuralnetwork/NeuralNetwork.hpp"
#include "talawa/rl/DQNAgent.hpp"

using namespace talawa::core;
using namespace talawa;
using namespace talawa::rl;
int main() {
  rl::agent::DQNConfig config{.num_actions = 4};
  config.sample_batch_size = 64;
  config.memory_warmup_size = 2000;
  config.memory_size = 50000;
  config.target_update_interval = 100;
  config.update_rule = rl::agent::UpdateRule::ZeroSum;
  auto builder = nn::NeuralNetworkBuilder::create({1, 4, 9});
  builder
      .add(nn::Conv2DLayerConfig{
          .filters = 32,
          .kernel_size = 3,
          .stride = 1,
          .padding = 1,
          .init = talawa::core::Initializer::GLOROT_UNIFORM,
          .act = talawa::core::Activation::RELU,
      })
      .add(nn::DenseLayerConfig{
          .neurons = 64,
          .act = talawa::core::Activation::RELU,
          .init = talawa::core::Initializer::GLOROT_UNIFORM,
      })
      .setLossFunction(std::make_unique<nn::loss::MeanSquaredError>())
      .setOptimizer(std::make_unique<core::Adam>(0.001f));
  rl::agent::DQNAgent dqn(builder, config);

  std::cout << "DQN Agent created.\n";
  dqn.print();

  auto input = core::Matrix::random(1, 4 * 9);
  auto action = dqn.act(env::Observation(input));

  dqn.update(env::Transition{
      .state = input,
      .action = action,
      .reward = 1.0f,
      .next_state = input,
      .status = env::EpisodeStatus::Terminated,
  });

  auto input2 = core::Matrix::random(1, 4 * 9);

  dqn.update(env::Transition{
      .state = input2,
      .action = dqn.act(env::Observation(input2)),
      .reward = 1.0f,
      .next_state = input2,
      .status = env::EpisodeStatus::Terminated,
  });
  return 0;
}
