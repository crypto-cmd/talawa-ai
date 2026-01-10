#pragma once

#include <memory>
#include <vector>

#include "talawa/core/Activation.hpp"
#include "talawa/core/Initializer.hpp"
#include "talawa/core/Matrix.hpp"
#include "talawa/neuralnetwork/Layer.hpp"
#include "talawa/neuralnetwork/NeuralNetwork.hpp"
#include "talawa/rl/IAgent.hpp"
#include "talawa/rl/ReplayBuffer.hpp"

namespace talawa::rl::agent {
using namespace talawa::core;
using namespace talawa::nn;
// Dueling DQN Head Layer: Input -> Split(Value, Advantage)
class DuelingHead : public talawa::nn::ILayer {
 private:
  talawa::nn::DenseLayer value_stream;
  talawa::nn::DenseLayer advantage_stream;

 public:
  DuelingHead(size_t input_dim, size_t num_actions,
              talawa::core::Activation act = talawa::core::Activation::RELU,
              talawa::core::Initializer init =
                  talawa::core::Initializer::GLOROT_UNIFORM);
  ~DuelingHead() = default;

  virtual Matrix forward(const Matrix& input, bool is_training = true) override;
  virtual Matrix backward(const Matrix& outputGradients) override;
  virtual std::vector<Matrix*> getParameters() override;
  virtual std::vector<Matrix*> getParameterGradients() override;

  // Debugging utility to print layer info
  virtual std::string info() const override;
  // Saving and loading
  virtual void save(std::ostream& out) const override;
  virtual void load(std::istream& in) override;

  virtual Shape getOutputShape() const override;
  virtual std::unique_ptr<ILayer> clone() const override {
    return std::make_unique<DuelingHead>(*this);
  }
};

enum class TargetNetworkUpdateType { HARD, SOFT };
enum class UpdateRule {
  Standard,  // For single-agent (Maze, CartPole) -> Uses PLUS
  ZeroSum    // For competitive games (Nim, Chess) -> Uses MINUS
};
struct DQNConfig {
  int num_actions;
  bool use_dueling = true;
  bool use_double_dqn = true;
  int sample_batch_size = 32;
  int memory_warmup_size = 5000;
  int memory_size = 10000;
  int target_update_interval = 8000;
  float learning_rate = 0.001f;
  float epsilon = 1.0f;
  UpdateRule update_rule = UpdateRule::Standard;
  TargetNetworkUpdateType target_update_type = TargetNetworkUpdateType::HARD;
  float tau = 0.005f;  // For soft updates
  float gamma = 0.99f;
  Initializer weight_initializer = Initializer::GLOROT_UNIFORM;
};
class DQNAgent : public IAgent, public ILearnable, public IExplorable {
 private:
  DQNConfig config;
  std::unique_ptr<NeuralNetwork> q_network;
  std::shared_ptr<NeuralNetwork> target_network;
  rl::memory::ReplayBuffer replay_buffer;
  int direction_;

  int steps_done = 0;

 public:
  DQNAgent(talawa::nn::NeuralNetworkBuilder&, const DQNConfig&);
  ~DQNAgent() = default;
  virtual env::Action act(
      const env::Observation& observation,
      const std::optional<core::Matrix>& mask = std::nullopt,
      bool training = false) override;
  virtual void update(const env::Transition& transition) override;
  virtual void print() const override;

  // Additional functions
  void updateTargetNetwork();

 private:
  struct TrainableBatch {
    core::Matrix states;
    core::Matrix next_states;
    core::Matrix actions;
    core::Matrix rewards;
    core::Matrix dones;
  };
  TrainableBatch transformForTraining(
      const std::vector<env::Transition>& transitions);
};
}  // namespace talawa::rl::agent
