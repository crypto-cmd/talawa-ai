#include "talawa/rl/DQNAgent.hpp"

#include <iostream>
#include <stdexcept>

#include "talawa/core/Matrix.hpp"
namespace talawa::rl::agent {
using namespace talawa::core;
using namespace talawa::nn;
DuelingHead::DuelingHead(size_t input_dim, size_t num_actions,
                         talawa::core::Activation act,
                         talawa::core::Initializer init)
    : value_stream(input_dim, 1, act, init),
      advantage_stream(input_dim, num_actions, act, init) {}

core::Matrix DuelingHead::forward(const core::Matrix& input, bool is_training) {
  // Forward pass through value and advantage streams
  core::Matrix value = value_stream.forward(input, is_training);  // (N, 1)
  core::Matrix advantage =
      advantage_stream.forward(input, is_training);  // (N, A)

  auto meanA = core::Matrix(advantage.rows, 1);
  advantage.forEach([&](int i, int j, float val) {
    meanA(i, 0) += val / static_cast<float>(advantage.cols);
  });

  // Libary doesnt support broadcasting so:
  auto normalized_advantage =
      advantage.map([&](int i, int j, float val) { return val - meanA(i, 0); });

  auto q_values = normalized_advantage.map(
      [&](int i, int j, float val) { return val + value(i, 0); });

  return q_values;  // (N, A)
}
core::Matrix DuelingHead::backward(const core::Matrix& outputGradients) {
  // Split gradients into value and advantage parts
  core::Matrix dVdL = core::Matrix(outputGradients.rows, 1);
  outputGradients.reduceToCol(dVdL);

  auto meanA_grad = core::Matrix(outputGradients.rows, 1);
  outputGradients.forEach([&](int i, int j, float val) {
    meanA_grad(i, 0) += val / static_cast<float>(outputGradients.cols);
  });
  auto dAdL = outputGradients.map(
      [&](int i, int j, float val) { return val - meanA_grad(i, 0); });

  // Backward pass through streams
  auto dX_V = value_stream.backward(dVdL);      // (N, input_dim)
  auto dX_A = advantage_stream.backward(dAdL);  // (N, input_dim)

  // Combine input gradients
  core::Matrix total_input_gradients = dX_V + dX_A;  // (N, input_dim)
  return total_input_gradients;
}

std::vector<core::Matrix*> DuelingHead::getParameters() {
  auto value_params = value_stream.getParameters();
  auto advantage_params = advantage_stream.getParameters();
  value_params.insert(value_params.end(), advantage_params.begin(),
                      advantage_params.end());
  return value_params;
}
std::vector<core::Matrix*> DuelingHead::getParameterGradients() {
  auto value_grads = value_stream.getParameterGradients();
  auto advantage_grads = advantage_stream.getParameterGradients();
  value_grads.insert(value_grads.end(), advantage_grads.begin(),
                     advantage_grads.end());
  return value_grads;
}

std::string DuelingHead::info() const {
  return "Dueling Head Layer:\nValue Stream: " + value_stream.info() +
         "\nAdvantage Stream: " + advantage_stream.info();
}
void DuelingHead::save(std::ostream& out) const {
  value_stream.save(out);
  advantage_stream.save(out);
}
void DuelingHead::load(std::istream& in) {
  value_stream.load(in);
  advantage_stream.load(in);
}

Shape DuelingHead::getOutputShape() const {
  Shape value_shape = value_stream.getOutputShape();
  Shape advantage_shape = advantage_stream.getOutputShape();
  if (value_shape.flat() != 1) {
    throw std::runtime_error(
        "[DuelingHead::getOutputShape] ERROR: Value stream output shape "
        "invalid.");
  }
  return advantage_shape;
}

DQNAgent::DQNAgent(talawa::nn::NeuralNetworkBuilder& builder,
                   const DQNConfig& config)
    : config(config),
      replay_buffer(config.memory_size),
      direction_((config.update_rule == UpdateRule::Standard) ? 1 : -1) {
  // Initialize explorable and learnable parameters from config
  epsilon_ = config.epsilon;
  learning_rate_ = config.learning_rate;

  // Build Q-Network
  if (config.use_dueling) {
    builder.inject([&](const Shape& input_shape) {
      size_t input_dim = input_shape.flat();
      auto dueling_head = std::make_unique<DuelingHead>(
          input_dim, config.num_actions, talawa::core::Activation::LINEAR,
          config.weight_initializer);
      Shape output_shape = {1, 1, config.num_actions};
      return std::make_pair(std::move(dueling_head), output_shape);
    });
  } else {
    builder.add(nn::DenseLayerConfig{
        .neurons = config.num_actions,
        .act = talawa::core::Activation::LINEAR,
        .init = config.weight_initializer,
    });
  }
  std::cout << "Building Q-Network...\n";
  q_network = builder.build(config.learning_rate);
  std::cout << "Q-Network built.\n";
  if (config.use_double_dqn) {
    std::cout << "Using Double DQN.\n";
    target_network = q_network->clone();
  }

  // Build Target Network (Clone of Q-Network)
  // target_network = std::make_unique<NeuralNetwork>(*q_network);
  std::cout << "DQN Agent initialized.\n";
}
env::Action DQNAgent::act(const env::Observation& observation,
                          const std::optional<core::Matrix>& mask,
                          bool training) {
  // Forward pass through Q-Network
  core::Matrix q_values = q_network->predict(observation);

  auto actions = core::Matrix(q_values.rows, 1);

  // Epsilon-greedy action selection
  if (explorability() == IExplorable::Explorability::EXPLORE && training) {
    // Explore: random action
    // Get a list of valid actions
    for (size_t i = 0; i < q_values.rows; ++i) {
      std::vector<int> valid_actions;
      for (size_t j = 0; j < q_values.cols; ++j) {
        if (!mask.has_value() || mask->operator()(i, j) != 0.0f) {
          valid_actions.push_back(static_cast<int>(j));
        }
      }
      if (valid_actions.empty()) {
        throw std::runtime_error(
            "[DQNAgent::act] ERROR: No valid actions available.");
      }
      int rand_index = rand() % valid_actions.size();
      actions(i, 0) = static_cast<float>(valid_actions[rand_index]);
    }
    return env::Action(actions);
  }

  // Exploit: select best action

  // Apply mask if provided (set invalid actions to very low value)
  if (mask.has_value()) {
    q_values.forEach([&](int i, int j, float val) {
      if (mask->operator()(i, j) == 0.0f) {
        q_values(i, j) = -1e9f;  // Large negative value
      }
    });
  }

  // Exploit: select best action
  for (size_t i = 0; i < q_values.rows; ++i) {
    // Select action with highest Q-value
    float max_q = q_values(i, 0);
    int best_action = 0;
    for (size_t j = 1; j < q_values.cols; ++j) {
      if (q_values(i, j) > max_q) {
        max_q = q_values(i, j);
        best_action = static_cast<int>(j);
      }
    }
    actions(i, 0) = static_cast<float>(best_action);
  }

  return env::Action(actions);
}
void DQNAgent::update(const env::Transition& transition) {
  replay_buffer.add(transition);

  if (steps_done % config.train_frequency != 0) {
    return;  // Skip training this step
  }

  if (replay_buffer.size() < static_cast<size_t>(config.memory_warmup_size)) {
    return;  // Not enough data to train
  }
  auto true_target_network =
      config.use_double_dqn ? target_network.get() : q_network.get();
  // Sample a batch of transitions
  auto batch = replay_buffer.sample(config.sample_batch_size);

  auto q_current = q_network->predict(batch.states);  // (N, num_actions)

  // q_current.print();
  auto q_next_student =
      q_network->predict(batch.next_states);  // (N, num_actions)
  auto best_next_actions = core::Matrix(q_next_student.rows, 1);
  q_next_student.forEach([&](int i, int j, float val) {
    if (j == 0 ||
        val > q_next_student(i, static_cast<int>(best_next_actions(i, 0)))) {
      best_next_actions(i, 0) = static_cast<float>(j);
    }
  });
  auto q_next_target =
      true_target_network->predict(batch.next_states);  // (N, num_actions)
  auto future_values = core::Matrix(q_next_target.rows, 1);
  q_next_target.forEach([&](int i, int j, float val) {
    if (static_cast<int>(j) == static_cast<int>(best_next_actions(i, 0))) {
      future_values(i, 0) = val;
    }
  });
  auto target_q_values = q_current.map([&](int i, int j, float val) {
    auto reward = batch.rewards(i, 0);
    auto done = batch.dones(i, 0);
    if (static_cast<int>(j) == static_cast<int>(batch.actions(i, 0))) {
      // If episode is done, there's no future value
      if (done > 0.5f) {
        return reward;
      }
      return reward + config.gamma * future_values(i, 0);
    }
    return val;
  });

  q_network->train(batch.states, target_q_values);

  auto newq = q_network->predict(batch.states);

  steps_done++;
  updateTargetNetwork();
}
void DQNAgent::updateTargetNetwork() {
  if (config.target_update_type == TargetNetworkUpdateType::HARD &&
      steps_done % config.target_update_interval == 0) {
    // Hard update: copy weights directly
    target_network = std::make_unique<NeuralNetwork>(*q_network);

  } else if (config.target_update_type == TargetNetworkUpdateType::SOFT &&
             steps_done % config.target_update_interval == 0) {
    // Formula: Target = (Tau * Source) + ((1 - Tau) * Target)
    float tau = config.tau;  // e.g., 0.005
    float one_minus_tau = 1.0f - tau;

    auto& source_layers = q_network->layers;
    auto& target_layers = target_network->layers;

    for (size_t i = 0; i < source_layers.size(); ++i) {
      auto source_params = source_layers[i]->getParameters();
      auto target_params = target_layers[i]->getParameters();

      // Iterate through Weights and Biases
      for (size_t j = 0; j < source_params.size(); ++j) {
        Matrix* src = source_params[j];
        Matrix* dst = target_params[j];

        Matrix term1 = (*src) * tau;
        Matrix term2 = (*dst) * one_minus_tau;

        // Update the target weight in-place
        *dst = term1 + term2;
      }
    }
  }
}
void DQNAgent::print() const {
  auto double_dqn_str = config.use_double_dqn ? "Double" : "Single";
  auto dueling_str = config.use_dueling ? "Dueling" : "Standard";
  std::cout << "DQNAgent<" << config.num_actions << " Action Space, "
            << dueling_str << ", " << double_dqn_str << ">\n";
  std::cout << "--- Q-Network Architecture ---\n";
  for (const auto& layer : q_network->getLayers()) {
    std::cout << layer->info() << "\n";
  }
  std::cout << "-----------------------------\n";
}
}  // namespace talawa::rl::agent
