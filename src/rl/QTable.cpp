#include "talawa-ai/rl/QTable.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>

namespace talawa_ai::rl::agent {

QTable::QTable(size_t num_actions, float learning_rate, float discount_factor,
               float epsilon)
    : num_actions_(num_actions),
      learning_rate_(learning_rate),
      discount_factor_(discount_factor),
      epsilon_(epsilon) {}

env::Action QTable::act(const env::GameState& state,
                        const std::optional<core::Matrix>& mask,
                        bool training) {
  size_t state_hash = state.hash();

  // Initialize Q-values for unseen states
  if (q_table_.find(state_hash) == q_table_.end()) {
    q_table_[state_hash] = std::vector<float>(num_actions_, 0.0f);
  }
  std::vector<float>& q_values = q_table_[state_hash];
  // Epsilon-greedy action selection
  if (training && (static_cast<float>(rng_()) / rng_.max()) < epsilon_) {
    // Explore: random action
    std::uniform_int_distribution<size_t> dist(0, num_actions_ - 1);
    size_t action_index = dist(rng_);
    return {{static_cast<float>(action_index)}};
  }
  // Exploit: best action
  size_t best_action = 0;
  float best_value = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < num_actions_; ++i) {
    if (q_values[i] > best_value) {
      best_value = q_values[i];
      best_action = i;
    }
  }
  return {{static_cast<float>(best_action)}};
}

std::string QTable::name() const {
  return "QTable<" + std::to_string(q_table_.size()) + ">";
}

void QTable::save(const std::string& filename) const {
  std::ofstream file(filename + ".qtable");
  if (!file.is_open()) {
    std::cerr << "Failed to open file for saving Q-table: " << filename << "\n";
    return;
  }
  for (const auto& [state_hash, q_values] : q_table_) {
    file << state_hash << ": [";
    ;
    for (const auto& q : q_values) {
      file << " " << q;
    }
    file << "]\n";
  }
  file.close();
  std::cout << "Q-table saved to " << filename << ".qtable\n";
}

void QTable::load(const std::string& filename) {
  std::ifstream file(filename + ".qtable");
  if (!file.is_open()) {
    std::cerr << "Failed to open file for loading Q-table: " << filename
              << "\n";
    return;
  }
  q_table_.clear();
  std::string line;
  while (std::getline(file, line)) {
    size_t colon_pos = line.find(':');
    if (colon_pos == std::string::npos) {
      throw std::runtime_error("Malformed Q-table line: " + line);
    }
    size_t state_hash = std::stoull(line.substr(0, colon_pos));
    size_t bracket_pos = line.find('[', colon_pos);
    size_t end_bracket_pos = line.find(']', bracket_pos);
    if (bracket_pos == std::string::npos ||
        end_bracket_pos == std::string::npos) {
      throw std::runtime_error("Malformed Q-table line: " + line);
    }
    std::string q_values_str =
        line.substr(bracket_pos + 1, end_bracket_pos - bracket_pos - 1);
    std::istringstream q_values_stream(q_values_str);
    std::vector<float> q_values;
    float q;
    while (q_values_stream >> q) {
      q_values.push_back(q);
    }
    q_table_[state_hash] = q_values;
  }
}

void QTable::observe(env::Transition transition) {
  last_transition_ = std::move(transition);
}
bool QTable::ready_to_learn() const { return last_transition_.has_value(); }
void QTable::learn() {
  if (!last_transition_->state) {
    std::cerr << "[QTable::learn] ERROR: last_transition_->state is nullptr!"
              << std::endl;
    return;
  }
  size_t state_hash = last_transition_->state->hash();
  float max_next_q = 0.0f;
  if (last_transition_->next_state) {
    size_t next_state_hash = last_transition_->next_state->hash();
    max_next_q = get_max_q(next_state_hash);
  } else {
  }
  auto& q_values = get_q(state_hash);
  auto action_index = static_cast<size_t>(last_transition_->action(0, 0));

  if (action_index >= q_values.size()) {
    std::cerr << "[QTable::learn] ERROR: action_index out of bounds!"
              << std::endl;
    return;
  }
  // Q(s, a) = Q(s, a) + lr * (reward + discount * maxQ(s') - Q(s, a))
  q_values[action_index] +=
      learning_rate_ * (last_transition_->reward +
                        discount_factor_ * max_next_q - q_values[action_index]);
}

}  // namespace talawa_ai::rl::agent
