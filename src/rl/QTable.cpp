#include "talawa/rl/QTable.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <utility>

namespace talawa::rl::agent {

env::Action QTable::act(const env::Observation& state,
                        const std::optional<core::Matrix>& mask,
                        bool training) {
  auto state_key = to_key(state);

  QValues& q_values = get_q(state_key);
  // Update availability based on mask
  if (mask.has_value()) {
    const core::Matrix& m = mask.value();
    for (size_t i = 0; i < q_values.size(); ++i) {
      if (m(0, static_cast<int>(i)) < 0.5f) {
        q_values[i].make_unavailable();
      }
    }
  }

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  if (training && dist(rng_) < epsilon_) {
    // Explore: choose a random available action
    std::vector<int> available_actions;

    for (size_t i = 0; i < q_values.size(); ++i) {
      if (q_values[i].availability == QValueAvailability::Available) {
        available_actions.push_back(static_cast<int>(i));
      }
    }
    if (available_actions.empty()) {
      throw std::runtime_error(
          "[QTable::act] ERROR: No available actions to choose from!");
    }
    std::uniform_int_distribution<size_t> action_dist(
        0, available_actions.size() - 1);
    int chosen_index = available_actions[action_dist(rng_)];
    return env::Action{{{static_cast<float>(chosen_index)}}};
  }
  // Exploit: choose the best available action
  float max_q = -std::numeric_limits<float>::infinity();
  int best_action = -1;
  for (size_t i = 0; i < q_values.size(); ++i) {
    if (q_values[i].availability == QValueAvailability::Available &&
        q_values[i].value > max_q) {
      max_q = q_values[i].value;
      best_action = static_cast<int>(i);
    }
  }
  if (best_action == -1) {
    throw std::runtime_error(
        "[QTable::act] ERROR: No available actions to choose from!");
  }
  return env::Action{{{static_cast<float>(best_action)}}};
}

void QTable::update(const env::Transition& transition) {
  auto state_hash = to_key(transition.state);

  float max_next_q = 0.0f;
  if (transition.status != env::EpisodeStatus::Terminated) {
    auto next_state_hash = to_key(transition.next_state);
    max_next_q = get_max_q(next_state_hash);
  }
  auto& q_values = get_q(state_hash);
  auto action_index = transition.action.item<int>();

  if (action_index >= q_values.size()) {
    std::cerr << "[QTable::update] ERROR: action_index out of bounds!"
              << std::endl;
    return;
  }

  // Update Q
  float target = 0.f;
  if (params_.update_rule == UpdateRule::Standard) {
    target = transition.reward + discount_factor_ * max_next_q;
  } else if (params_.update_rule == UpdateRule::ZeroSum) {
    target = transition.reward - discount_factor_ * max_next_q;
  } else {
    throw std::runtime_error("[QTable::update] ERROR: Unknown update rule.");
  }

  const float old_q = q_values[action_index].value;

  // Q(s, a) = Q(s, a) + lr * (reward +/- discount * maxQ(s') - Q(s, a))
  q_values[action_index].value = old_q + learning_rate_ * (target - old_q);
}

QTable::QValues& QTable::get_q(const QTable::HashKey& state_hash) {
  if (q_table_.find(state_hash) == q_table_.end()) {
    q_table_[state_hash] =
        QValues(num_actions_, QValue(params_.starting_q_value));
  }
  return q_table_[state_hash];
}

float QTable::get_max_q(const QTable::HashKey& state_hash) const {
  auto it = q_table_.find(state_hash);
  if (it == q_table_.end()) {
    return 0.0f;  // Unseen state so reward is 0
  }
  const QValues& q_values = it->second;
  float max_q = -std::numeric_limits<float>::infinity();
  ;
  for (const auto& qv : q_values) {
    if (qv.availability == QValueAvailability::Available && qv.value > max_q) {
      max_q = qv.value;
    }
  }
  if (max_q == -std::numeric_limits<float>::infinity()) {
    std::cerr << "[QTable::get_max_q] WARNING: All Q-values unavailable for "
                 "state "
              << state_hash << ". Returning 0.0f.\n";
    return 0.0f;
  }
  return max_q;
}

}  // namespace talawa::rl::agent
