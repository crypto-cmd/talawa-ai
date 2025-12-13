#pragma once

#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "talawa-ai/env/RLTypes.hpp"
#include "talawa-ai/rl/Agent.hpp"

namespace talawa_ai::rl::agent {

class QTable : public Agent {
 public:
  QTable(size_t num_actions, float learning_rate = 0.1f,
         float discount_factor = 0.99f, float epsilon = 1.0f);

  ~QTable() override = default;

  env::Action act(const env::GameState& state,
                  const std::optional<core::Matrix>& mask = std::nullopt,
                  bool training = false) override;

  std::string name() const override;

  void observe(env::Transition transition) override;
  virtual void learn() override;
  virtual bool ready_to_learn() const override;
  virtual void save(const std::string& filename) const override;
  virtual void load(const std::string& filename) override;
  void print() const override {
    for (const auto& [state_hash, q_values] : q_table_) {
      printf("State Hash: %zu | Q-Values: ", state_hash);
      for (const auto& q : q_values) {
        printf("%.6f ", q);
      }
      printf("\n");
    }
  }
  void set_epsilon(float epsilon) { epsilon_ = epsilon; }
  float get_epsilon() const { return epsilon_; }

 private:
  size_t num_actions_;
  float learning_rate_;
  float discount_factor_;
  float epsilon_;

  std::optional<env::Transition> last_transition_;

  std::map<size_t, std::vector<float>> q_table_;
  mutable std::mt19937 rng_{std::random_device{}()};

  std::vector<float>& get_q(size_t state_hash) {
    if (q_table_.find(state_hash) == q_table_.end()) {
      q_table_[state_hash] = std::vector<float>(num_actions_, 0.0f);
    }
    return q_table_[state_hash];
  }

  float get_max_q(size_t state_hash) const {
    auto it = q_table_.find(state_hash);
    if (it == q_table_.end()) {
      return 0.0f;  // Unseen state so reward is 0
    }
    return *std::max_element(it->second.begin(), it->second.end());
  }
};

}  // namespace talawa_ai::rl::agent
