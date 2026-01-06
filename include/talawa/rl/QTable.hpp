#pragma once

#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "talawa/env/interfaces/IEnvironment.hpp"
#include "talawa/rl/IAgent.hpp"
#include "talawa/utils/ISerializable.hpp"

namespace talawa::rl::agent {

class QTable : public IAgent {
 public:
  enum class UpdateRule {
    Standard,  // For single-agent (Maze, CartPole) -> Uses PLUS
    ZeroSum    // For competitive games (Nim, Chess) -> Uses MINUS
  };

 private:
  enum class QValueAvailability { Available, Unavailable };

  struct QValue {
    float value;
    QValueAvailability availability;

    QValue() : value(0.0f), availability(QValueAvailability::Available) {}
    QValue(float v) : value(v), availability(QValueAvailability::Available) {}

    void make_unavailable() { availability = QValueAvailability::Unavailable; }
  };
  using QValues = std::vector<QValue>;
  using HashKey = std::string;
  struct HyperParameters {
    float learning_rate = 0.1f;
    float discount_factor = 0.99f;
    float epsilon = 1.0f;
    float starting_q_value = 0.0f;
    UpdateRule update_rule = UpdateRule::Standard;
  };

  HyperParameters params_;

 public:
  QTable(env::Space space, HyperParameters params)
      : params_(params),
        learning_rate_(params.learning_rate),
        discount_factor_(params.discount_factor),
        epsilon_(params.epsilon) {
    if (space.type != env::SpaceType::DISCRETE) {
      throw std::runtime_error("QTable only supports discrete action spaces.");
    }
    num_actions_ = space.n();
  }

  ~QTable() override = default;

  virtual env::Action act(
      const env::Observation& state,
      const std::optional<core::Matrix>& mask = std::nullopt,
      bool training = false) override;

  virtual void update(const env::Transition& transition) override;

  void print() const override {
    for (const auto& [state_hash, q_values] : q_table_) {
      std::cout << "State [" << state_hash << "]: Q-values = [";
      for (size_t i = 0; i < q_values.size(); ++i) {
        std::cout << (q_values[i].availability == QValueAvailability::Available
                          ? std::to_string(q_values[i].value)
                          : "*");
        if (i < q_values.size() - 1) std::cout << ", ";
      }
      std::cout << "]\n";
    }
  }
  void set_epsilon(float epsilon) { epsilon_ = epsilon; }
  float get_epsilon() const { return epsilon_; }

  void set_learning_rate(float lr) { learning_rate_ = lr; }
  float get_learning_rate() const { return learning_rate_; }

 private:
  int num_actions_;
  float learning_rate_;
  float discount_factor_;
  float epsilon_;

  HashKey to_key(const env::Observation& obs) const {
    HashKey key;
    auto size = obs.cols * obs.rows;
    key.reserve(size);

    const float* data = obs.rawData();
    for (int i = 0; i < size; ++i) {
      key += std::to_string(data[i]);
      if (i < size - 1) key += "_";
    }
    return key;
  }
  std::map<HashKey, QValues> q_table_;
  mutable std::mt19937 rng_{std::random_device{}()};
  QValues& get_q(const HashKey& state_hash);
  float get_max_q(const HashKey& state_hash) const;
};

}  // namespace talawa::rl::agent
