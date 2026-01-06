#pragma once
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "talawa/env/types.hpp"
#include "talawa/rl/IAgent.hpp"

namespace talawa::env {
struct StepReport {
  env::Observation previous_state;    // S_t
  env::Action action;                 // A_t
  float reward;                       // R_t
  env::Observation resulting_state;   // S_{t+1}
  env::EpisodeStatus episode_status;  // Status after action
};

struct AgentData {
  AgentID id;
  std::string name;
  StepReport report;
};

/**
 * @brief Interface for an environment in which N agents can operate
 */
using AgentID = size_t;
class IEnvironment {
 protected:
  std::vector<AgentID> agent_order_;
  std::unordered_map<AgentID, AgentData> agents_data_;
  std::unordered_map<AgentID, std::reference_wrapper<rl::agent::IAgent>>
      agents_instances_;
  std::unordered_map<AgentID, float> cumulative_rewards_;

  int num_agents_;

 public:
  virtual ~IEnvironment() = default;

  // --- Core Agent-Environment Interaction Methods ---
  virtual void reset(size_t random_seed = 42) = 0;
  virtual AgentID get_active_agent() const = 0;

  // --- State & Observation Methods ---
  virtual Observation observe(const AgentID&) const = 0;
  virtual void step(const Action& action) = 0;
  virtual StepReport last(const AgentID&) const = 0;

  // ---- Metadata about the environment ----
  virtual Space get_action_space(const AgentID&) const = 0;
  virtual Space get_observation_space(const AgentID&) const = 0;

  virtual std::unique_ptr<IEnvironment> clone() const = 0;

  virtual std::optional<ActionMask> get_legal_mask(const AgentID&) {
    return std::nullopt;
  }
  float get_total_reward(AgentID id) const {
    auto it = cumulative_rewards_.find(id);
    if (it != cumulative_rewards_.end()) {
      return it->second;
    }
    return 0.0f;
  }

  virtual bool is_done() const = 0;

  virtual void register_agent(const AgentID& agent_id, rl::agent::IAgent& agent,
                              std::string name = "") {
    agents_data_[agent_id] = {
        .id = agent_id,
        .name = name,
        .report = StepReport{},
    };

    agents_instances_.emplace(agent_id, agent);
  }
  std::vector<AgentID> get_agent_order() const { return agent_order_; }
  rl::agent::IAgent& get_agent(const AgentID& agent_id) {
    return agents_instances_.at(agent_id).get();
  }
  std::string get_agent_name(const AgentID& agent_id) const {
    return agents_data_.at(agent_id).name;
  }
  // Determines if agent is allowed to act in general (not just in current step)
  virtual bool is_agent_available(const AgentID& agent_id) const {
    return agents_instances_.find(agent_id) != agents_instances_.end();
  }
};
};  // namespace talawa::env
