#pragma once

#include <algorithm>
#include <cmath>
#include <format>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

#include "talawa/env/interfaces/IEnvironment.hpp"
#include "talawa/rl/IAgent.hpp"
#include "talawa/rl/QTable.hpp"
#include "talawa/visuals/IRenderer.hpp"
using namespace talawa;

namespace talawa::arena {

class Arena {
 private:
  env::IEnvironment& environment_;
  std::set<env::AgentID> retired_agents_;

 public:
  Arena() = default;
  ~Arena() = default;

  explicit Arena(env::IEnvironment& env) : environment_(env) {}

  struct AgentMetrics {
    std::vector<float> reward_history;  // History of every game played

    int wins = 0;
    int losses = 0;
    int draws = 0;
    float max_reward = -std::numeric_limits<float>::infinity();
    float min_reward = std::numeric_limits<float>::infinity();

    // --- Computed Properties ---

    float avg_reward() const {
      if (reward_history.empty()) return 0.0f;
      float sum =
          std::accumulate(reward_history.begin(), reward_history.end(), 0.0f);
      return sum / reward_history.size();
    }

    float variance() const {
      if (reward_history.size() < 2) return 0.0f;
      float mean = avg_reward();
      float sq_sum =
          std::inner_product(reward_history.begin(), reward_history.end(),
                             reward_history.begin(), 0.0f);
      auto variance = (sq_sum / reward_history.size()) - (mean * mean);
      return variance;
    }

    float win_rate() const {
      if (reward_history.empty()) return 0.0f;
      return (static_cast<float>(wins) / reward_history.size()) * 100.0f;
    }
  };

  // The Report Card returned to the user
  struct TournamentStats {
    std::shared_ptr<Arena> arena;
    int episodes_played = 0;
    std::unordered_map<env::AgentID, AgentMetrics>
        agents;  // Map for easy ID lookup

    void print() const {
      std::cout << "\n========== TOURNAMENT REPORT ==========\n";
      std::cout << "Episodes Played: " << episodes_played << "\n";
      std::cout << "---------------------------------------\n";

      auto multi_agent = arena->environment_.get_agent_order().size() > 1;
      for (const auto& [id, m] : agents) {
        auto name = arena->environment_.get_agent_name(id);
        auto win_performance =
            multi_agent ? std::format(
                              "Results:\n  Win Rate: {:.2f}% (W:{} "
                              "L:{} D:{})",
                              m.win_rate(), m.wins, m.losses, m.draws)
                        : "";
        auto variance = m.variance();
        auto std_dev = std::sqrt(variance);

        auto report = std::format(
            "Agent ({}, {}){}\n"
            " Avg Reward: {:.3f} (+/- {:.3f})\n"
            " Range: [{}, {}]\n\n",
            id, name, win_performance, m.avg_reward(), std_dev, m.min_reward,
            m.max_reward);
        std::cout << report;
      }
      std::cout << "=======================================\n";
    }
  };

  struct MatchResult {
    std::unordered_map<env::AgentID, float> final_rewards;
    int steps_taken = 0;

    void print() const {
      std::cout << "\n========== MATCH RESULT ==========\n";
      for (const auto& [id, reward] : final_rewards) {
        std::cout << "Agent (" << id << ") Final Reward: " << reward << "\n";
      }
      std::cout << "Steps Taken: " << steps_taken << "\n";
      std::cout << "==================================\n";
    }
  };

  struct MatchConfig {
    int max_steps = 1000;  // Max steps per episode to avoid infinite loops
    bool training = true;
  };

  struct TournamentConfig {
    int rounds = 100;      // Number of episodes to play
    int max_steps = 1000;  // Max steps per episode to avoid infinite loops
  };
  // Runs an episode in that environment with the registered agents
  MatchResult match(MatchConfig config =
                        {
                            .max_steps = 1000,
                            .training = true,
                        },
                    visualizer::IRenderer* renderer = nullptr) {
    auto max_steps = config.max_steps;
    auto training = config.training;

    MatchResult result;
    environment_.reset();
    retired_agents_.clear();

    int ticks = 0;
    while (retired_agents_.size() < environment_.get_agent_order().size()) {
      for (const auto& agent_id : environment_.get_agent_order()) {
        rl::agent::IAgent& agent = environment_.get_agent(agent_id);

        if (retired_agents_.count(agent_id)) continue;  // Skip finalized agents
        if (!environment_.is_done() &&
            environment_.is_agent_available(agent_id)) {
          // Act
          auto obs = environment_.observe(agent_id);
          auto mask = environment_.get_legal_mask(agent_id);
          auto action = agent.act(obs, mask, training);

          environment_.step(action);
        }

        auto report = environment_.last(agent_id);

        //   Learn (only during training)
        if (training) {
          agent.update({
              .state = report.previous_state,
              .action = report.action,
              .reward = report.reward,
              .next_state = report.resulting_state,
              .status = report.episode_status,
          });
        }
        if (report.episode_status != env::EpisodeStatus::Running) {
          retired_agents_.insert(agent_id);
        }
      }

      if (renderer != nullptr) {
        if (!renderer->rendering_initialized()) {
          renderer->initRendering();
        }
        renderer->update();
        renderer->render();
        if (!renderer->is_active()) {
          break;  // Exit if the rendering window is closed
        }
      }
      ticks++;
      if (ticks >= max_steps) {
        break;  // Prevent infinite loops
      }
    }

    // Provide the match report
    for (auto agent_id : environment_.get_agent_order()) {
      float final_reward = environment_.get_total_reward(agent_id);
      result.final_rewards[agent_id] = final_reward;
    }
    result.steps_taken = ticks;
    return result;
  };
  /**
   * @brief Runs a competitive tournament to evaluate agent performance.
   * @param rounds Number of episodes to play.
   * @return TournamentStats containing wins, losses, and reward
   * distributions.
   * * NOTE: Training is strictly DISABLED during the tournament.
   */
  TournamentStats tournament(TournamentConfig config,
                             visualizer::IRenderer* renderer = nullptr) {
    TournamentStats stats;
    stats.arena = std::make_shared<Arena>(*this);
    stats.episodes_played = config.rounds;

    for (int i = 0; i < config.rounds; ++i) {
      // 1. DELEGATE: Play one complete game (Atomic Unit)
      // We force training=false so we are measuring skill, not learning.
      match(MatchConfig{.max_steps = config.max_steps, .training = false},
            renderer);

      // 2. COLLECT: Gather scores from the finished environment
      std::map<env::AgentID, float> episode_scores;
      float highest_score = -std::numeric_limits<float>::infinity();

      for (auto agent_id : environment_.get_agent_order()) {
        float score = environment_.get_total_reward(agent_id);
        episode_scores[agent_id] = score;

        if (score > highest_score) {
          highest_score = score;
        }
      }

      // 3. ANALYZE: Determine Winners, Losers, and Draws
      // We count how many agents achieved the highest score BUT only if there
      // is more than one agent in the environment.
      int winners_count = 0;
      for (const auto& [id, score] : episode_scores) {
        // Use epsilon for float comparison safety
        if (std::abs(score - highest_score) < 0.0001f) {
          winners_count++;
        }
      }

      // 4. UPDATE STATS
      for (const auto& [id, score] : episode_scores) {
        auto& m = stats.agents[id];

        // Record Raw Data
        m.reward_history.push_back(score);
        if (score > m.max_reward) m.max_reward = score;
        if (score < m.min_reward) m.min_reward = score;

        // Win/Loss/Draw Logic
        if (std::abs(score - highest_score) < 0.0001f) {
          if (winners_count > 1) {
            m.draws++;  // Shared victory = Draw
          } else {
            m.wins++;  // Unique victory = Win
          }
        } else {
          m.losses++;
        }
      }
    }
    return stats;
  }
};
}  // namespace talawa::arena
