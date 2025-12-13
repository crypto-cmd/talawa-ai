#pragma once

#include <unordered_map>
#include <vector>

#include "talawa-ai/env/Environment.hpp"

namespace talawa_ai::env {

class GraphEnvironmentGameState : public GameState {
 public:
  GraphEnvironmentGameState(int node, bool done) : node_(node), done_(done) {}

  std::unique_ptr<GameState> clone() const override {
    return std::make_unique<GraphEnvironmentGameState>(node_, done_);
  }

  bool equals(const GameState& other) const override {
    const auto* o = dynamic_cast<const GraphEnvironmentGameState*>(&other);
    if (!o) return false;
    return node_ == o->node_ && done_ == o->done_;
  }

  size_t hash() const override { return std::hash<int>()(node_); }

  int node_;
  bool done_;
};

/**
 * @brief A graph-based environment with complex topology.
 *
 * The graph can have multiple paths, dead ends, loops, and shortcuts.
 * The agent must navigate from a start node to a goal node.
 *
 * Actions correspond to edge indices from the current node (0 to max_edges-1).
 * Invalid actions (edges that don't exist) result in staying in place with a
 * small penalty.
 */
class GraphEnvironment : public Environment {
 public:
  struct Edge {
    int to;        // Target node
    float reward;  // Reward for taking this edge
    bool is_trap;  // If true, ends episode with penalty
  };

  /**
   * @brief Construct a GraphEnvironment.
   * @param num_nodes Total number of nodes in the graph.
   * @param max_edges Maximum number of outgoing edges per node (action space).
   * @param start_node The starting node index.
   * @param goal_node The goal node index.
   */
  GraphEnvironment(int num_nodes, int max_edges, int start_node, int goal_node);
  ~GraphEnvironment() override = default;

  /**
   * @brief Add a directed edge to the graph.
   * @param from Source node.
   * @param edge_index Action index (0 to max_edges-1).
   * @param to Target node.
   * @param reward Reward for taking this edge.
   * @param is_trap Whether this edge leads to a trap (episode termination).
   */
  void add_edge(int from, int edge_index, int to, float reward = 0.0f,
                bool is_trap = false);

  /**
   * @brief Create a predefined convoluted graph for testing.
   * @return A GraphEnvironment with a complex structure.
   */
  static GraphEnvironment create_convoluted_graph();

  void reset() override;
  Observation observe() override;
  Transition step(const Action& action) override;

  std::unique_ptr<GameState> snapshot() const override;
  void restore(const GameState& state) override;

  bool is_done() const override;

  ActionType get_action_type() const override;
  std::string name() const override;
  std::vector<int> get_observation_shape() const override;
  int get_action_space_size() const override;

  std::optional<Action> get_legal_mask() override;

  GraphEnvironment* clone() const override;

  void render() const override;

  // Accessors for graph structure
  int num_nodes() const { return num_nodes_; }
  int current_node() const { return state_.node_; }
  const std::vector<Edge>& get_edges(int node) const;

 private:
  GraphEnvironmentGameState state_;
  int num_nodes_;
  int max_edges_;
  int start_node_;
  int goal_node_;

  // Adjacency list: node -> vector of edges (indexed by action)
  // edges_[node][action_index] = Edge (or invalid if not set)
  std::vector<std::vector<Edge>> edges_;

  // Sentinel for invalid edges
  static constexpr int INVALID_NODE = -1;
};

}  // namespace talawa_ai::env
