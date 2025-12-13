#include "talawa-ai/env/GraphEnvironment.hpp"

#include <iostream>
#include <stdexcept>

namespace talawa_ai::env {

GraphEnvironment::GraphEnvironment(int num_nodes, int max_edges, int start_node,
                                   int goal_node)
    : state_(start_node, false),
      num_nodes_(num_nodes),
      max_edges_(max_edges),
      start_node_(start_node),
      goal_node_(goal_node),
      edges_(num_nodes) {
  // Initialize all edges as invalid
  for (int i = 0; i < num_nodes; ++i) {
    edges_[i].resize(max_edges, Edge{INVALID_NODE, 0.0f, false});
  }
}

void GraphEnvironment::add_edge(int from, int edge_index, int to, float reward,
                                bool is_trap) {
  if (from < 0 || from >= num_nodes_ || to < 0 || to >= num_nodes_) {
    throw std::invalid_argument("Invalid node index in add_edge");
  }
  if (edge_index < 0 || edge_index >= max_edges_) {
    throw std::invalid_argument("Invalid edge index in add_edge");
  }
  edges_[from][edge_index] = Edge{to, reward, is_trap};
}

GraphEnvironment GraphEnvironment::create_convoluted_graph() {
  /*
   * Creates a convoluted graph with 15 nodes:
   *
   *                    [0] START
   *                   / | \
   *                  /  |  \
   *                [1] [2] [3]
   *                /|\  |   |\
   *               / | \ |   | \
   *             [4][5][6]  [7][8]--TRAP
   *              |   \ | /  |
   *              |    \|/   |
   *             [9]   [10] [11]
   *              |    / \   |
   *              |   /   \  |
   *             [12]     [13]
   *               \       /
   *                \     /
   *                 [14] GOAL
   *
   * Features:
   * - Multiple paths to goal (some shorter, some longer)
   * - Dead ends and loops
   * - Trap nodes that end the episode with penalty
   * - Shortcuts with small penalties
   * - Cycles that can trap a naive agent
   */
  GraphEnvironment env(15, 4, 0,
                       14);  // 15 nodes, 4 max edges, start=0, goal=14

  // From node 0 (START) - 3 choices
  env.add_edge(0, 0, 1, -0.01f);  // Left path
  env.add_edge(0, 1, 2, -0.01f);  // Middle path
  env.add_edge(0, 2, 3, -0.01f);  // Right path

  // From node 1 - branches to 4, 5, 6
  env.add_edge(1, 0, 4, -0.01f);
  env.add_edge(1, 1, 5, -0.01f);
  env.add_edge(1, 2, 6, -0.01f);
  env.add_edge(1, 3, 0, -0.05f);  // Back to start (loop)

  // From node 2 - connects to 6
  env.add_edge(2, 0, 6, -0.01f);
  env.add_edge(2, 1, 0, -0.05f);  // Back to start

  // From node 3 - branches to 7, 8 (trap)
  env.add_edge(3, 0, 7, -0.01f);
  env.add_edge(3, 1, 8, -0.5f, true);  // TRAP! Looks tempting but bad
  env.add_edge(3, 2, 0, -0.05f);       // Back to start

  // From node 4 - leads to 9 (longer path)
  env.add_edge(4, 0, 9, -0.01f);
  env.add_edge(4, 1, 1, -0.02f);  // Back to 1 (small loop)

  // From node 5 - connects to 10
  env.add_edge(5, 0, 10, -0.01f);
  env.add_edge(5, 1, 6, -0.01f);  // Shortcut to 6

  // From node 6 - hub node, connects to 10
  env.add_edge(6, 0, 10, -0.01f);
  env.add_edge(6, 1, 5, -0.02f);  // Loop back
  env.add_edge(6, 2, 2, -0.02f);  // Loop back

  // From node 7 - connects to 11
  env.add_edge(7, 0, 11, -0.01f);
  env.add_edge(7, 1, 10, -0.01f);  // Cross-connect
  env.add_edge(7, 2, 3, -0.02f);   // Back

  // Node 8 is a TRAP - no outgoing edges (terminal state)
  // Agent loses if it goes here

  // From node 9 - connects to 12
  env.add_edge(9, 0, 12, -0.01f);
  env.add_edge(9, 1, 4, -0.03f);  // Loop

  // From node 10 - hub, connects to 12 and 13
  env.add_edge(10, 0, 12, -0.01f);
  env.add_edge(10, 1, 13, -0.01f);
  env.add_edge(10, 2, 6, -0.03f);  // Loop back

  // From node 11 - connects to 13
  env.add_edge(11, 0, 13, -0.01f);
  env.add_edge(11, 1, 7, -0.03f);  // Back

  // From node 12 - connects to goal
  env.add_edge(12, 0, 14, 1.0f);    // GOAL! Big reward
  env.add_edge(12, 1, 9, -0.03f);   // Loop back
  env.add_edge(12, 2, 10, -0.02f);  // Alternative

  // From node 13 - connects to goal
  env.add_edge(13, 0, 14, 1.0f);    // GOAL! Big reward
  env.add_edge(13, 1, 11, -0.03f);  // Back
  env.add_edge(13, 2, 10, -0.02f);  // Back

  // Node 14 is GOAL - terminal state (no edges needed)

  return env;
}

void GraphEnvironment::reset() {
  state_ = GraphEnvironmentGameState(start_node_, false);
}

Observation GraphEnvironment::observe() {
  // One-hot encoding of current node
  core::Matrix obs(1, num_nodes_);
  obs.fill(0.0f);
  if (state_.node_ >= 0 && state_.node_ < num_nodes_) {
    obs(0, state_.node_) = 1.0f;
  }
  return obs;
}

Transition GraphEnvironment::step(const Action& action) {
  if (state_.done_) {
    throw std::runtime_error(
        "Cannot step in a finished environment. Please reset.");
  }

  auto prev_state = snapshot();
  int action_idx = static_cast<int>(action(0, 0));

  // Clamp action to valid range
  if (action_idx < 0) action_idx = 0;
  if (action_idx >= max_edges_) action_idx = max_edges_ - 1;

  const Edge& edge = edges_[state_.node_][action_idx];
  float reward = 0.0f;

  if (edge.to == INVALID_NODE) {
    // Invalid action - stay in place with penalty
    reward = -0.1f;
  } else {
    // Valid action - move to new node
    state_.node_ = edge.to;
    reward = edge.reward;

    // Check for trap
    if (edge.is_trap) {
      state_.done_ = true;
      reward = -1.0f;  // Big penalty for trap
    }
    // Check for goal
    else if (state_.node_ == goal_node_) {
      state_.done_ = true;
      // reward already includes the edge reward (should be positive for goal)
    }
  }

  auto next_state = snapshot();
  return Transition{.state = std::move(prev_state),
                    .action = action,
                    .reward = reward,
                    .next_state = std::move(next_state),
                    .terminated = state_.done_};
}

std::unique_ptr<GameState> GraphEnvironment::snapshot() const {
  return std::make_unique<GraphEnvironmentGameState>(state_.node_,
                                                     state_.done_);
}

void GraphEnvironment::restore(const GameState& state) {
  const auto& s = dynamic_cast<const GraphEnvironmentGameState&>(state);
  state_.node_ = s.node_;
  state_.done_ = s.done_;
}

bool GraphEnvironment::is_done() const { return state_.done_; }

ActionType GraphEnvironment::get_action_type() const {
  return ActionType::DISCRETE;
}

std::string GraphEnvironment::name() const {
  return "GraphEnvironment [" + std::to_string(num_nodes_) + " nodes, " +
         std::to_string(max_edges_) + " actions]";
}

std::vector<int> GraphEnvironment::get_observation_shape() const {
  return {1, num_nodes_};
}

int GraphEnvironment::get_action_space_size() const { return max_edges_; }

std::optional<Action> GraphEnvironment::get_legal_mask() {
  core::Matrix mask(1, max_edges_);
  for (int i = 0; i < max_edges_; ++i) {
    // Action is legal if the edge exists (not INVALID_NODE)
    mask(0, i) = (edges_[state_.node_][i].to != INVALID_NODE) ? 1.0f : 0.0f;
  }
  return mask;
}

GraphEnvironment* GraphEnvironment::clone() const {
  return new GraphEnvironment(*this);
}

const std::vector<GraphEnvironment::Edge>& GraphEnvironment::get_edges(
    int node) const {
  return edges_[node];
}

void GraphEnvironment::render() const {
  // Helper to render a node with current position indicator
  auto node_str = [this](int n, int width = 4) -> std::string {
    std::string label = std::to_string(n);
    bool is_current = (n == state_.node_);
    bool is_start = (n == start_node_);
    bool is_goal = (n == goal_node_);

    std::string result;
    if (is_current) {
      result = "(" + label + ")";  // Current position
    } else if (is_goal) {
      result = "[" + label + "]";  // Goal
    } else if (is_start) {
      result = "<" + label + ">";  // Start
    } else {
      result = " " + label + " ";  // Normal
    }

    // Pad to width
    while (static_cast<int>(result.size()) < width) {
      result = " " + result;
    }
    return result;
  };

  // Check if node 8 is a trap (for the default graph)
  bool node8_is_trap = false;
  for (int i = 0; i < max_edges_; ++i) {
    if (edges_[3][i].to == 8 && edges_[3][i].is_trap) {
      node8_is_trap = true;
      break;
    }
  }

  std::cout << "\n";
  std::cout << "  ╔═══════════════════════════════════════════════════════╗\n";
  std::cout << "  ║           CONVOLUTED GRAPH ENVIRONMENT                ║\n";
  std::cout << "  ╠═══════════════════════════════════════════════════════╣\n";
  std::cout << "  ║                                                       ║\n";
  std::cout << "  ║                      " << node_str(0)
            << "  START                      ║\n";
  std::cout << "  ║                     ╱  │  ╲                           ║\n";
  std::cout << "  ║                    ╱   │   ╲                          ║\n";
  std::cout << "  ║                   ╱    │    ╲                         ║\n";
  std::cout << "  ║               " << node_str(1) << " " << node_str(2) << " "
            << node_str(3) << "                       ║\n";
  std::cout << "  ║              ╱│╲   │    │╲                            ║\n";
  std::cout << "  ║             ╱ │ ╲  │    │ ╲                           ║\n";
  std::cout << "  ║            ╱  │  ╲ │    │  ╲                          ║\n";
  std::cout << "  ║         " << node_str(4) << node_str(5) << node_str(6)
            << "  " << node_str(7) << node_str(8)
            << (node8_is_trap ? " ☠ TRAP" : "") << "           ║\n";
  std::cout << "  ║           │    ╲ │╱     │                             ║\n";
  std::cout << "  ║           │     ╲│      │                             ║\n";
  std::cout << "  ║         " << node_str(9) << "   " << node_str(10) << "   "
            << node_str(11) << "                        ║\n";
  std::cout << "  ║           │    ╱   ╲    │                             ║\n";
  std::cout << "  ║           │   ╱     ╲   │                             ║\n";
  std::cout << "  ║         " << node_str(12) << "         " << node_str(13)
            << "                       ║\n";
  std::cout << "  ║             ╲       ╱                                 ║\n";
  std::cout << "  ║              ╲     ╱                                  ║\n";
  std::cout << "  ║               ╲   ╱                                   ║\n";
  std::cout << "  ║                " << node_str(14)
            << "  ★ GOAL                       ║\n";
  std::cout << "  ║                                                       ║\n";
  std::cout << "  ╠═══════════════════════════════════════════════════════╣\n";

  // Status line
  std::string status;
  if (state_.done_) {
    if (state_.node_ == goal_node_) {
      status = "★ GOAL REACHED! ★";
    } else {
      status = "✗ TRAPPED / DEAD END";
    }
  } else {
    status = "Agent at node " + std::to_string(state_.node_);
  }

  // Center the status
  int padding = (55 - static_cast<int>(status.size())) / 2;
  std::cout << "  ║" << std::string(padding, ' ') << status
            << std::string(55 - padding - static_cast<int>(status.size()), ' ')
            << "║\n";

  std::cout << "  ╚═══════════════════════════════════════════════════════╝\n";
  std::cout << "     Legend: (n)=current  <n>=start  [n]=goal  ☠=trap\n";
  std::cout << "\n";
}

}  // namespace talawa_ai::env
