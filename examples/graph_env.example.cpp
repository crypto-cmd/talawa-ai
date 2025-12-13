#include <iostream>
#include <talawa-ai/env/GraphEnvironment.hpp>
#include <talawa-ai/rl/QTable.hpp>

using namespace talawa_ai;

int main() {
  // Create a convoluted graph environment
  auto env = env::GraphEnvironment::create_convoluted_graph();

  std::cout << "Environment: " << env.name() << "\n";
  std::cout << "Nodes: " << env.num_nodes() << "\n";
  std::cout << "Actions: " << env.get_action_space_size() << "\n\n";

  // Q-learning agent with appropriate parameters for the graph
  rl::agent::QTable agent(env.get_action_space_size(), 0.2f,  // learning rate
                          0.95f,                              // discount
                          1.0f,                               // epsilon start
                          0.9999f,                            // epsilon decay
                          0.05f);                             // epsilon min

  const int num_episodes = 50000;
  int successful_episodes = 0;
  float total_reward = 0.0f;

  for (int episode = 0; episode < num_episodes; ++episode) {
    env.reset();
    auto state = env.snapshot();
    float episode_reward = 0.0f;
    int steps = 0;

    while (!env.is_done() && steps++ < 100) {
      auto legal_mask = env.get_legal_mask();
      auto action = agent.act(*state, legal_mask, true);

      auto transition = env.step(action);
      agent.update(transition);

      episode_reward += transition.reward;
      state = std::move(transition.next_state);
    }

    agent.decay_epsilon();

    // Check if agent reached the goal (node 14)
    if (env.current_node() == 14) {
      successful_episodes++;
    }
    total_reward += episode_reward;

    // Print progress every 10000 episodes
    if ((episode + 1) % 10000 == 0) {
      float success_rate = 100.0f * successful_episodes / (episode + 1);
      float avg_reward = total_reward / (episode + 1);
      std::cout << "Episode " << (episode + 1)
                << " | Success Rate: " << success_rate
                << "% | Avg Reward: " << avg_reward << "\n";
    }
  }

  std::cout << "\n=== Training Complete ===\n";
  std::cout << "Final Success Rate: "
            << (100.0f * successful_episodes / num_episodes) << "%\n\n";

  std::cout << "Final Q-Table: " << agent.name() << "\n";
  agent.print();

  // Demo: Run one episode with the trained agent (greedy policy)
  std::cout << "\n=== Demo Episode (Greedy Policy) ===\n";
  env.reset();
  env.render();

  auto state = env.snapshot();
  int step = 0;

  while (!env.is_done() && step++ < 20) {
    auto legal_mask = env.get_legal_mask();
    auto action = agent.act(*state, legal_mask, false);  // greedy

    // std::cout << "\nStep " << step << ": Taking action "
    //           << static_cast<int>(action(0, 0)) << "\n";

    auto transition = env.step(action);
    state = std::move(transition.next_state);

    std::cout << "Reward: " << transition.reward << "\n";
    env.render();
  }

  return 0;
}
