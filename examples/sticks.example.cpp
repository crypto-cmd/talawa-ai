#include "talawa/env/SticksGameEnvironment.hpp"
#include "talawa/rl/Arena.hpp"
#include "talawa/rl/QTable.hpp"
using namespace talawa;

int main() {
  auto env = env::StickGameEnv();

  // Create a Q-learning agent
  rl::agent::QTable ai(
      env.get_action_space(env::StickGameEnv::Player::PLAYER_1),
      {
          .learning_rate = 0.3f,
          .discount_factor = 0.99f,
          .epsilon = 1.0f,
          .starting_q_value = 0.0f,
          .update_rule = rl::agent::QTable::UpdateRule::ZeroSum,
      });
  // Q-learning agent is doing self-play
  env.register_agent(env::StickGameEnv::Player::PLAYER_1, ai, "QAgent1");
  env.register_agent(env::StickGameEnv::Player::PLAYER_2, ai, "QAgent2");

  talawa::arena::Arena arena(env);
  arena.tournament(10).print();
  for (int episode = 0; episode < 10000; ++episode) {
    arena.match();  // Train for 10000 episodes
    std::cout << "Completed episode " << episode + 1 << "/10000\r";
    ai.set_epsilon(std::max(0.015f, ai.get_epsilon() * 0.999f));
  }
  arena.tournament(10).print();
}
