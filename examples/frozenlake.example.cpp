#include "talawa/env/FrozenLake.hpp"

#include "talawa/rl/Arena.hpp"
#include "talawa/rl/QTable.hpp"
using namespace talawa;

int main() {
  auto env = env::FrozenLake();

  // Create a Q-learning agent
  rl::agent::QTable ai(
      env.get_action_space(0),
      {
          .learning_rate = 0.3f,
          .discount_factor = 0.9f,
          .epsilon = 1.0f,
          .starting_q_value = 0.0f,
      });
  env.register_agent(0, ai, "QAgent1");
  auto tournamentConfig = talawa::arena::Arena::TournamentConfig{
      .rounds = 10,
      .max_steps = 30,
  };
  talawa::arena::Arena arena(env);
  arena.tournament(tournamentConfig).print();
  for (int episode = 0; episode < 10000; ++episode) {
    arena.match(30);  // Train for 10000 episodes
    std::cout << "Completed episode " << episode + 1 << "/10000\r";
    ai.set_epsilon(std::max(0.015f, ai.get_epsilon() * 0.999f));
  }
  arena.tournament(tournamentConfig).print();

  ai.print();
}
