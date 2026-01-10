#include "talawa/env/TicTacToe.hpp"

#include "talawa/env/SticksGameEnvironment.hpp"
#include "talawa/rl/Arena.hpp"
#include "talawa/rl/HumanAgent.hpp"
#include "talawa/rl/QTable.hpp"
using namespace talawa;

int main() {
  auto env = env::TicTacToe();

  // Create a Q-learning agent
  auto as = env.get_action_space(0);
  rl::agent::QTable ai(
      as, {
              .learning_rate = 0.2f,
              .discount_factor = 0.99f,
              .epsilon = 1.0f,
              .starting_q_value = 0.0f,
              .update_rule = rl::agent::QTable::UpdateRule::ZeroSum,
          });
  // Q-learning agent is doing self-play
  env.register_agent(0, ai, "QAgent1");
  env.register_agent(1, ai, "QAgent2");
  auto tournamentConfig = talawa::arena::Arena::TournamentConfig{
      .rounds = 10,
      .max_steps = 30,
  };
  talawa::arena::Arena arena(env);
  arena.tournament(tournamentConfig).print();
  for (int episode = 0; episode < 100000; ++episode) {
    arena.match(50);  // Train for 10000 episodes
    std::cout << "Completed episode " << episode + 1 << "/100000\r";
    if (episode > 40000) {
      ai.set_epsilon(std::max(0.015f, ai.get_epsilon() * 0.999f));
      ai.set_learning_rate(std::max(0.05f, ai.get_learning_rate() * 0.999f));
    }
  }
  arena.tournament(tournamentConfig).print();

  // ai.print();
  std::cout << "Final Q-Table size: " << ai.getQTable().size() << std::endl;

  rl::agent::HumanAgent human(as.n());

  std::cout << "Starting a match against the trained Q-agent!\n";
  env.register_agent(0, human, "HumanPlayer");
  env.register_agent(1, ai, "TrainedQAgent");
  arena.match(50, false);  // One match with human
}
