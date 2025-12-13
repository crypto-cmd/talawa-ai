#include <iostream>
#include <memory>
#include <talawa-ai/env/TicTacToeEnvironment.hpp>
#include <talawa-ai/rl/HumanAgent.hpp>
#include <talawa-ai/rl/QTable.hpp>
#include <talawa-ai/rl/Scheduler.hpp>
#include <talawa-ai/rl/SelfPlayTrainer.hpp>

using namespace talawa_ai;

int main() {
  auto env = env::TicTacToeEnvironment();
  rl::agent::QTable ai(env.get_action_space_size(), 0.2f, 0.95f, 1.0f);

  // Phase 1: Self-play training
  std::cout << "=== TIC TAC TOE AI TRAINER ===\n\n";
  std::cout << "Training via self-play...\n";

  rl::SelfPlayTrainer trainer(env, ai);

  auto config = rl::SelfPlayConfig{
      .episodes = 300000,
      .max_steps_per_game = 9,
  };

  // Add epsilon decay scheduler
  config.schedulers.add(
      rl::scheduler::schedule("epsilon")
          .use(
              rl::scheduler::chain()
                  .add(std::make_unique<rl::scheduler::ConstantScheduler>(1.0f),
                       100000)
                  .add(std::make_unique<rl::scheduler::ExponentialDecay>(
                           1.0f, 0.05f, 0.99999f),
                       rl::scheduler::UNTIL_END)
                  .build())
          .bind_to([&ai](float value) { ai.set_epsilon(value); })
          .on(rl::scheduler::ScheduleEvent::OnEpisodeEnd)
          .build());

  config.on_game_end = [&ai](int ep, env::GameOutcome outcome) {
    if (ep % 40000 == 0) {
      const char* result = (outcome == env::GameOutcome::Win)    ? "P1 Win"
                           : (outcome == env::GameOutcome::Loss) ? "P2 Win"
                                                                 : "Draw";
      std::cout << "Episode " << ep << " | Last: " << result
                << " | Epsilon: " << ai.get_epsilon()
                << " | Q-Table Size: " << ai.name() << "\n";
    }
  };
  auto result = trainer.train(std::move(config));

  std::cout << "\nTraining complete!\n";
  std::cout << "Results over " << result.total_episodes << " games:\n";
  std::cout << "  P1 (X) Wins: " << result.p1_wins << "\n";
  std::cout << "  P2 (O) Wins: " << result.p2_wins << "\n";
  std::cout << "  Draws:       " << result.draws << "\n";
  std::cout << "  Q-Table entries: " << ai.name() << "\n\n";

  ai.save("tictactoe");

  // Phase 2: Play against Self-trained AI
  ai.set_epsilon(0.0f);  // No exploration - pure exploitation

  // AI (agent1) vs AI (agent2)
  rl::SelfPlayTrainer play_session(env, ai, ai);
  auto outcome = play_session.play_game(true);

  std::cout << "\n=== SELF-PLAY GAME RESULT ===\n";
  if (outcome == env::GameOutcome::Win) {
    std::cout << "AI (X) wins!\n";
  } else if (outcome == env::GameOutcome::Loss) {
    std::cout << "AI (O) wins!\n";
  } else {
    std::cout << "It's a draw!\n";
  }

  return 0;
}
