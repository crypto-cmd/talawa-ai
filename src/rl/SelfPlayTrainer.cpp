#include "talawa-ai/rl/SelfPlayTrainer.hpp"

#include <iostream>
#include <memory>
namespace talawa_ai::rl {

SelfPlayTrainer::SelfPlayTrainer(env::TwoPlayerEnvironment& env,
                                 agent::Agent& agent)
    : env_(env), agent1_(agent), agent2_(agent), same_agent_(true) {}

SelfPlayTrainer::SelfPlayTrainer(env::TwoPlayerEnvironment& env,
                                 agent::Agent& agent1, agent::Agent& agent2)
    : env_(env), agent1_(agent1), agent2_(agent2), same_agent_(false) {}

agent::Agent& SelfPlayTrainer::current_agent() {
  return (env_.current_player() == 1) ? agent1_ : agent2_;
}
SelfPlayResult SelfPlayTrainer::train(SelfPlayConfig config) {
  SelfPlayResult result;
  config.schedulers.initialize();

  for (int i = 0; i < config.episodes; ++i) {
    result.total_episodes++;

    env_.reset();

    auto history_p1 = env::Transition();
    auto history_p2 = env::Transition();

    auto turn = env::PLAYER_1;
    auto state = env_.snapshot();

    auto steps = 0;

    while (!env_.is_done() && steps < config.max_steps_per_game) {
      steps++;

      // Observe
      state = env_.snapshot();  // Get the newest state

      // Learn from the previous move
      if (turn == env::PLAYER_1 && history_p1.state != nullptr) {
        auto prev_state = history_p1.state->clone();
        auto prev_action = history_p1.action;
        auto reward = history_p1.reward;

        auto new_state = state->clone();  // State after P2's move

        agent1_.observe(env::Transition{
            .state = std::move(prev_state),
            .action = std::move(prev_action),
            .reward = reward,
            .next_state = std::move(new_state),
            .terminated = false,
        });
        agent1_.learn();
      } else if (turn == env::PLAYER_2 && history_p2.state != nullptr) {
        auto prev_state = history_p2.state->clone();
        auto prev_action = history_p2.action;
        auto reward = history_p2.reward;

        auto new_state = state->clone();  // State after P2's move

        agent1_.observe(env::Transition{
            .state = std::move(prev_state),
            .action = std::move(prev_action),
            .reward = reward,
            .next_state = std::move(new_state),
            .terminated = false,
        });
        agent1_.learn();
      }

      // Act
      auto mask = env_.get_legal_mask();
      auto action = agent1_.act(*state, mask, true);

      auto transition = env_.step(action);

      if (turn == env::PLAYER_1) {
        // Svae the stat that was before taking action
        history_p1.state = state->clone();
        history_p1.action = action;
        history_p1.reward = transition.reward;
      } else {
        history_p2.state = state->clone();
        history_p2.action = action;
        history_p2.reward = transition.reward;
      }

      // Check terminal states
      if (transition.terminated) {
        if (env_.outcome_for(turn) == env::GameOutcome::Win) {
          result.p1_wins += (turn == env::PLAYER_1) ? 1 : 0;
          result.p2_wins += (turn == env::PLAYER_2) ? 1 : 0;
          result.draws += 0;

          // Update winner agent where he played just won from state -> terminal
          auto winning_transition = env::Transition{
              .state = std::move(transition.state->clone()),
              .action = transition.action,
              .reward = transition.reward,
              .next_state = std::move(transition.next_state->clone()),
              .terminated = true,
          };
          agent1_.observe(std::move(winning_transition));
          agent1_.learn();

          // Update loser agent where he played bad from previous state ->
          // terminal
          auto* loser_history =
              (turn == env::PLAYER_1) ? &history_p2 : &history_p1;
          auto losing_transition = env::Transition{
              .state = std::move(loser_history->state->clone()),
              .action = loser_history->action,
              .reward = -transition.reward,  // Assume symmetric reward
              .next_state = std::move(transition.next_state->clone()),
              .terminated = true,
          };
          agent1_.observe(std::move(losing_transition));
          agent1_.learn();

        } else if (env_.outcome_for(turn) == env::GameOutcome::Draw) {
          result.draws += 1;
          agent1_.observe(std::move(transition));
          agent1_.learn();
        }
      }
      turn = turn == env::PLAYER_1 ? env::PLAYER_2 : env::PLAYER_1;
    }
    config.schedulers.on_episode_end({
        .episode = i,
        .step = steps,
        .total_steps = result.total_episodes * steps,
        .episode_reward = 0.0f,
        .last_reward = 0.0f,
    });
    if (config.on_game_end) {
      config.on_game_end(i, env_.outcome_for(turn));
    }
  }
  return result;
}


env::GameOutcome SelfPlayTrainer::play_game(bool render) {
  env_.reset();

  while (!env_.is_done()) {
    if (render) {
      env_.render(env_.current_player());
    }

    auto& agent = current_agent();
    auto state = env_.snapshot();
    auto mask = env_.get_legal_mask();
    auto action = agent.act(*state, mask, false);  // Exploit mode
    env_.step(action);
  }

  if (render) {
    env_.render(0);
  }

  return env_.outcome_for(1);
}

}  // namespace talawa_ai::rl
