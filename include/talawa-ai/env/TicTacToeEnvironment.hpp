#pragma once

#include <array>
#include <iostream>
#include <random>
#include <sstream>

#include "talawa-ai/env/TwoPlayerEnvironment.hpp"

namespace talawa_ai::env {

class TicTacToeState : public GameState {
 public:
  std::array<int, 9> board;
  int current_player;
  bool done;
  int winner;  // 0=none/draw, 1=X wins, 2=O wins

  TicTacToeState()
      : board{}, current_player(PLAYER_1), done(false), winner(0) {}

  std::unique_ptr<GameState> clone() const override {
    return std::make_unique<TicTacToeState>(*this);
  }

  bool equals(const GameState& other) const override {
    const auto* o = dynamic_cast<const TicTacToeState*>(&other);
    if (!o) return false;
    return board == o->board && current_player == o->current_player;
  }

  size_t hash() const override {
    /** Hash based on board state observed by current player and whose turn it
    is 0,1,2 for each cell + current player, eg: "1202001001" is 120200100 | 1
    where 1 is current player, 2 is opponent, and the rest is the board state
    **/
    std::stringstream ss;
    for (int cell : board) {
      ss << (cell == current_player ? '1' : (cell == 0 ? '0' : '2'));
    }
    // ss << (current_player == PLAYER_1
    //            ? '1'
    //            : '2');  // Append current player at the end
    std::string state_str = ss.str();
    size_t hash_val = 0;
    try {
      hash_val = static_cast<size_t>(std::stoull(state_str));
    } catch (const std::exception& e) {
      // Fallback: use a simple sum of digits if conversion fails
      for (char c : state_str) {
        if (c >= '0' && c <= '9') hash_val = hash_val * 10 + (c - '0');
      }
    }
    return hash_val;
  }
};

class TicTacToeEnvironment : public TwoPlayerEnvironment {
 public:
  TicTacToeEnvironment() { reset(); }

  void reset() override { state_ = TicTacToeState(); }

  Observation observe() override {
    // 9 cells + current player
    core::Matrix obs(1, 10);
    for (int i = 0; i < 9; ++i) {
      obs(0, i) = static_cast<float>(state_.board[i]);
    }
    obs(0, 9) = static_cast<float>(state_.current_player);
    return obs;
  }

  Transition step(const Action& action) override {
    auto prev_state = snapshot();
    int cell = static_cast<int>(action(0, 0));

    float reward = 0.0f;

    if (cell < 0 || cell > 8 || state_.board[cell] != 0) {
      // Invalid move - penalty and end game
      reward = -10.0f;
      state_.done = true;
    } else {
      state_.board[cell] = state_.current_player;

      if (check_win(state_.current_player)) {
        state_.winner = state_.current_player;
        state_.done = true;
        reward = 1.0f;  // Win
      } else if (is_board_full()) {
        state_.done = true;
        reward = 0.01f;  // Draw (slight positive)
      } else {
        // Switch player
        state_.current_player =
            (state_.current_player == PLAYER_1) ? PLAYER_2 : PLAYER_1;
        reward = -0.1f;
      }
    }

    auto next_state = snapshot();
    return Transition{
        .state = std::move(prev_state),
        .action = action,
        .reward = reward,
        .next_state = std::move(next_state),
        .terminated = state_.done,
    };
  }

  std::unique_ptr<GameState> snapshot() const override {
    return std::make_unique<TicTacToeState>(state_);
  }

  void restore(const GameState& state) override {
    state_ = dynamic_cast<const TicTacToeState&>(state);
  }

  bool is_done() const override { return state_.done; }

  int current_player() const override { return state_.current_player; }

  int get_winner() const { return state_.winner; }

  GameOutcome outcome_for(int player) const override {
    if (!state_.done) return GameOutcome::Ongoing;
    if (state_.winner == 0) return GameOutcome::Draw;
    return (state_.winner == player) ? GameOutcome::Win : GameOutcome::Loss;
  }

  ActionType get_action_type() const override { return ActionType::DISCRETE; }
  std::string name() const override { return "TicTacToe"; }
  std::vector<int> get_observation_shape() const override { return {1, 10}; }
  int get_action_space_size() const override { return 9; }

  std::optional<Action> get_legal_mask() override {
    core::Matrix mask(1, 9);
    for (int i = 0; i < 9; ++i) {
      mask(0, i) = (state_.board[i] == 0) ? 1.0f : 0.0f;
    }
    return mask;
  }

  TicTacToeEnvironment* clone() const override {
    return new TicTacToeEnvironment(*this);
  }

  void render() const override { render(0); }

  void render(int perspective) const override {
    (void)perspective;
    std::cout << "\n";
    for (int row = 0; row < 3; ++row) {
      std::cout << " ";
      for (int col = 0; col < 3; ++col) {
        int cell = state_.board[row * 3 + col];
        char c = (cell == 0) ? '.' : (cell == 1) ? 'X' : 'O';
        std::cout << c;
        if (col < 2) std::cout << " | ";
      }
      std::cout << "\n";
      if (row < 2) std::cout << "---+---+---\n";
    }
    std::cout << "\n";
  }

 private:
  TicTacToeState state_;

  bool check_win(int player) const {
    const int lines[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8},  // rows
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8},  // cols
        {0, 4, 8}, {2, 4, 6}              // diagonals
    };
    for (const auto& line : lines) {
      if (state_.board[line[0]] == player && state_.board[line[1]] == player &&
          state_.board[line[2]] == player) {
        return true;
      }
    }
    return false;
  }

  bool is_board_full() const {
    for (int i = 0; i < 9; ++i) {
      if (state_.board[i] == 0) return false;
    }
    return true;
  }
};

}  // namespace talawa_ai::env
