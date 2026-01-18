#include <raylib.h>

#include <iostream>

#include "talawa/env/F1.hpp"
#include "talawa/env/MountainCar.hpp"
#include "talawa/rl/HumanAgent.hpp"

using namespace talawa;

int main() {
  // 1. Setup
  env::F1 env;
  env.initRendering();  // Opens the Raylib window

  std::cout << "--- Mountain Car Interactive Mode ---\n";
  std::cout << "Controls:\n";
  std::cout << "  [LEFT ARROW]  Push Left\n";
  std::cout << "  [RIGHT ARROW] Push Right\n";
  std::cout << "  [NO KEY]      Neutral (Coast)\n";
  std::cout << "-------------------------------------\n";

  auto human = rl::agent::HumanAgent(2);
  env.register_agent(0, human, "HumanPlayer");
  // 2. Game Loop
  while (!env.is_done() && env.is_active()) {
    // A. Get Input
    // int action_idx = 1;  // Default: 1 = Neutral

    // if (IsKeyDown(KEY_LEFT)) {
    //   action_idx = 0;  // Push Left
    // } else if (IsKeyDown(KEY_RIGHT)) {
    //   action_idx = 2;  // Push Right
    // }

    // B. Step Physics
    // Wrap integer action into a Matrix
    // auto action = core::Matrix({{0, 0}});
    // env.step(action);

    // C. Render
    env.render();
    env.update();
  }

  if (env.is_done()) {
    std::cout << "SUCCESS! You reached the goal.\n";
  }

  return 0;
}
