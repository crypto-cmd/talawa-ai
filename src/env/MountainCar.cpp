#include "talawa/env/MountainCar.hpp"

#include <random>

namespace talawa::env {

MountainCar::MountainCar(Friction friction)
    : IEnvironment({0}), friction(friction) {
  reset();
}

void MountainCar::reset(size_t) {
  // Random start position between -0.6 and -0.4
  unsigned int seed = static_cast<unsigned int>(std::random_device{}());
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-0.6f, -0.4f);

  state[0] = dist(gen);  // Position
  state[1] = 0.0f;       // Velocity

  done_ = false;
  cumulative_rewards_[0] = 0.0f;

  for (auto& [id, data] : agents_data_) data.report = StepReport{};
}

Observation MountainCar::observe(const AgentID&) const {
  core::Matrix obs(1, 2);
  // Normalize Inputs for Neural Net Stability (Crucial!)
  // Position: [-1.2, 0.6] -> Map roughly to [-1, 1]
  // Velocity: [-0.07, 0.07] -> Map roughly to [-1, 1]

  obs(0, 0) = (state[0] + 0.3f) / 0.9f;
  obs(0, 1) = state[1] / 0.07f;
  return Observation(obs);
}

void MountainCar::step(const Action& action) {
  if (done_) throw std::runtime_error("Env Terminated");

  auto active_agent = get_active_agent();
  agents_data_.at(active_agent).report.previous_state = observe(active_agent);
  agents_data_.at(active_agent).report.action = action;

  // Actions: 0 = Push Left, 1 = No Push, 2 = Push Right
  int act = static_cast<int>(action.item<float>());

  float position = state[0];
  float velocity = state[1];

  // Physics
  auto friction_Value = 1.0f;
  if (friction == Friction::LOW) {
    friction_Value = 0.99f;
  } else if (friction == Friction::MEDIUM) {
    friction_Value = 0.985f;
  } else if (friction == Friction::HIGH) {
    friction_Value = 0.975f;
  }
  velocity += (act - 1) * force + std::cos(3 * position) * (-gravity);
  velocity *= friction_Value;
  velocity = std::clamp(velocity, -max_speed, max_speed);

  position += velocity;
  position = std::clamp(position, min_position, max_position);

  // Wall collision (Left side acts as a wall, resets velocity)
  if (position == min_position && velocity < 0) velocity = 0;

  state[0] = position;
  state[1] = velocity;

  // Rewards: -1 per step until goal.
  // Standard RL uses -1.0, but for stability, we sometimes use -0.1
  bool reached_goal = (position >= goal_position);
  done_ = reached_goal;

  float reward = -1.f;
  if (reached_goal) reward = 0.0f;  // Alternatively give +100 here

  cumulative_rewards_[active_agent] += reward;

  agents_data_.at(active_agent).report.reward = reward;
  agents_data_.at(active_agent).report.resulting_state = observe(active_agent);
  agents_data_.at(active_agent).report.episode_status =
      done_ ? EpisodeStatus::Terminated : EpisodeStatus::Running;
}

Space MountainCar::get_action_space(const AgentID&) const {
  return Space::Discrete(3);
}

Space MountainCar::get_observation_space(const AgentID&) const {
  // 2 Continuous variables
  return Space::Continuous({2}, {-1.f, 1.f}, {-1.f, 1.f});
}

std::unique_ptr<IEnvironment> MountainCar::clone() const {
  return std::make_unique<MountainCar>(*this);
}

void MountainCar::render() {
  BeginDrawing();
  ClearBackground(RAYWHITE);

  int screen_w = 800;
  int screen_h = 400;

  // 1. Helper Lambda: Convert Physics World (X, Y) to Screen Pixels (X, Y)
  // Scale X: [-1.2, 0.6] -> [0, 800]
  // Scale Y: [-1.0, 1.0] -> [400, 0] (Inverted Y for screen)
  auto worldToScreen = [&](float wx, float wy) -> Vector2 {
    float sx = (wx - (-1.2f)) / (0.6f - (-1.2f)) * screen_w;
    float sy = (1.0f - (wy - (-1.0f)) / 2.0f) * screen_h;  // Map -1..1 to H..0
    // Shift Y down a bit so the mountain is centered
    sy += 50;
    return {sx, sy};
  };

  // 2. Draw the Track (The Sine Wave)
  // We draw connected lines across the screen width
  for (int i = 0; i < screen_w; i += 5) {
    // Convert Pixel i -> World X
    float wx1 = -1.2f + (static_cast<float>(i) / screen_w) * (0.6f - (-1.2f));
    float wy1 = std::sin(3 * wx1);

    float wx2 =
        -1.2f + (static_cast<float>(i + 5) / screen_w) * (0.6f - (-1.2f));
    float wy2 = std::sin(3 * wx2);

    Vector2 p1 = worldToScreen(wx1, wy1);
    Vector2 p2 = worldToScreen(wx2, wy2);

    DrawLineEx(p1, p2, 2.0f, DARKGRAY);
  }

  // 3. Draw the Goal (Flag)
  Vector2 goal_pos = worldToScreen(goal_position, std::sin(3 * goal_position));
  DrawLine(goal_pos.x, goal_pos.y, goal_pos.x, goal_pos.y - 30, RED);
  DrawTriangle({goal_pos.x, goal_pos.y - 30}, {goal_pos.x, goal_pos.y - 20},
               {goal_pos.x + 10, goal_pos.y - 25}, RED);

  // 4. Draw the Car
  float car_x = state[0];
  float car_y = std::sin(3 * car_x);
  Vector2 car_screen_pos = worldToScreen(car_x, car_y);

  // Draw simple car body
  DrawCircleV(car_screen_pos, 10.0f, BLUE);

  // Optional: Draw wheels for style
  DrawCircle(car_screen_pos.x - 8, car_screen_pos.y + 5, 4, BLACK);
  DrawCircle(car_screen_pos.x + 8, car_screen_pos.y + 5, 4, BLACK);

  // 5. Draw Info Stats
  DrawText("Mountain Car", 10, 10, 20, BLACK);
  DrawText(TextFormat("Pos: %.2f", state[0]), 10, 30, 20, DARKGRAY);
  DrawText(TextFormat("Vel: %.3f", state[1]), 10, 50, 20, DARKGRAY);

  EndDrawing();
}

}  // namespace talawa::env
