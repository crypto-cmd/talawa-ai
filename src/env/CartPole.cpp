#include "talawa/env/CartPole.hpp"

#include <cmath>
#include <random>

namespace talawa::env {
CartPole::CartPole() : done_(false) {
  agent_order_ = {0};  // Single agent with ID 0
  reset();
}

void CartPole::reset(size_t) {
  // If random_seed is 0, use a random device for true randomness
  unsigned int seed = static_cast<unsigned int>(std::random_device{}());
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
  for (int i = 0; i < 4; ++i) {
    state[i] = dist(gen);
  }
  done_ = false;

  // Reset cumulative rewards
  cumulative_rewards_[0] = 0.0f;
  // Clear previous reports
  for (auto& [id, data] : agents_data_) {
    data.report = StepReport{};
  }
}
AgentID CartPole::get_active_agent() const {
  return agent_order_[0];  // Single agent environment
}

Observation CartPole::observe(const AgentID&) const {
  core::Matrix obs(1, 4);
  auto obs_space = get_observation_space(0);
  // Normalize observation to [-1, 1] based on observation space limits
  for (int i = 0; i < 4; ++i) {
    auto min_val = obs_space.low(i);
    auto max_val = obs_space.high(i);
    float range = max_val - min_val;
    float val = state[i];
    // Scale to [-1, 1]
    obs(0, i) = state[i] / max_val;
  }
  return Observation(obs);
}
void CartPole::step(const Action& action) {
  if (done_) {
    throw std::runtime_error(
        "Episode has terminated. Please reset the environment.");
  }
  // Update the current info report for the agent
  auto active_agent = get_active_agent();
  agents_data_.at(active_agent).report.previous_state = observe(active_agent);
  agents_data_.at(active_agent).report.action = action;
  // Extract action
  int action_value = static_cast<int>(action.item<int>());
  float force = (action_value == 1) ? force_mag : -force_mag;
  // Unpack state
  float x = state[0];
  float x_dot = state[1];
  float theta = state[2];
  float theta_dot = state[3];
  // Compute dynamics
  float costheta = std::cos(theta);
  float sintheta = std::sin(theta);
  float temp =
      (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
  float thetaacc =
      (gravity * sintheta - costheta * temp) /
      (length * (4.0f / 3.0f - masspole * costheta * costheta / total_mass));
  float xacc = temp - polemass_length * thetaacc;

  // Euler
  x += tau * x_dot;
  x_dot += tau * xacc;
  theta += tau * theta_dot;
  theta_dot += tau * thetaacc;

  state[0] = x;
  state[1] = x_dot;
  state[2] = theta;
  state[3] = theta_dot;

  // Check for termination
  done_ = x < -x_threshold || x > x_threshold ||
          theta < -theta_threshold_radians || theta > theta_threshold_radians;
  // Reward is 1 for every step taken except when pole falls
  float reward = done_ ? -1.0f : 1.0f;
  agents_data_.at(active_agent).report.reward = reward;
  cumulative_rewards_[active_agent] += reward;
  // Update resulting state
  agents_data_.at(active_agent).report.resulting_state = observe(active_agent);
  agents_data_.at(active_agent).report.episode_status =
      done_ ? EpisodeStatus::Terminated : EpisodeStatus::Running;
}

Space CartPole::get_action_space(const AgentID&) const {
  return Space::Discrete(2);  // Two actions: left (0), right (1)
}
Space CartPole::get_observation_space(const AgentID&) const {
  // state: [x, x_dot, theta, theta_dot]
  // x: (-2.4, 2.4)
  // x_dot: (-5, +5) (unbounded in theory)
  // theta: (+0.418, -0.418) (~24 degrees)
  // theta_dot: (+5, -5) (unbounded in theory)
  return Space::Continuous({4}, {-2.4f, -5.0f, -0.418f, -5.0f},
                           {2.4f, 5.0f, 0.418f, 5.0f});
}

StepReport CartPole::last(const AgentID& agentid) const {
  return agents_data_.at(agentid).report;
}

std::unique_ptr<IEnvironment> CartPole::clone() const {
  return std::make_unique<CartPole>(*this);
}

// // Rendering
void CartPole::render() {
  // Simple rendering using raylib
  BeginDrawing();
  ClearBackground(RAYWHITE);

  // Draw cart
  float cart_y = 300.0f;
  float cart_width = 100.0f;
  float cart_height = 20.0f;
  float cart_x = 400.0f + state[0] * 100.0f - cart_width / 2.0f;
  DrawRectangle(static_cast<int>(cart_x), static_cast<int>(cart_y),
                static_cast<int>(cart_width), static_cast<int>(cart_height),
                BLUE);

  // Draw pole
  float pole_length = 100.0f;
  float pole_x = cart_x + cart_width / 2.0f;
  float pole_y = cart_y;
  float pole_end_x = pole_x + pole_length * std::sin(state[2]);
  float pole_end_y = pole_y - pole_length * std::cos(state[2]);
  DrawLine(static_cast<int>(pole_x), static_cast<int>(pole_y),
           static_cast<int>(pole_end_x), static_cast<int>(pole_end_y), RED);

  EndDrawing();
}

void CartPole::update() {
  // For this simple environment, no dynamic updates are needed outside of
  //   step
}
}  // namespace talawa::env
