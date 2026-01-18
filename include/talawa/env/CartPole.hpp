#pragma once
#include <cmath>

#include "talawa/env/interfaces/IEnvironment.hpp"
#include "talawa/visuals/IRenderer.hpp"
namespace talawa::env {
class CartPole : public IEnvironment, public visualizer::IRenderer {
 private:
  // --- Physics Constants ---
  const float gravity = 9.8f;
  const float masscart = 1.0f;
  const float masspole = 0.1f;
  const float total_mass = 1.1f;        // masscart + masspole
  const float length = 0.5f;            // actually half the pole's length
  const float polemass_length = 0.05f;  // masspole * length
  const float force_mag = 10.0f;
  const float tau = 0.02f;  // seconds between state updates

  int steps_ = 0;
  // --- Thresholds for Game Over ---
  const float theta_threshold_radians = 15 * 2 * M_PI / 360;  // ~15 degrees
  const float x_threshold = 2.4f;

  // --- Current State ---
  // [x, x_dot, theta, theta_dot]
  float state[4];

 public:
  CartPole();
  ~CartPole() = default;

  void reset(size_t random_seed = 42) override;

  Observation observe(const AgentID&) const override;
  void step(const Action& action) override;
  StepReport last(const AgentID&) const override;

  Space get_action_space(const AgentID&) const override;
  Space get_observation_space(const AgentID&) const override;

  std::unique_ptr<IEnvironment> clone() const override;

  // Rendering
  void render() override;
  void update() override;
};
}  // namespace talawa::env
