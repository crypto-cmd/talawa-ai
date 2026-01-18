#pragma once
#include <algorithm>
#include <cmath>

#include "talawa/env/interfaces/IEnvironment.hpp"
#include "talawa/visuals/IRenderer.hpp"

namespace talawa::env {

class MountainCar : public IEnvironment, public visualizer::IRenderer {
 public:
  enum class Friction { NONE, LOW, MEDIUM, HIGH };

 private:
  // Physics constants
  const float min_position = -1.2f;
  const float max_position = 0.6f;
  const float max_speed = 0.07f;
  const float goal_position = 0.5f;
  const float force = 0.001f;
  const float gravity = 0.0025f;

  const Friction friction = Friction::NONE;

  float state[2];  // [position, velocity]

 public:
  MountainCar(Friction friction = Friction::NONE);
  ~MountainCar() = default;

  void reset(size_t random_seed = 42) override;

  Observation observe(const AgentID&) const override;
  void step(const Action& action) override;

  Space get_action_space(const AgentID&) const override;
  Space get_observation_space(const AgentID&) const override;

  std::unique_ptr<IEnvironment> clone() const override;

  // --- Rendering Methods ---
  void render() override;
  void update() override {};  // Standard update loop
};
}  // namespace talawa::env
