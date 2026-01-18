#pragma once
#include <raylib.h>

#include <cmath>
#include <vector>

#include "talawa/env/interfaces/IEnvironment.hpp"
#include "talawa/visuals/IRenderer.hpp"
namespace talawa::env {

namespace racing {
struct TrackMesh {
  // We store the raw vertices for the road edges
  std::vector<Vector2> left_verts;
  std::vector<Vector2> right_verts;
  std::vector<Vector2> center_line;
};

class ProceduralTrack {
 private:
  const int num_anchors_ = 16;
  const float track_width_ = 70.0f;  // Slightly wider for better drift

  // The geometry data
  TrackMesh mesh_;
  std::vector<Vector2> anchors_;

  // Helpers
  Vector2 get_spline_point(float t, Vector2 p0, Vector2 p1, Vector2 p2,
                           Vector2 p3);

 public:
  ProceduralTrack() = default;

  void generate(int seed);
  void draw() const;

  // Accessors for physics
  const TrackMesh& get_mesh() const { return mesh_; }
  Vector2 get_start_position() const {
    return mesh_.center_line.empty() ? Vector2{0, 0} : mesh_.center_line[0];
  }
};
}  // namespace racing

class F1 : public IEnvironment, public visualizer::IRenderer {
 public:
  F1();
  ~F1() override = default;

  void reset(size_t random_seed = 42) override;
  AgentID get_active_agent() const override;

  Observation observe(const AgentID&) const override;
  void step(const Action& action) override;

  Space get_action_space(const AgentID&) const override;
  Space get_observation_space(const AgentID&) const override;

  StepReport last(const AgentID& agentid) const override;

  std::unique_ptr<IEnvironment> clone() const override;

  bool is_done() const override { return done_; }
  // Rendering
  void render() override;
  void update() override;

 private:
  Camera2D camera_;  // <--- Add this
  racing::ProceduralTrack track_;
  Vector2 car_position_;
  float car_angle_;     // In radians
  float car_speed_;     // Current speed
  float car_steering_;  // Current steering angle
  int steps_;           // Step counter
};
}  // namespace talawa::env
