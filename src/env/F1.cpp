#include "talawa/env/F1.hpp"

#include <raymath.h>  // Useful Raylib math functions

#include <algorithm>
#include <random>

namespace talawa::env {
namespace racing {
Vector2 ProceduralTrack::get_spline_point(float t, Vector2 p0, Vector2 p1,
                                          Vector2 p2, Vector2 p3) {
  // Catmull-Rom implementation
  float t2 = t * t;
  float t3 = t2 * t;

  float x = 0.5f * ((2.0f * p1.x) + (-p0.x + p2.x) * t +
                    (2.0f * p0.x - 5.0f * p1.x + 4.0f * p2.x - p3.x) * t2 +
                    (-p0.x + 3.0f * p1.x - 3.0f * p2.x + p3.x) * t3);
  float y = 0.5f * ((2.0f * p1.y) + (-p0.y + p2.y) * t +
                    (2.0f * p0.y - 5.0f * p1.y + 4.0f * p2.y - p3.y) * t2 +
                    (-p0.y + 3.0f * p1.y - 3.0f * p2.y + p3.y) * t3);
  return {x, y};
}

void ProceduralTrack::generate(int seed) {
  mesh_ = TrackMesh();  // Clear old data
  anchors_.clear();

  std::mt19937 gen(seed);
  // Safer parameters to avoid self-intersection loops
  std::uniform_real_distribution<float> dist_radius(250.0f, 450.0f);
  std::uniform_real_distribution<float> dist_jitter(-0.2f, 0.2f);

  // 1. Generate Anchors (Circular)
  float angle_step = (2.0f * PI) / num_anchors_;
  for (int i = 0; i < num_anchors_; ++i) {
    float angle = (i * angle_step) + dist_jitter(gen);
    float radius = dist_radius(gen);
    float x = std::cos(angle) * radius + 800.0f;  // Center X
    float y = std::sin(angle) * radius + 600.0f;  // Center Y
    anchors_.push_back({x, y});
  }

  // 2. Build the Spline (High Resolution)
  // We calculate just the CENTER line first
  std::vector<Vector2> raw_center;
  int points_per_segment = 20;

  for (int i = 0; i < num_anchors_; ++i) {
    Vector2 p0 = anchors_[(i - 1 + num_anchors_) % num_anchors_];
    Vector2 p1 = anchors_[i];
    Vector2 p2 = anchors_[(i + 1) % num_anchors_];
    Vector2 p3 = anchors_[(i + 2) % num_anchors_];

    for (int s = 0; s < points_per_segment; ++s) {
      float t = (float)s / points_per_segment;
      raw_center.push_back(get_spline_point(t, p0, p1, p2, p3));
    }
  }

  // 3. Extrude Walls (With Smoothing)
  // This is where we prevent artifacts.
  for (size_t i = 0; i < raw_center.size(); ++i) {
    Vector2 current = raw_center[i];

    // Look ahead 2 points for a smoother normal (avoids jitter)
    size_t next_idx = (i + 2) % raw_center.size();
    Vector2 next = raw_center[next_idx];

    Vector2 dir = Vector2Normalize(Vector2Subtract(next, current));
    Vector2 normal = {-dir.y, dir.x};  // Rotate 90 deg

    // Store data
    mesh_.center_line.push_back(current);
    mesh_.left_verts.push_back(
        Vector2Add(current, Vector2Scale(normal, track_width_ / 2.0f)));
    mesh_.right_verts.push_back(
        Vector2Subtract(current, Vector2Scale(normal, track_width_ / 2.0f)));
  }
}

void ProceduralTrack::draw() const {
  if (mesh_.center_line.empty()) return;

  // Use specific colors for clarity
  Color tarmac = {19, 10, 6, 255};
  Color wall = {200, 0, 0, 255};

  // We draw the track as a series of Quads (2 Triangles per segment)
  size_t count = mesh_.center_line.size();

  for (size_t i = 0; i < count; ++i) {
    size_t next = (i + 1) % count;

    Vector2 l1 = mesh_.left_verts[i];
    Vector2 r1 = mesh_.right_verts[i];
    Vector2 l2 = mesh_.left_verts[next];
    Vector2 r2 = mesh_.right_verts[next];

    // 1. Draw Road Surface (No gaps!)
    DrawTriangle(l1, r1, l2, tarmac);
    DrawTriangle(r1, r2, l2, tarmac);

    // 2. Draw Walls
    // Instead of DrawLineEx, we draw THIN QUADS for walls.
    // This is 100% artifact free because it's just geometry.
    float wall_thick = 4.0f;

    // Left Wall
    // Calculate outer edge for the wall
    Vector2 l1_out = Vector2Add(
        l1,
        Vector2Scale(Vector2Normalize(Vector2Subtract(l1, r1)), wall_thick));
    Vector2 l2_out = Vector2Add(
        l2,
        Vector2Scale(Vector2Normalize(Vector2Subtract(l2, r2)), wall_thick));

    DrawTriangle(l1_out, l1, l2_out, wall);
    DrawTriangle(l1, l2, l2_out, wall);

    // Right Wall
    Vector2 r1_out = Vector2Add(
        r1,
        Vector2Scale(Vector2Normalize(Vector2Subtract(r1, l1)), wall_thick));
    Vector2 r2_out = Vector2Add(
        r2,
        Vector2Scale(Vector2Normalize(Vector2Subtract(r2, l2)), wall_thick));

    DrawTriangle(r1, r1_out, r2, wall);
    DrawTriangle(r1_out, r2_out, r2, wall);

    // 3. Center Line (Simple lines are fine here as they don't connect)
    if (i % 8 < 4) {
      DrawLineV(mesh_.center_line[i], mesh_.center_line[next], YELLOW);
    }
  }
}
}  // namespace racing

F1::F1()
    : IEnvironment({0}),
      IRenderer({1280, 720}, "F1"),
      car_speed_(0.0f),
      car_steering_(0.0f),
      steps_(0) {
  track_.generate(42);  // Default seed
  reset();
  initRendering();

  // 1. Target: Where the camera looks (Center of the track)
  camera_.target = {800.0f, 600.0f};

  // 2. Offset: Where that target appears on screen (Center of the screen)
  camera_.offset = {1280.0f / 2.0f, 720.0f / 2.0f};

  // 3. Rotation: 0
  camera_.rotation = 0.0f;

  // 4. Zoom: 0.6x lets you see the whole 1600x1200 world in a 720p window
  camera_.zoom = 0.6f;
}
void F1::reset(size_t) {
  // Reset Car State
  car_position_ = track_.get_start_position();
  car_speed_ = 0.0f;
  car_steering_ = 0.0f;
  done_ = false;
  steps_ = 0;

  // Reset cumulative rewards
  cumulative_rewards_[0] = 0.0f;
  // Clear previous reports
  for (auto& [id, data] : agents_data_) {
    data.report = StepReport{};
  }
}
AgentID F1::get_active_agent() const {
  return agent_order_[0];  // Single agent environment
}
Observation F1::observe(const AgentID&) const {
  core::Matrix obs(1, 4);
  // Normalize position and speed for neural net stability
  obs(0, 0) = car_position_.x / 1600.0f;  // Assuming world width ~1600
  obs(0, 1) = car_position_.y / 1200.0f;  // Assuming world height ~1200
  obs(0, 2) = car_speed_ / 200.0f;        // Max speed ~200
  obs(0, 3) = car_steering_ / (PI / 4);   // Max steering angle ~45 degrees
  return Observation(obs);
}
void F1::step(const Action& action) {}
Space F1::get_action_space(const AgentID&) const {
  // Action: [Acceleration (-1 to 1), Steering (-1 to 1)]
  return Space::Continuous({2}, {{-1.0f, -1.0f}}, {{1.0f, 1.0f}});
}
Space F1::get_observation_space(const AgentID&) const {
  // Observation: [x_pos, y_pos, speed, steering_angle]
  return Space::Continuous({4}, {{0.0f, 0.0f, 0.0f, -1.0f}},
                           {{1.0f, 1.0f, 1.0f, 1.0f}});
}
std::unique_ptr<IEnvironment> F1::clone() const {
  return std::make_unique<F1>(*this);
}

void F1::render() {
  BeginDrawing();
  ClearBackground(RAYWHITE);

  BeginMode2D(camera_);
  // Draw Track
  track_.draw();
  EndMode2D();

  //   // Draw Car (Triangle for simplicity)
  //   Vector2 car_points[3];
  //   car_points[0] = {
  //       car_position_.x + std::cos(car_angle_) * 15.0f,
  //       car_position_.y + std::sin(car_angle_) * 15.0f,
  //   };
  //   car_points[1] = {
  //       car_position_.x + std::cos(car_angle_ + 2.5f) * 10.0f,
  //       car_position_.y + std::sin(car_angle_ + 2.5f) * 10.0f,
  //   };
  //   car_points[2] = {
  //       car_position_.x + std::cos(car_angle_ - 2.5f) * 10.0f,
  //       car_position_.y + std::sin(car_angle_ - 2.5f) * 10.0f,
  //   };
  //   DrawTriangle(car_points[0], car_points[1], car_points[2], RED);

  DrawText("Speed: 120 km/h", 10, 10, 20, BLACK);
  EndDrawing();
}
void F1::update() {
  // No dynamic elements to update for now
}

StepReport F1::last(const AgentID& agentid) const {
  return agents_data_.at(agentid).report;
}

}  // namespace talawa::env
