#pragma once
#include <raylib.h>

#include <string>

namespace talawa::visualizer {
struct WindowSize {
  int width;
  int height;
};

template <typename T>
class Visualizer {
 public:
  int value = 42;
  Visualizer() {};
  ~Visualizer() { CloseWindow(); }

  // Initialize the visualizer
  virtual void initialize(const WindowSize size,
                          const std::string& title) final {
    InitWindow(size.width, size.height, title.c_str());
    SetTargetFPS(10);
  }

  virtual void update(T& environment) = 0;  // For physics updates, if any
  virtual void draw() = 0;                  // For rendering per frame

  bool is_active() const { return !WindowShouldClose(); }
};

}  // namespace talawa::visualizer
