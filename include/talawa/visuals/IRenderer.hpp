#pragma once
#include <raylib.h>

#include <string>

namespace talawa::visualizer {
struct WindowSize {
  int width;
  int height;
};

class IRenderer {
 private:
  std::string window_title_;
  WindowSize window_size_;

 public:
  IRenderer(WindowSize size = {800, 600},
            std::string title = "Talawa Visualization")
      : window_title_(title), window_size_(size) {
    std::cout << "IRenderer created with title: " << window_title_ << "\n";
  }
  ~IRenderer() {
    CloseWindow();
    std::cout << "IRenderer destroyed for window: " << window_title_ << "\n";
  }

  void initRendering() {
    InitWindow(window_size_.width, window_size_.height, window_title_.c_str());
    SetTargetFPS(60);
    std::cout << "Rendering initialized for window: " << window_title_ << "\n";
  }

  bool rendering_initialized() const { return IsWindowReady(); }
  bool is_active() const { return !WindowShouldClose(); }
  virtual void render() = 0;
  virtual void update() = 0;
};

}  // namespace talawa::visualizer
