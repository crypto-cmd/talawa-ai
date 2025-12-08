#pragma once
#include <chrono>
#include <iostream>
#include <string>

struct ScopedTimer {
  std::string name;
  std::chrono::steady_clock::time_point start;

  ScopedTimer(const std::string& func_name)
      : name(func_name), start(std::chrono::steady_clock::now()) {}

  ~ScopedTimer() {
    auto end = std::chrono::steady_clock::now();
    auto ms = (end - start).count() / 1e6;  // Convert to milliseconds
    auto seconds = ms / 1000.0;
    auto duration =
        static_cast<int>(seconds < 1.0 ? ms : seconds * 1000.0) / 1000.0;
    if (duration < 0.001) {
      duration = 0.001;  // Minimum display time
    }
    std::string unit = seconds < 1.0 ? "ms" : "s";
    std::cout << "[TIMER] " << name << ": " << duration << " " << unit << "\n";
  }
};

// Macro to make it easy to drop in anywhere
#define MEASURE_FUNCTION() ScopedTimer timer(__func__)
#define MEASURE_SCOPE(name) ScopedTimer timer(name)
