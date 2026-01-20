#pragma once

#include <vector>

#include "talawa/core/Matrix.hpp"
#include "talawa/env/interfaces/IEnvironment.hpp"
#include "talawa/env/types.hpp"
namespace talawa::rl::memory {
struct Experience {
  core::Matrix states;
  core::Matrix next_states;
  core::Matrix actions;
  std::vector<float> rewards;
  std::vector<env::EpisodeStatus> dones;
};

class ReplayBuffer {
 private:
  Experience buffer_;
  Experience sample_;
  int max_size_;
  int size_ = 0;
  int cursor_ = 0;

  int expected_sample_size_ = 1;

 public:
  ReplayBuffer(int max_size);
  ~ReplayBuffer() = default;
  Experience sample(size_t batch_size = 1);
  void add(const env::Transition& transition);
  size_t size() const { return size_; }
};

}  // namespace talawa::rl::memory
