#pragma once

#include <vector>

#include "talawa/env/types.hpp"
namespace talawa::rl::memory {

class ReplayBuffer {
 private:
  std::vector<env::Transition> buffer_;
  int max_size_;
  int cursor_ = 0;

 public:
  ReplayBuffer(int max_size);
  ~ReplayBuffer() = default;
  std::vector<env::Transition> sample(size_t batch_size = 1) const;
  void add(const env::Transition& transition);
  size_t size() const { return buffer_.size(); }
};

}  // namespace talawa::rl::memory
