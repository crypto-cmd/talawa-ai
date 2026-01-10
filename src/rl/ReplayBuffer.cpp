#include "talawa/rl/ReplayBuffer.hpp"
namespace talawa::rl::memory {
ReplayBuffer::ReplayBuffer(int max_size) : max_size_(max_size) {}
void ReplayBuffer::add(const env::Transition& transition) {
  if (buffer_.size() < static_cast<size_t>(max_size_)) {
    buffer_.push_back(transition);
  } else {
    buffer_[cursor_] = transition;
  }
  cursor_ = (cursor_ + 1) % max_size_;
}
std::vector<env::Transition> ReplayBuffer::sample(size_t batch_size) const {
  if (batch_size > buffer_.size()) {
    throw std::runtime_error("Requested batch size larger than buffer size.");
  }
  std::vector<env::Transition> batch;
  batch.reserve(batch_size);
  int current = 0;
  while (current < batch_size) {
    auto index = rand() % buffer_.size();
    batch.push_back(buffer_[index]);
    current++;
  }
  return batch;
}
}
