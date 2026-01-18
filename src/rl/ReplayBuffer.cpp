#include "talawa/rl/ReplayBuffer.hpp"
namespace talawa::rl::memory {
ReplayBuffer::ReplayBuffer(int max_size) : max_size_(max_size) {}
void ReplayBuffer::add(const env::Transition& transition) {
  if (size_ == 0) {
    // First addition, need to resize matrices based on transition sizes
    buffer_.states = core::Matrix::zeros(max_size_, transition.state.size());
    buffer_.next_states =
        core::Matrix::zeros(max_size_, transition.next_state.size());
    buffer_.actions = core::Matrix::zeros(max_size_, transition.action.size());
    buffer_.rewards = core::Matrix::zeros(max_size_, 1);
    buffer_.dones = core::Matrix::zeros(max_size_, 1);
  }
  // add to the batch matrix
  buffer_.states.setRow(cursor_, transition.state.flatten());
  buffer_.next_states.setRow(cursor_, transition.next_state.flatten());
  buffer_.actions.setRow(cursor_, transition.action.flatten());
  buffer_.rewards(cursor_, 0) = transition.reward;
  buffer_.dones(cursor_, 0) =
      (transition.status != env::EpisodeStatus::Running) ? 1.0f : 0.0f;
  if (size_ < max_size_) {
    size_++;
  }
  cursor_ = (cursor_ + 1) % max_size_;
}
Experience ReplayBuffer::sample(size_t batch_size) {
  if (batch_size > size_) {
    throw std::runtime_error("Requested batch size larger than buffer size.");
  }

  if (batch_size != expected_sample_size_) {
    // Resize sample matrices
    sample_.states = core::Matrix::zeros(batch_size, buffer_.states.cols);
    sample_.next_states =
        core::Matrix::zeros(batch_size, buffer_.next_states.cols);
    sample_.actions = core::Matrix::zeros(batch_size, buffer_.actions.cols);
    sample_.rewards = core::Matrix::zeros(batch_size, 1);
    sample_.dones = core::Matrix::zeros(batch_size, 1);
    expected_sample_size_ =
        static_cast<int>(batch_size);  // Update expected size
  }

  int current = 0;
  while (current < batch_size) {
    auto index = rand() % size_;
    // Copy data from buffer to batch
    sample_.states.setRow(current,
                          buffer_.states.slice(index, index + 1).flatten());
    sample_.next_states.setRow(
        current, buffer_.next_states.slice(index, index + 1).flatten());
    sample_.actions.setRow(current,
                           buffer_.actions.slice(index, index + 1).flatten());
    sample_.rewards(current, 0) = buffer_.rewards(index, 0);
    sample_.dones(current, 0) = buffer_.dones(index, 0);
    current++;
  }
  return sample_;
}
}  // namespace talawa::rl::memory
