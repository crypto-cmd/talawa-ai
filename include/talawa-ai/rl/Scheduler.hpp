#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace talawa_ai::rl::scheduler {

struct ScheduleContext {
  int episode;
  int step;
  int total_steps;
  float episode_reward;
  float last_reward;
};
using Condition = std::function<bool(const ScheduleContext&)>;

constexpr int UNTIL_END = -1;
namespace conditions {
// Step every N episodes
inline Condition every_n_episodes(int n) {
  return [n](const ScheduleContext& ctx) { return ctx.episode % n == 0; };
}

// Step every N total steps
inline Condition every_n_steps(int n) {
  return [n](const ScheduleContext& ctx) { return ctx.total_steps % n == 0; };
}

// Always step
inline Condition always() {
  return [](const ScheduleContext&) { return true; };
}

// Step when reward drops by more than percent%
inline Condition on_reward_decrease(float percent = 0.0f) {
  return [percent](const ScheduleContext& ctx) {
    float decrease = ctx.last_reward - ctx.episode_reward;
    float threshold = std::abs(ctx.last_reward) * percent;
    return decrease > threshold;
  };
}

// Step after N episodes (for warmup ending)
inline Condition after_episode(int n) {
  return [n](const ScheduleContext& ctx) { return ctx.episode >= n; };
}

// Combine conditions with AND
inline Condition all_of(Condition a, Condition b) {
  return [a = std::move(a), b = std::move(b)](const ScheduleContext& ctx) {
    return a(ctx) && b(ctx);
  };
}
}  // namespace conditions

enum class ScheduleEvent {
  OnStep,
  OnEpisodeEnd,
};

class Scheduler {
 public:
  virtual ~Scheduler() = default;
  virtual float value() const = 0;
  virtual void step() = 0;
  virtual void reset() = 0;
  virtual std::unique_ptr<Scheduler> clone() const = 0;
};

class ConstantScheduler : public Scheduler {
  float value_;

 public:
  explicit ConstantScheduler(float value) : value_(value) {}
  float value() const override { return value_; }
  void step() override {}
  void reset() override {}
  std::unique_ptr<Scheduler> clone() const override {
    return std::make_unique<ConstantScheduler>(*this);
  }
};
// current = max(min, current - decay_amount)
class LinearDecay : public Scheduler {
  float initial_, current_, min_, decay_amount_;

 public:
  LinearDecay(float start, float min, float decay_amount)
      : initial_(start),
        current_(start),
        min_(min),
        decay_amount_(decay_amount) {}

  float value() const override { return current_; }
  void step() override { current_ = std::max(min_, current_ - decay_amount_); }
  void reset() override { current_ = initial_; }
  std::unique_ptr<Scheduler> clone() const override {
    return std::make_unique<LinearDecay>(*this);
  }
};

// current = max(min, current * decay_factor)
class ExponentialDecay : public Scheduler {
  float initial_, current_, min_, decay_factor_;

 public:
  ExponentialDecay(float start, float min, float decay_factor)
      : initial_(start),
        current_(start),
        min_(min),
        decay_factor_(decay_factor) {}

  float value() const override { return current_; }
  void step() override { current_ = std::max(min_, current_ * decay_factor_); }
  void reset() override { current_ = initial_; }
  std::unique_ptr<Scheduler> clone() const override {
    return std::make_unique<ExponentialDecay>(*this);
  }
};

// Step function: value changes at specific points
// current = current * decay_factor every step_size_ steps
class StepDecay : public Scheduler {
  float initial_, current_, decay_factor_;
  int step_count_ = 0;
  int step_size_;  // decay every N steps
 public:
  StepDecay(float start, float decay_factor, int step_size)
      : initial_(start),
        current_(start),
        decay_factor_(decay_factor),
        step_size_(step_size) {}

  float value() const override { return current_; }
  void step() override {
    ++step_count_;
    if (step_count_ % step_size_ == 0) {
      current_ *= decay_factor_;
    }
  }
  void reset() override {
    current_ = initial_;
    step_count_ = 0;
  }
  std::unique_ptr<Scheduler> clone() const override {
    return std::make_unique<StepDecay>(*this);
  }
};

class ChainedScheduler : public Scheduler {
 public:
  struct Phase {
    std::unique_ptr<Scheduler> scheduler;
    int duration;  // -1 for "until end"
  };

 private:
  std::vector<Phase> phases_;
  size_t current_phase_ = 0;
  int steps_in_phase_ = 0;

 public:
  ChainedScheduler& add(std::unique_ptr<Scheduler> scheduler,
                        int duration = -1) {
    phases_.push_back({std::move(scheduler), duration});
    return *this;
  }

  float value() const override {
    if (current_phase_ >= phases_.size()) {
      return phases_.back().scheduler->value();
    }
    return phases_[current_phase_].scheduler->value();
  }

  void step() override {
    if (current_phase_ >= phases_.size()) return;

    phases_[current_phase_].scheduler->step();
    ++steps_in_phase_;

    // Check for phase transition
    int duration = phases_[current_phase_].duration;
    if (duration > 0 && steps_in_phase_ >= duration) {
      ++current_phase_;
      steps_in_phase_ = 0;
    }
  }

  void reset() override {
    current_phase_ = 0;
    steps_in_phase_ = 0;
    for (auto& phase : phases_) {
      phase.scheduler->reset();
    }
  }

  std::unique_ptr<Scheduler> clone() const override {
    auto copy = std::make_unique<ChainedScheduler>();
    for (const auto& phase : phases_) {
      copy->phases_.push_back({phase.scheduler->clone(), phase.duration});
    }
    return copy;
  }
};

class ChainedSchedulerBuilder {
  std::unique_ptr<ChainedScheduler> scheduler_;

 public:
  ChainedSchedulerBuilder()
      : scheduler_(std::make_unique<ChainedScheduler>()) {}

  ChainedSchedulerBuilder& add(std::unique_ptr<Scheduler> scheduler,
                               int duration = UNTIL_END) {
    scheduler_->add(std::move(scheduler), duration);
    return *this;
  }

  std::unique_ptr<Scheduler> build() { return std::move(scheduler_); }
};

inline ChainedSchedulerBuilder chain() { return ChainedSchedulerBuilder(); }

class SchedulerBinding {
 public:
  using Setter = std::function<void(float)>;

  SchedulerBinding(std::string name, std::unique_ptr<Scheduler> scheduler,
                   Setter setter, ScheduleEvent event,
                   Condition condition = conditions::always())
      : name_(std::move(name)),
        scheduler_(std::move(scheduler)),
        setter_(std::move(setter)),
        event_(event),
        condition_(std::move(condition)) {}

  // Move-only
  SchedulerBinding(SchedulerBinding&&) = default;
  SchedulerBinding& operator=(SchedulerBinding&&) = default;

  void apply() { setter_(scheduler_->value()); }

  void maybe_step(const ScheduleContext& ctx) {
    if (condition_(ctx)) {
      scheduler_->step();
      apply();
    }
  }

  void reset() {
    scheduler_->reset();
    apply();
  }

  const std::string& name() const { return name_; }
  ScheduleEvent event() const { return event_; }
  float value() const { return scheduler_->value(); }

 private:
  std::string name_;
  std::unique_ptr<Scheduler> scheduler_;
  Setter setter_;
  ScheduleEvent event_;
  Condition condition_;
};

class SchedulerSet {
 public:
  using LogCallback = std::function<void(const std::string& name, float value,
                                         const ScheduleContext& ctx)>;

  void add(SchedulerBinding binding) {
    bindings_.push_back(std::move(binding));
  }

  void set_log_callback(LogCallback cb) { log_callback_ = std::move(cb); }

  void on_step(const ScheduleContext& ctx) {
    for (auto& b : bindings_) {
      if (b.event() == ScheduleEvent::OnStep) {
        b.maybe_step(ctx);
        if (log_callback_) log_callback_(b.name(), b.value(), ctx);
      }
    }
  }

  void on_episode_end(const ScheduleContext& ctx) {
    for (auto& b : bindings_) {
      if (b.event() == ScheduleEvent::OnEpisodeEnd) {
        b.maybe_step(ctx);
        if (log_callback_) log_callback_(b.name(), b.value(), ctx);
      }
    }
  }

  void initialize() {
    for (auto& b : bindings_) {
      b.apply();
    }
  }

  void reset() {
    for (auto& b : bindings_) {
      b.reset();
    }
  }

  // Query current values (for logging/debugging)
  std::vector<std::pair<std::string, float>> values() const {
    std::vector<std::pair<std::string, float>> result;
    for (const auto& b : bindings_) {
      result.emplace_back(b.name(), b.value());
    }
    return result;
  }

  bool empty() const { return bindings_.empty(); }

 private:
  std::vector<SchedulerBinding> bindings_;
  LogCallback log_callback_ = nullptr;
};

class SchedulerBuilder {
  std::string name_;
  std::unique_ptr<Scheduler> scheduler_;
  SchedulerBinding::Setter setter_;
  ScheduleEvent event_ = ScheduleEvent::OnEpisodeEnd;
  Condition condition_ = conditions::always();

 public:
  explicit SchedulerBuilder(std::string name) : name_(std::move(name)) {}

  SchedulerBuilder& use(std::unique_ptr<Scheduler> s) {
    scheduler_ = std::move(s);
    return *this;
  }

  template <typename S, typename... Args>
  SchedulerBuilder& use(Args&&... args) {
    scheduler_ = std::make_unique<S>(std::forward<Args>(args)...);
    return *this;
  }

  SchedulerBuilder& bind_to(SchedulerBinding::Setter setter) {
    setter_ = std::move(setter);
    return *this;
  }

  SchedulerBuilder& on(ScheduleEvent event) {
    event_ = event;
    return *this;
  }

  SchedulerBuilder& when(Condition cond) {
    condition_ = std::move(cond);
    return *this;
  }

  SchedulerBinding build() {
    return SchedulerBinding(std::move(name_), std::move(scheduler_),
                            std::move(setter_), event_, std::move(condition_));
  }
};

inline SchedulerBuilder schedule(std::string name) {
  return SchedulerBuilder(std::move(name));
}

}  // namespace talawa_ai::rl::scheduler
