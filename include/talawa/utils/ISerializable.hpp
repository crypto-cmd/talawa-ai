#pragma once

#include <string>

namespace talawa::utils {
class ISerializable {
 public:
  // --- Serialization Methods ---
  virtual void save(const std::string& filename) const = 0;
  virtual void load(const std::string& filename) = 0;
};
}  // namespace talawa::utils
