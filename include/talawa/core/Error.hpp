#pragma once
#define THROW_talawa_ERROR(type, msg)                      \
  do {                                                     \
    std::stringstream ss;                                  \
    ss << "\n[Talawa AI (" << #type << ") Error]: " << msg \
       << "\n[Location]: " << __FILE__ << ":" << __LINE__; \
    throw std::invalid_argument(ss.str());                 \
  } while (0)
