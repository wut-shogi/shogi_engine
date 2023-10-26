#pragma once

#include <string>
#include "instance.hpp"

namespace shogi {
namespace engine {

/// @brief Interface to engine instance. Used to communicate with engine
/// according to Universal Shogi Interface.
class interface {
 private:
  instance _instance;

 public:
  /// @brief Processes commands according to Universal Shogi Interface.
  /// @param command Command string. It should be a single ASCII line,
  /// f.e. passed from stdin.
  void process_command(const std::string& command);
};
}  // namespace engine
}  // namespace shogi