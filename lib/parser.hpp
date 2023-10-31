#pragma once

#include <optional>
#include <string>
#include "command/command_base.hpp"

namespace shogi::engine {
class Parser {
 public:
  std::optional<command::CommandPtr> parse(const std::string& input);
};
}  // namespace shogi::engine