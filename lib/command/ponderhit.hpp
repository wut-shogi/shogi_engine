#pragma once

#include "command_base.hpp"
#include "result_base.hpp"

namespace shogi {
namespace engine {
namespace command {

/// @brief The user has played the expected move. This will be sent if the
/// engine was told to ponder on the same move the user has played. The engine
/// should continue searching but switch from pondering to normal search.
/// (USI 5.3)
class ponderhit : public command_base {
 public:
  virtual void execute(invoker& invoker) override{
      // invoker.post_result(std::move(result));
  };
};
}  // namespace command
}  // namespace engine
}  // namespace shogi