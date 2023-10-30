#pragma once

#include "command_base.hpp"
#include "result_base.hpp"

namespace shogi {
namespace engine {
namespace command {

/// @brief (Shogidogoro) Informs the engine that the game has ended with the
/// specified result, from the engine's own point or view. (USI 5.3)
class gameover : public command_base {
 public:
  virtual void execute(instance& instance) override{
      // instance.post_result(std::move(result));
  };
};
}  // namespace command
}  // namespace engine
}  // namespace shogi