#pragma once

#include "command_base.hpp"
#include "result_base.hpp"

namespace shogi {
namespace engine {
namespace command {

/// @brief Stop calculating as soon as possible. Don't forget the bestmove and
/// possibly the ponder token when finishing the search. (USI 5.3)
class stop : public command_base {
 public:
  virtual void execute(instance& instance) override{
      // instance.post_result(std::move(result));
  };
};
}  // namespace command
}  // namespace engine
}  // namespace shogi