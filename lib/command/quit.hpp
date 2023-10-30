#pragma once

#include "command_base.hpp"
#include "result_base.hpp"

namespace shogi {
namespace engine {
namespace command {

/// @brief Quit the program as soon as possible. (USI 5.3)
class quit : public command_base {
 public:
  virtual void execute(instance& instance) override{
      // instance.post_result(std::move(result));
  };
};
}  // namespace command
}  // namespace engine
}  // namespace shogi