#pragma once

#include "command_base.hpp"
#include "result_base.hpp"
#include "readyok.hpp"

namespace shogi {
namespace engine {
namespace command {

class isready : public command_base {
 public:
  virtual void execute(invoker& invoker) override {
    result::ResultPtr result = std::make_unique<result::readyok>();
    //invoker.post_result(std::move(result));
  };
};
}  // namespace command
}  // namespace engine
}  // namespace shogi