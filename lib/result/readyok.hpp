#pragma once

#include "result_base.hpp"

namespace shogi::engine::result {

class ReadyOk : public ResultBase {
 public:
  std::string toString() const override { return "readyok\n"; }
};
}  // namespace shogi::engine::result