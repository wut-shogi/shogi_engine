#pragma once

#include "result_base.hpp"

namespace shogi::engine::result {

class UsiOk : public ResultBase {
  std::string toString() const override { return "usiok\n"; }
};
}  // namespace shogi::engine::result