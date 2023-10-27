#pragma once

#include "result_base.hpp"

namespace shogi {
namespace engine {
namespace result {

class readyok : public result_base {
  virtual std::string to_string() const override { return "readyok\n"; }
};
}  // namespace result
}  // namespace engine
}  // namespace shogi