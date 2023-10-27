#pragma once

#include "result_base.hpp"
#include <vector>

namespace shogi {
namespace engine {
namespace result {

class result_list : public result_base {

 std::vector<ResultPtr> _results;

 public:

  result_list(std::initializer_list<ResultPtr> results) : _results {results} {};

  virtual std::string to_string() const override {
  }
};

}  // namespace result
}  // namespace engine
}  // namespace shogi