#pragma once
#include <sstream>
#include "result_base.hpp"

namespace shogi::engine::result {
class Id : public ResultBase {
 private:
  static constexpr std::string_view name = "wut shogi";
  static constexpr std::string_view author = "us";

 public:
  std::string toString() const override {
    std::ostringstream out;
    out << "id name " << name << "\n id author " << author << "\n";
    return out.str();
  }
};
}  // namespace shogi::engine::result