#pragma once

#include <iostream>
#include <memory>
#include <string>

namespace shogi::engine::result {

class ResultBase;

using ResultPtr = std::unique_ptr<ResultBase>;

class ResultBase {
 public:
  virtual std::string toString() const = 0;
  
  virtual ~ResultBase() = default;
  ResultBase() = default;
  ResultBase(ResultBase&) = default;
  ResultBase(ResultBase&&) = default;
  ResultBase& operator=(const ResultBase&) = default;
  ResultBase& operator=(ResultBase&&) = default;

  friend std::ostream& operator<<(std::ostream& out, const ResultBase& res) {
    return out << res.toString();
  }
};

}  // namespace shogi::engine::result