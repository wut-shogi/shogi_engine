#pragma once

#include <string>

namespace shogi::engine::option {
template <typename TValue, TValue DEFAULT>
class BaseOption {
 private:
  TValue _value = DEFAULT;

 public:
  void set(TValue value) { this->_value = value; };

  const TValue& get() const { return this->_value; };
};

/// @brief A checkbox that can either be true or false. (USI 5.3)
/// @tparam DEFAULT default value
template <bool DEFAULT>
class Check : public BaseOption<bool, DEFAULT> {};

/// @brief A spin wheel or slider that can be an integer in a certain range.
/// (USI 5.3)
/// @tparam DEFAULT default value
/// @tparam MINIMUM minimum value
/// @tparam MAXIMUM maximum value
template <int DEFAULT, int MINIMUM, int MAXIMUM>
class Spin : public BaseOption<int, DEFAULT> {};
}  // namespace shogi::engine::option