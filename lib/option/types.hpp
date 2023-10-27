#pragma once

#include <string>

namespace shogi {
namespace engine {
namespace option {
template <typename TValue, TValue DEFAULT>
class base_option {
 private:
  TValue _value = DEFAULT;

 protected:
    static constexpr name std::string_view;

 public:
  void set(TValue value) { this->_value = value; };

  const TValue& get() const {return this->_value};
};

/// @brief A checkbox that can either be true or false. (USI 5.3)
/// @tparam DEFAULT default value
template <bool DEFAULT>
class check : public base_option<bool, DEFAULT> {};

/// @brief A spin wheel or slider that can be an integer in a certain range.
/// (USI 5.3)
/// @tparam DEFAULT default value
/// @tparam MINIMUM minimum value
/// @tparam MAXIMUM maximum value
template <int DEFAULT, int MINIMUM, int MAXIMUM>
class spin : public base_option<int, DEFAULT> {};
}  // namespace option
}  // namespace engine
}  // namespace shogi