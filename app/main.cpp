#include <iostream>
#include <memory>
#include <shogi/engine.hpp>

int main(int argc, char* argv[]) {
  std::cout << "Hello world!\n";
  shogi::engine::interface interface {
    shogi::engine::instance {}
  };

  std::cout << *(interface.await_result()) << std::endl;
  return 0;
}
