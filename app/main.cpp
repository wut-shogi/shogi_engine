#include <iostream>
#include <shogi/engine.hpp>

int main(int argc, char* argv[]) {
  std::cout << "Hello world!\n";
  shogi::engine::interface interface {
    shogi::engine::instance {}
  };
  interface.accept_input("test");
  std::cout << *(interface.await_result()) << std::endl;
  return 0;
}
