#pragma once
#include <vector>
#include <chrono>
#include "GameTree.h"
#include "MoveGenHelpers.h"
#include "gpuInterface.h"
#include "LookUpTables.h"

namespace shogi {
namespace engine {

class GameSimulator {
 public:
  void Run() {
    LookUpTables::CPU::init();
    LookUpTables::GPU::init();
    d_BufferSize = 1900000000;
    cudaError_t error = cudaMalloc((void**)&d_Buffer, d_BufferSize);
    board = Boards::STARTING_BOARD();
    isWhite = false;
    print_Board(board);
    std::string command;
    while (std::cin >> command) {
      if (command == "q") {
        break;
      } else if (command == "n") {
        auto start = std::chrono::high_resolution_clock::now();
        Move bestMove =
            GetBestMove(d_Buffer, d_BufferSize, board, isWhite, 0, 5);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time: " << duration.count() << " ms" << std::endl;
        makeMove(board, bestMove);
        print_Board(board);
        isWhite = !isWhite;
      }
    }
  }

 private:
  Board board;
  bool isWhite;
  uint8_t* d_Buffer;
  uint32_t d_BufferSize;
};

}  // namespace engine
}  // namespace shogi
