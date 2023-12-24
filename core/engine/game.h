#pragma once
#include <chrono>
#include <vector>
#include "lookUpTables.h"
#include "moveGenHelpers.h"
#include "search.h"

namespace shogi {
namespace engine {

class GameSimulator {
 public:
  GameSimulator(
      std::vector<Move (*)(const Board&, bool, uint16_t)> getBestMoveFuncs)
      : getBestMoveFuncs(getBestMoveFuncs) {
  }
  void Run() {
    bool result = SEARCH::init();
    if (!result) {
      std::cout << "init Error" << std::endl;
    }
    board = Boards::STARTING_BOARD();
    isWhite = false;
    print_Board(board);
    std::string command;
    while (std::cin >> command) {
      if (command == "q") {
        break;
      } else if (command == "n") {
        Move bestMove;
        for (int i = 0; i < getBestMoveFuncs.size(); i++) {
          auto start = std::chrono::high_resolution_clock::now();
          bestMove = getBestMoveFuncs[i](board, isWhite, 5);
          auto stop = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
              stop - start);
          std::cout << "Time: " << duration.count() << " ms" << std::endl;
          printf("(%d) best Move: %s\n", i, moveToUSI(bestMove));
        }
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
  std::vector<Move (*)(const Board&, bool, uint16_t)> getBestMoveFuncs;
  std::vector<Move> bestMoves;
};

}  // namespace engine
}  // namespace shogi
