#include <iostream>
#include "GameTree.h"
#include "engine.h"
#include "MoveGen.h"
#include "cpuInterface.h"

namespace shogi {
namespace engine {

void test() {
  Board startingBoard = Boards::STARTING_BOARD();
  bool isWhite = false;
  GameTree tree(startingBoard, isWhite, 5);
  Move bestMove = tree.FindBestMove();
  std::cout << "Best found move (from, to, promotion): (" << bestMove.from
            << ", " << bestMove.to << ", " << bestMove.promotion << ")"
            << std::endl;

  /*std::vector<uint32_t> offsets(2);
  Move move;
  Bitboard valid, attacked, pinned;
  bool isMate = false;
  CPU::countBlackMoves((uint32_t)1, (int16_t)0, startingBoard, &move, offsets.data() + 1, &valid,
                       &attacked, &pinned, &isMate);
  CPU::prefixSum(offsets.data(), offsets.size());
  std::vector<Move> moves(offsets.back());
  CPU::generateBlackMoves(1, 0, startingBoard, &move, offsets.data(), &valid,
                          &attacked, &pinned, moves.data());*/


  // Trzymaæ ruchy w osobnych tablicach

  std::cout << "Done!" << std::endl;
}

}  // namespace engine
}  // namespace shogi
