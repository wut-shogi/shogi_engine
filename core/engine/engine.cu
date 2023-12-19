#include <iostream>
#include "GameTree.h"
#include "engine.h"
#include "MoveGen.h"
#include "cpuInterface.h"
#include "game.h"

#include "gpuInterface.h"
#include "MoveGenHelpers.h"
namespace shogi {
namespace engine {

void test() {
  Board startingBoard = Boards::STARTING_BOARD();
  print_Board(startingBoard);
  bool isWhite = true;
  Board board = Board::FromSFEN(
      "1R3G1nl/4g1kg1/1p2p+bpp1/p+B1Ls1s1p/Pn3+l3/KSPpP3P/3P+rpP2/2G6/L8 b "
      "NPsn3p 1",
      isWhite);
  /*Move move;
  move.from = 55;
  move.to = 46;
  move.promotion = true;
  makeMove(startingBoard, move);
  print_Board(startingBoard);
  move.from = 16;
  move.to = 46;
  move.promotion = true;
  makeMove(startingBoard, move);
  print_Board(startingBoard);*/
  /*GameTree tree(startingBoard, isWhite, 5);
  Move bestMove = tree.FindBestMove();
  std::cout << "Best found move (from, to, promotion): (" << bestMove.from
            << ", " << bestMove.to << ", " << bestMove.promotion << ")"
            << std::endl;*/

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
  /*GPU::initLookUpArrays();
  Move bestMove = GetBestMove(board, isWhite, 0, 6);
  std::cout << "From: " << bestMove.from << ", To: " << bestMove.to
            << ", Promotion: " << bestMove.promotion << std::endl;*/
  GameSimulator simulator;
  simulator.Run();
  std::cout << "Done!" << std::endl;
}

}  // namespace engine
}  // namespace shogi
