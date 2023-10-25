#include <iostream>
#include "MoveGen.h"
#include "Board.h"

namespace shogi {
namespace engine {

void test2() {
  std::cout << "Test!\n";
}

void test() {
  std::cout << "Test!\n";
  Board startingBoard = Boards::STARTING_BOARD();
  /*for (int i = 0; i < BitboardType::SIZE; i++) {
    std::cout << "Bitboard: " << i << std::endl;
    print_BB(startingBoard[(BitboardType)i]);
  }*/
  Bitboard occupiedAll = startingBoard[BitboardType::ALL_WHITE] |
                         startingBoard[BitboardType::ALL_BLACK];

  //std::cout << "Occupied all" << std::endl;
  //print_BB(occupiedAll);
  //std::cout << "Occupied all rotate 90 clock" << std::endl;
  //print_BB(Rotate90Clockwise(occupiedAll));
  //std::cout << "Occupied all rotate 90 anti clock" << std::endl;
  //print_BB(Rotate90AntiClockwise(occupiedAll));
  //std::cout << "Occupied all rotate 45 clock" << std::endl;
  //print_BB(Rotate45Clockwise(occupiedAll));
  //std::cout << "Occupied all rotate 45 anti clock" << std::endl;
  //print_BB(Rotate45AntiClockwise(occupiedAll));
  int position = 66;
  std::cout << "Rank attacks" << std::endl;
  auto moves = RankAttacks[getRankBlockPattern(
      startingBoard[BitboardType::ALL_WHITE] |
          startingBoard[BitboardType::ALL_BLACK],
      position)][position];
  print_BB(moves);
  std::cout << std::endl;
  std::cout << "File attacks" << std::endl;
  moves = FileAttacks[getRankBlockPattern(startingBoard[BitboardType::OCCUPIED_ROT90],
      position)][position];
  print_BB(moves);
  std::cout << std::endl;
  std::cout << "Diagonal right attacks" << std::endl;
   moves = DiagRightAttacks[getDiagRightBlockPattern(
      startingBoard[BitboardType::OCCUPIED_ROTR45], position)][position];
  print_BB(moves);
  std::cout << std::endl;
  std::cout << "Diagonal left attacks" << std::endl;
   moves = DiagLeftAttacks[getDiagLeftBlockPattern(
      startingBoard[BitboardType::OCCUPIED_ROTL45], position)][position];
  print_BB(moves);

  /*std::cout << "Test 90" << std::endl;
  std::cout << "Rot90Clock then Rot90anticlock" << std::endl;
  print_BB(Rotate90AntiClockwise(Rotate90Clockwise(occupiedAll)));
  std::cout << "Rot90antiClock then Rot90clock" << std::endl;
  print_BB(Rotate90Clockwise(Rotate90AntiClockwise(occupiedAll)));
  std::cout << "Test 45" << std::endl;
  std::cout << "Rot45Clock then Rot45anticlock" << std::endl;
  print_BB(Rotate45AntiClockwise(Rotate45Clockwise(occupiedAll)));
  std::cout << "Rot45antiClock then Rot45clock" << std::endl;
  print_BB(Rotate45Clockwise(Rotate45AntiClockwise(occupiedAll)));

  std::cout << "Rot90Clock" << std::endl;
  print_BB(Rotate90Clockwise(occupiedAll));
  std::cout << "Rot90AntiClock" << std::endl;
  print_BB(Rotate90AntiClockwise(occupiedAll));*/
}


}  // namespace engine
}  // namespace shogi
