#include <vector>
#include "USIconverter.h"

namespace shogi {
namespace engine {
Board SFENToBoard(const std::string& boardSFEN, bool& isWhite) {
  Board result;
  std::string boardString, player, captures;
  std::istringstream iss(boardSFEN);
  iss >> boardString >> player >> captures;
  int rank = 0, file = 0;
  for (char c : boardString) {
    if (c == '/') {
      rank++;
      file = 0;
    } else if (isdigit(c)) {
      file += (c - '0');
    } else if (c == '+') {
      setSquare(result.bbs[BB::Type::PROMOTED], rankFileToSquare(rank, file));
    } else {
      if (c >= 'a' && c <= 'z') {
        setSquare(result.bbs[BB::Type::ALL_WHITE],
                  rankFileToSquare(rank, file));
      } else if (c >= 'A' && c <= 'Z') {
        setSquare(result.bbs[BB::Type::ALL_BLACK],
                  rankFileToSquare(rank, file));
        c += 32;
      }
      switch (c) {
        case 'p':
          setSquare(result.bbs[BB::Type::PAWN], rankFileToSquare(rank, file));
          break;
        case 'l':
          setSquare(result.bbs[BB::Type::LANCE], rankFileToSquare(rank, file));
          break;
        case 'n':
          setSquare(result.bbs[BB::Type::KNIGHT], rankFileToSquare(rank, file));
          break;
        case 's':
          setSquare(result.bbs[BB::Type::SILVER_GENERAL],
                    rankFileToSquare(rank, file));
          break;
        case 'g':
          setSquare(result.bbs[BB::Type::GOLD_GENERAL],
                    rankFileToSquare(rank, file));
          break;
        case 'b':
          setSquare(result.bbs[BB::Type::BISHOP], rankFileToSquare(rank, file));
          break;
        case 'r':
          setSquare(result.bbs[BB::Type::ROOK], rankFileToSquare(rank, file));
          break;
        case 'k':
          setSquare(result.bbs[BB::Type::KING], rankFileToSquare(rank, file));
          break;
        default:
          break;
      }
      file++;
    }
  }

  int count = 1;
  for (char c : captures) {
    if (isdigit(c)) {
      count = c - '0';
    } else {
      switch (c) {
        case 'p':
          result.inHand.pieceNumber.WhitePawn = count;
          break;
        case 'l':
          result.inHand.pieceNumber.WhiteLance = count;
          break;
        case 'n':
          result.inHand.pieceNumber.WhiteKnight = count;
          break;
        case 's':
          result.inHand.pieceNumber.WhiteSilverGeneral = count;
          break;
        case 'g':
          result.inHand.pieceNumber.WhiteGoldGeneral = count;
          break;
        case 'b':
          result.inHand.pieceNumber.WhiteBishop = count;
          break;
        case 'r':
          result.inHand.pieceNumber.WhiteRook = count;
          break;
        case 'P':
          result.inHand.pieceNumber.BlackPawn = count;
          break;
        case 'L':
          result.inHand.pieceNumber.BlackLance = count;
          break;
        case 'N':
          result.inHand.pieceNumber.BlackKnight = count;
          break;
        case 'S':
          result.inHand.pieceNumber.BlackSilverGeneral = count;
          break;
        case 'G':
          result.inHand.pieceNumber.BlackGoldGeneral = count;
          break;
        case 'B':
          result.inHand.pieceNumber.BlackBishop = count;
          break;
        case 'R':
          result.inHand.pieceNumber.BlackRook = count;
          break;
        default:
          break;
      }
      count = 1;
    }
  }

  isWhite = player[0] == 'w' ? true : false;

  return result;
}

std::string BoardToSFEN(const Board& board, bool isWhite) {
  std::vector<std::string> boardRepresentation = boardToStringVector(board);
  std::string result = "";
  int number = 0;
  for (int i = 0; i < BOARD_SIZE; i++) {
    if (boardRepresentation[i].empty()) {
      number++;
    } else {
      if (number > 0) {
        result += std::to_string(number);
        number = 0;
      }
      result += boardRepresentation[i];
    }
    if ((i + 1) % BOARD_DIM == 0) {
      if (number > 0) {
        result += std::to_string(number);
        number = 0;
      }
      if (i != BOARD_SIZE - 1)
        result += "/";
    }
  }

  result += " ";
  result += isWhite ? "w " : "b ";
  std::string inHandString = inHandToString(board.inHand);
  result += inHandString.empty() ? "-" : inHandString;
  return result;
}

Move USIToMove(const std::string& USImove) {
  if (USImove.size() < 4 || USImove.size() > 5)
    return {0, 0, 0};
  Move move;
  uint32_t row = 0, col = 0, square;
  if (std::isdigit(USImove[0])) {
    col = 9 - (USImove[0] - '0');
    row = USImove[1] - 'a';
    square = row * BOARD_DIM + col;
    if (square >= Square::SQUARE_SIZE) {
      return {0, 0, 0};
    }
    move.from = static_cast<Square>(square);
  } else {
    if (USImove[1] != '*')
      return {0, 0, 0};
    switch (USImove[0]) {
      case 'p':
        move.from = WHITE_PAWN_DROP;
        break;
      case 'l':
        move.from = WHITE_LANCE_DROP;
        break;
      case 'n':
        move.from = WHITE_KNIGHT_DROP;
        break;
      case 's':
        move.from = WHITE_SILVER_GENERAL_DROP;
        break;
      case 'g':
        move.from = WHITE_GOLD_GENERAL_DROP;
        break;
      case 'b':
        move.from = WHITE_BISHOP_DROP;
        break;
      case 'r':
        move.from = WHITE_ROOK_DROP;
        break;
      case 'P':
        move.from = BLACK_PAWN_DROP;
        break;
      case 'L':
        move.from = BLACK_LANCE_DROP;
        break;
      case 'N':
        move.from = BLACK_KNIGHT_DROP;
        break;
      case 'S':
        move.from = BLACK_SILVER_GENERAL_DROP;
        break;
      case 'G':
        move.from = BLACK_GOLD_GENERAL_DROP;
        break;
      case 'B':
        move.from = BLACK_BISHOP_DROP;
        break;
      case 'R':
        move.from = BLACK_ROOK_DROP;
        break;
      default:
        return {0, 0, 0};
    }
  }
  if (std::isdigit(USImove[2])) {
    col = 9 - (USImove[2] - '0');
    row = USImove[3] - 'a';
    square = row * BOARD_DIM + col;
    if (square >= Square::SQUARE_SIZE) {
      return {0, 0, 0};
    }
    move.to = static_cast<Square>(square);
  }
  move.promotion = 0;
  if (USImove.size() == 5 && USImove[4] == '+') {
    move.promotion = 1;
  }
  return move;
}
std::string MoveToUSI(Move move) {
  static const std::string pieceSymbols[14] = {
      "p", "l", "n", "s", "g", "b", "r", "P", "L", "N", "S", "G", "B", "R"};
  std::string moveString = "";
  if (move.from >= shogi::engine::WHITE_PAWN_DROP) {
    moveString +=
        pieceSymbols[move.from - shogi::engine::WHITE_PAWN_DROP] + "*";
  } else {
    int fromFile = shogi::engine::squareToFile(
        static_cast<shogi::engine::Square>(move.from));
    int fromRank = shogi::engine::squareToRank(
        static_cast<shogi::engine::Square>(move.from));
    moveString += std::to_string(BOARD_DIM - fromFile) +
                  static_cast<char>('a' + fromRank);
  }
  int toFile =
      shogi::engine::squareToFile(static_cast<shogi::engine::Square>(move.to));
  int toRank =
      shogi::engine::squareToRank(static_cast<shogi::engine::Square>(move.to));
  moveString +=
      std::to_string(BOARD_DIM - toFile) + static_cast<char>('a' + toRank);
  if (move.promotion) {
    moveString += '+';
  }
  return moveString;
}

std::vector<std::string> boardToStringVector(const Board& board) {
  std::vector<std::string> boardRepresentation(BOARD_SIZE);
  Bitboard promoted = board[BB::Type::PROMOTED];
  Bitboard notPromoted = ~promoted;
  Bitboard playerMask;
  BitboardIterator iterator;

  // White
  playerMask = board[BB::Type::ALL_WHITE];
  // Pawns
  iterator.Init(board[BB::Type::PAWN] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "p";
  }
  iterator.Init(board[BB::Type::PAWN] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+p";
  }
  // Lances
  iterator.Init(board[BB::Type::LANCE] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "l";
  }
  iterator.Init(board[BB::Type::LANCE] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+l";
  }
  // Knight
  iterator.Init(board[BB::Type::KNIGHT] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "n";
  }
  iterator.Init(board[BB::Type::KNIGHT] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+n";
  }
  // Silver Generals
  iterator.Init(board[BB::Type::SILVER_GENERAL] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "s";
  }
  iterator.Init(board[BB::Type::SILVER_GENERAL] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+s";
  }
  // Gold Generals
  iterator.Init(board[BB::Type::GOLD_GENERAL] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "g";
  }
  // Bishops
  iterator.Init(board[BB::Type::BISHOP] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "b";
  }
  iterator.Init(board[BB::Type::BISHOP] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+b";
  }
  // Rooks
  iterator.Init(board[BB::Type::ROOK] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "r";
  }
  iterator.Init(board[BB::Type::ROOK] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+r";
  }
  // Kings
  iterator.Init(board[BB::Type::KING] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "k";
  }

  // Black
  playerMask = board[BB::Type::ALL_BLACK];
  // Pawns
  iterator.Init(board[BB::Type::PAWN] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "P";
  }
  iterator.Init(board[BB::Type::PAWN] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+P";
  }
  // Lances
  iterator.Init(board[BB::Type::LANCE] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "L";
  }
  iterator.Init(board[BB::Type::LANCE] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+L";
  }
  // Knight
  iterator.Init(board[BB::Type::KNIGHT] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "N";
  }
  iterator.Init(board[BB::Type::KNIGHT] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+N";
  }
  // Silver Generals
  iterator.Init(board[BB::Type::SILVER_GENERAL] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "S";
  }
  iterator.Init(board[BB::Type::SILVER_GENERAL] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+S";
  }
  // Gold Generals
  iterator.Init(board[BB::Type::GOLD_GENERAL] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "G";
  }
  // Bishops
  iterator.Init(board[BB::Type::BISHOP] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "B";
  }
  iterator.Init(board[BB::Type::BISHOP] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+B";
  }
  // Rooks
  iterator.Init(board[BB::Type::ROOK] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "R";
  }
  iterator.Init(board[BB::Type::ROOK] & playerMask & promoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "+R";
  }
  // Kings
  iterator.Init(board[BB::Type::KING] & playerMask & notPromoted);
  while (iterator.Next()) {
    boardRepresentation[iterator.GetCurrentSquare()] = "K";
  }

  return boardRepresentation;
}

std::string inHandToString(const InHandLayout& inHand) {
  std::string result;
  if (inHand.pieceNumber.BlackRook) {
    if (inHand.pieceNumber.BlackRook > 1)
      result += std::to_string(inHand.pieceNumber.BlackRook);
    result += "R";
  }
  if (inHand.pieceNumber.BlackBishop) {
    if (inHand.pieceNumber.BlackBishop > 1)
      result += std::to_string(inHand.pieceNumber.BlackBishop);
    result += "B";
  }
  if (inHand.pieceNumber.BlackGoldGeneral) {
    if (inHand.pieceNumber.BlackGoldGeneral > 1)
      result += std::to_string(inHand.pieceNumber.BlackGoldGeneral);
    result += "G";
  }
  if (inHand.pieceNumber.BlackSilverGeneral) {
    if (inHand.pieceNumber.BlackSilverGeneral > 1)
      result += std::to_string(inHand.pieceNumber.BlackSilverGeneral);
    result += "S";
  }
  if (inHand.pieceNumber.BlackKnight) {
    if (inHand.pieceNumber.BlackKnight > 1)
      result += std::to_string(inHand.pieceNumber.BlackKnight);
    result += "N";
  }
  if (inHand.pieceNumber.BlackLance) {
    if (inHand.pieceNumber.BlackLance > 1)
      result += std::to_string(inHand.pieceNumber.BlackLance);
    result += "L";
  }
  if (inHand.pieceNumber.BlackPawn) {
    if (inHand.pieceNumber.BlackPawn > 1)
      result += std::to_string(inHand.pieceNumber.BlackPawn);
    result += "P";
  }
  if (inHand.pieceNumber.WhiteRook) {
    if (inHand.pieceNumber.WhiteRook > 1)
      result += std::to_string(inHand.pieceNumber.WhiteRook);
    result += "r";
  }
  if (inHand.pieceNumber.WhiteBishop) {
    if (inHand.pieceNumber.WhiteBishop > 1)
      result += std::to_string(inHand.pieceNumber.WhiteBishop);
    result += "b";
  }
  if (inHand.pieceNumber.WhiteGoldGeneral) {
    if (inHand.pieceNumber.WhiteGoldGeneral > 1)
      result += std::to_string(inHand.pieceNumber.WhiteGoldGeneral);
    result += "g";
  }
  if (inHand.pieceNumber.WhiteSilverGeneral) {
    if (inHand.pieceNumber.WhiteSilverGeneral > 1)
      result += std::to_string(inHand.pieceNumber.WhiteSilverGeneral);
    result += "s";
  }
  if (inHand.pieceNumber.WhiteKnight) {
    if (inHand.pieceNumber.WhiteKnight > 1)
      result += std::to_string(inHand.pieceNumber.WhiteKnight);
    result += "n";
  }
  if (inHand.pieceNumber.WhiteLance) {
    if (inHand.pieceNumber.WhiteLance > 1)
      result += std::to_string(inHand.pieceNumber.WhiteLance);
    result += "l";
  }
  if (inHand.pieceNumber.WhitePawn) {
    if (inHand.pieceNumber.WhitePawn > 1)
      result += std::to_string(inHand.pieceNumber.WhitePawn);
    result += "p";
  }

  return result;
}
}  // namespace engine
}  // namespace shogi