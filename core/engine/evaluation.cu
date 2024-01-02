#include "evaluation.h"

namespace shogi {
namespace engine {
RUNTYPE int16_t evaluate(const Board& board, bool isWhite) {
  int16_t whitePoints = 0, blackPoints = 0;
  Bitboard pieces;
  // White
  // Pawns
  pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::PAWN;
  pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_PAWN;
  whitePoints += board.inHand.pieceNumber.WhitePawn * PieceValue::IN_HAND_LANCE;
  // Lances
  pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::LANCE;
  pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_LANCE;
  whitePoints +=
      board.inHand.pieceNumber.WhiteLance * PieceValue::IN_HAND_LANCE;
  // Knights
  pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::KNIGHT;
  pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_KNIGHT;
  whitePoints +=
      board.inHand.pieceNumber.WhiteKnight * PieceValue::IN_HAND_KNIGHT;
  // SilverGenerals
  pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::SILVER_GENERAL;
  pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_SILVER_GENERAL;
  whitePoints += board.inHand.pieceNumber.WhiteSilverGeneral *
                 PieceValue::IN_HAND_SILVER_GENERAL;
  // GoldGenerals
  pieces = board[BB::Type::GOLD_GENERAL] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::GOLD_GENERAL;
  whitePoints += board.inHand.pieceNumber.WhiteGoldGeneral *
                 PieceValue::IN_HAND_GOLD_GENERAL;
  // Bishops
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::BISHOP;
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_BISHOP;
  whitePoints +=
      board.inHand.pieceNumber.WhiteBishop * PieceValue::IN_HAND_BISHOP;
  // Rooks
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
           ~board[BB::Type::PROMOTED];
  whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::ROOK;
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_WHITE] &
           board[BB::Type::PROMOTED];
  whitePoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_ROOK;
  whitePoints += board.inHand.pieceNumber.WhiteRook * PieceValue::IN_HAND_ROOK;

  pieces = board[BB::Type::ALL_WHITE];
  whitePoints += pieces[MID] * 10 + pieces[BOTTOM] * 20;

  // Black
  // Pawns
  pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::PAWN;
  pieces = board[BB::Type::PAWN] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_PAWN;
  blackPoints += board.inHand.pieceNumber.BlackPawn * PieceValue::IN_HAND_LANCE;
  // Lances
  pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::LANCE;
  pieces = board[BB::Type::LANCE] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_LANCE;
  blackPoints +=
      board.inHand.pieceNumber.BlackLance * PieceValue::IN_HAND_LANCE;
  // Knights
  pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::KNIGHT;
  pieces = board[BB::Type::KNIGHT] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_KNIGHT;
  blackPoints +=
      board.inHand.pieceNumber.BlackKnight * PieceValue::IN_HAND_KNIGHT;
  // SilverGenerals
  pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::SILVER_GENERAL;
  pieces = board[BB::Type::SILVER_GENERAL] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_SILVER_GENERAL;
  blackPoints += board.inHand.pieceNumber.BlackSilverGeneral *
                 PieceValue::IN_HAND_SILVER_GENERAL;
  // GoldGenerals
  pieces = board[BB::Type::GOLD_GENERAL] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::GOLD_GENERAL;
  blackPoints += board.inHand.pieceNumber.BlackGoldGeneral *
                 PieceValue::IN_HAND_GOLD_GENERAL;
  // Bishops
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::BISHOP;
  pieces = board[BB::Type::BISHOP] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_BISHOP;
  blackPoints +=
      board.inHand.pieceNumber.BlackBishop * PieceValue::IN_HAND_BISHOP;
  // Rooks
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
           ~board[BB::Type::PROMOTED];
  blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::ROOK;
  pieces = board[BB::Type::ROOK] & board[BB::Type::ALL_BLACK] &
           board[BB::Type::PROMOTED];
  blackPoints += (popcount(pieces[TOP]) + popcount(pieces[MID]) +
                  popcount(pieces[BOTTOM])) *
                 PieceValue::PROMOTED_ROOK;
  blackPoints += board.inHand.pieceNumber.BlackRook * PieceValue::IN_HAND_ROOK;

  pieces = board[BB::Type::ALL_BLACK];
  whitePoints += pieces[MID] * 10 + pieces[TOP] * 20;

  int16_t score = whitePoints - blackPoints;
  return score;
}
}  // namespace engine
}  // namespace shogi