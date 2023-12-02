#include <cstdint>
enum Square : int32_t {
	A9, A8, A7, A6, A5, A4, A3, A2, A1, //
	B9, B8, B7, B6, B5, B4, B3, B2, B1, //
	C9, C8, C7, C6, C5, C4, C3, C2, C1, //
	D9, D8, D7, D6, D5, D4, D3, D2, D1, //
	E9, E8, E7, E6, E5, E4, E3, E2, E1, //
	F9, F8, F7, F6, F5, F4, F3, F2, F1, //
	G9, G8, G7, G6, G5, G4, G3, G2, G1, //
	H9, H8, H7, H6, H5, H4, H3, H2, H1, //
	I9, I8, I7, I6, I5, I4, I3, I2, I1, //
	SQUARE_SIZE,
	NONE,
	UP = -9,
	UP_RIGHT = -8,
	RIGHT = 1,
	DOWN_RIGHT = 10,
	DOWN = 9,
	DOWN_LEFT = 8,
	LEFT = -1,
	UP_LEFT = -10,

};

inline int squareToRank(Square square) {
  return square / BOARD_DIM;
}

inline int squareToFile(Square square) {
  return square % BOARD_DIM;
}