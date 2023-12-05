#pragma once

#ifdef SHOGILIBRARY_EXPORTS
#define SHOGILIBRARY_API __declspec(dllexport)
#else
#define SHOGILIBRARY_API __declspec(dllimport)
#endif
#include <vector>
#include <string>


/// <summary>
/// Function returns all legal moves from given square
/// </summary>
/// <param name="SFENstring">Current board representation in SFEN format</param>
/// <param name="rank">Row</param>
/// <param name="file">Column</param>
/// <returns>List of all square which are legal moves. Move is represented as pair of two ints, first one is rank ande second once is file</returns>
extern "C" SHOGILIBRARY_API std::vector<std::pair<int, int>>
getLegalMoves(const std::string SFENstring, int rank, int file);