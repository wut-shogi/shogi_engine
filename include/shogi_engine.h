#pragma once
#include <windows.h>

#define SHOGILIBRARY_API __declspec(dllexport)

extern "C" SHOGILIBRARY_API bool init();

extern "C" SHOGILIBRARY_API void cleanup();

extern "C" SHOGILIBRARY_API BSTR
getAllLegalMoves(const char* SFENstring);

extern "C" SHOGILIBRARY_API BSTR getBestMove(
    const char* SFENstring,
    unsigned int maxDepth,
    unsigned int maxTime);