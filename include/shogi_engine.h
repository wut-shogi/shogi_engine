#pragma once
#include <windows.h>

#define SHOGILIBRARY_API __declspec(dllexport)

extern "C" SHOGILIBRARY_API bool init();

extern "C" SHOGILIBRARY_API void cleanup();

extern "C" SHOGILIBRARY_API int getAllLegalMoves(const char* SFENstring,
                                                 char* output);

extern "C" SHOGILIBRARY_API int getBestMove(const char* SFENstring,
                                            unsigned int maxDepth,
                                            unsigned int maxTime,
                                            bool useGPU,
                                            char* output);

extern "C" SHOGILIBRARY_API int makeMove(const char* SFENString,
                                         const char* moveString,
                                         char* output);