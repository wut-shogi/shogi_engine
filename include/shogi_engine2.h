#pragma once

#define SHOGILIBRARY2_API __declspec(dllexport)

extern "C" SHOGILIBRARY2_API void getAllLegalMoves(const char* input,
                                                   char* output);

extern "C" SHOGILIBRARY2_API void getBestMove(const char* input,
                                              unsigned int maxDepth,
                                              unsigned int maxTime,
                                              char* output);