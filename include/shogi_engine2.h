#pragma once

//#define SHOGILIBRARY2_API __declspec(dllexport)

extern "C" void getAllLegalMoves(const char* input,
                                                   char* output);

extern "C" void getBestMove(const char* input,
                                              unsigned int maxDepth,
                                              unsigned int maxTime,
                                              char* output);