#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <stack>
#include "GameTree.h"
#include "cpuInterface.h"
#include "gpuInterface.h"

namespace shogi {
namespace engine {
struct TreeNode {
  thrust::host_vector<Board> boards;
  thrust::host_vector<Move> moves;
  thrust::host_vector<uint32_t> moveOffsets;
  thrust::host_vector<uint32_t> moveToBoardIdx;
  bool isWhite;
  uint32_t depth;
  TreeNode() {}
  TreeNode(const Board& board, bool isWhite, uint32_t depth)
      : isWhite(isWhite), depth(depth) {
    boards = {board};
  }
  static TreeNode Root(const Board& board,
                       bool isWhite,
                       uint32_t maxDepth,
                       thrust::host_vector<Board>& moveBoards);
};

bool ConstructNodeMovesCPU(TreeNode& node) {
  node.moveOffsets.resize(node.boards.size() + 1);
  thrust::host_vector<Bitboard> validMoves(node.boards.size());
  thrust::host_vector<Bitboard> attackedByEnemy(node.boards.size());
  thrust::host_vector<Bitboard> pinned(node.boards.size());
  bool isMate = false;
  if (node.isWhite) {
    CPU::countWhiteMoves(node.boards.data(), node.boards.size(),
                         validMoves.data(), attackedByEnemy.data(),
                         pinned.data(), node.moveOffsets.data() + 1, &isMate);
  } else {
    CPU::countBlackMoves(node.boards.data(), node.boards.size(),
                         validMoves.data(), attackedByEnemy.data(),
                         pinned.data(), node.moveOffsets.data() + 1, &isMate);
  }
  if (isMate)
    return true;
  CPU::prefixSum(node.moveOffsets.data(), node.moveOffsets.size());
  uint32_t movesCount = node.moveOffsets.back();
  node.moves.resize(movesCount);
  node.moveToBoardIdx.resize(movesCount);
  if (node.isWhite) {
    CPU::generateWhiteMoves(node.boards.data(), node.boards.size(),
                            validMoves.data(), attackedByEnemy.data(),
                            pinned.data(), node.moveOffsets.data(),
                            node.moves.data(), node.moveToBoardIdx.data());
  } else {
    CPU::generateBlackMoves(node.boards.data(), node.boards.size(),
                            validMoves.data(), attackedByEnemy.data(),
                            pinned.data(), node.moveOffsets.data(),
                            node.moves.data(), node.moveToBoardIdx.data());
  }
  return false;
}

bool ConstructNodeMovesGPU(TreeNode& node) {
  thrust::device_vector<uint32_t> moveOffsets(node.boards.size() + 1);
  thrust::device_vector<Bitboard> validMoves(node.boards.size());
  thrust::device_vector<Bitboard> attackedByEnemy(node.boards.size());
  thrust::device_vector<Bitboard> pinned(node.boards.size());
  thrust::device_vector<Board> boards = node.boards;
  thrust::device_vector<bool> isMate(1);
  if (node.isWhite) {
    GPU::countWhiteMoves(boards.data(), boards.size(), validMoves.data(),
                         attackedByEnemy.data(), pinned.data(),
                         moveOffsets.data() + 1, isMate.data());
  } else {
    GPU::countBlackMoves(boards.data(), boards.size(), validMoves.data(),
                         attackedByEnemy.data(), pinned.data(),
                         moveOffsets.data() + 1, isMate.data());
  }
  thrust::host_vector<bool> isMateHost = isMate;
  if (isMate.front()) {
    return true;
  }
  GPU::prefixSum(moveOffsets.data().get(), moveOffsets.size());
  uint32_t movesCount;
  thrust::copy(moveOffsets.end() - 1, moveOffsets.end(), &movesCount);
  thrust::device_vector<Move> moves(movesCount);
  thrust::device_vector<uint32_t> moveToBoardIdx(movesCount);
  if (node.isWhite) {
    GPU::generateWhiteMoves(
        boards.data(), boards.size(), validMoves.data(), attackedByEnemy.data(),
        pinned.data(), moveOffsets.data(), moves.data(), moveToBoardIdx.data());
  } else {
    GPU::generateBlackMoves(
        boards.data(), boards.size(), validMoves.data(), attackedByEnemy.data(),
        pinned.data(), moveOffsets.data(), moves.data(), moveToBoardIdx.data());
  }
  node.moves = moves;
  node.moveOffsets = moveOffsets;
  node.moveToBoardIdx = moveToBoardIdx;
  return false;
}

void GenerateNewNodeCPU(TreeNode& node,
                        TreeNode& newNode,
                        uint32_t movesToProcess,
                        int movesProccessed) {
  newNode.isWhite = !node.isWhite;
  newNode.depth = node.depth - 1;
  newNode.boards.resize(movesToProcess);
  if (node.isWhite) {
    CPU::generateWhiteBoards(
        node.moves.data() + movesProccessed, movesToProcess, node.boards.data(),
        node.moveToBoardIdx.data() + movesProccessed, newNode.boards.data());
  } else {
    CPU::generateBlackBoards(
        node.moves.data() + movesProccessed, movesToProcess, node.boards.data(),
        node.moveToBoardIdx.data() + movesProccessed, newNode.boards.data());
  }
}

void GenerateNewNodeGPU(TreeNode& node,
                        TreeNode& newNode,
                        uint32_t movesToProcess,
                        int movesProccessed) {
  newNode.isWhite = !node.isWhite;
  newNode.depth = node.depth - 1;
  thrust::device_vector<Board> newBoards(movesToProcess);
  thrust::device_vector<Board> boards = node.boards;
  thrust::device_vector<Move> moves(
      node.moves.data() + movesProccessed,
      node.moves.data() + movesProccessed + movesToProcess);
  thrust::device_vector<uint32_t> moveToBoardIdx(
      node.moveToBoardIdx.data() + movesProccessed,
      node.moveToBoardIdx.data() + movesProccessed + movesToProcess);
  if (node.isWhite) {
    GPU::generateWhiteBoards(moves.data(), movesToProcess, boards.data(),
                             moveToBoardIdx.data(), newBoards.data());
  } else {
    GPU::generateBlackBoards(moves.data(), movesToProcess, boards.data(),
                             moveToBoardIdx.data(), newBoards.data());
  }
  newNode.boards = newBoards;
}

int16_t EvaluateMovesCPU(thrust::host_vector<Board> boards, bool isWhite) {
  thrust::host_vector<int16_t> values(boards.size());
  CPU::evaluateBoards(boards.data(), boards.size(), values.data());
  if (isWhite) {
    return *thrust::min_element(thrust::host, values.data(),
                                values.data() + values.size());
  } else {
    return *thrust::max_element(thrust::host, values.data(),
                                values.data() + values.size());
  }
}

int16_t EvaluateMovesGPU(thrust::host_vector<Board> boards, bool isWhite) {
  thrust::device_vector<Board> d_boards = boards;
  thrust::device_vector<int16_t> values(boards.size());
  GPU::evaluateBoards(d_boards.data(), d_boards.size(), values.data());
  if (isWhite) {
    return *thrust::min_element(thrust::device, values.data(),
                                values.data() + values.size());
  } else {
    return *thrust::max_element(thrust::device, values.data(),
                                values.data() + values.size());
  }
}

int16_t GameTree::SearchNode(TreeNode& node) {
  positionsSearched[node.depth] += node.boards.size();
  if (node.depth == 0) {
    return (node.boards.size() >= m_minBoardsGPU
                ? EvaluateMovesGPU(node.boards, node.isWhite)
                : EvaluateMovesCPU(node.boards, node.isWhite));
  }
  if (node.boards.size() >= m_minBoardsGPU ? ConstructNodeMovesGPU(node)
                                           : ConstructNodeMovesCPU(node)) {
    return (node.isWhite ? -(PieceValue::MATE + node.depth)
                         : PieceValue::MATE + node.depth);
  }
  uint32_t movesProcessed = 0;
  std::vector<int16_t> values(
      std::ceil(node.moves.size() / (double)m_maxProcessedSize));
  int idx = 0;
  while (movesProcessed < node.moves.size()) {
    uint32_t movesToProcess = std::min(node.moves.size() - movesProcessed,
                                       (size_t)m_maxProcessedSize);
    TreeNode newNode;
    movesToProcess >= m_minBoardsGPU
        ? GenerateNewNodeGPU(node, newNode, movesToProcess, movesProcessed)
        : GenerateNewNodeCPU(node, newNode, movesToProcess, movesProcessed);
    values[idx] = SearchNode(newNode);
    idx++;
    movesProcessed += movesToProcess;
  }
  if (node.isWhite)
    return *std::max_element(values.begin(), values.end());
  else
    return *std::min_element(values.begin(), values.end());
}

Move GameTree::FindBestMove() {
  positionsSearched.resize(m_maxDepth);
  GPU::initLookUpArrays();
  thrust::host_vector<Board> moveBoards;
  TreeNode root = TreeNode::Root(m_startingBoard, m_startingIsWhite, m_maxDepth,
                                 moveBoards);
  std::vector<int16_t> moveValues(root.moves.size());
  for (int i = 0; i < moveBoards.size(); i++) {
    TreeNode node(moveBoards[i], !root.isWhite, root.depth - 1);
    moveValues[i] = SearchNode(node);
  }
  int bestMoveIdx = 0;
  if (root.isWhite) {
    auto elem = std::max_element(moveValues.begin(), moveValues.end());
    bestMoveIdx = elem - moveValues.begin();
  } else {
    auto elem = std::min_element(moveValues.begin(), moveValues.end());
    bestMoveIdx = elem - moveValues.begin();
  }
  for (int i = m_maxDepth - 1; i >= 0; i--) {
    std::cout << "Positions on depth " << i << " : " << positionsSearched[i]
              << std::endl;
  }
  return root.moves[bestMoveIdx];
}

TreeNode TreeNode::Root(const Board& board,
                        bool isWhite,
                        uint32_t maxDepth,
                        thrust::host_vector<Board>& moveBoards) {
  TreeNode root;
  root.boards = {board};
  root.isWhite = isWhite;
  root.depth = maxDepth;
  ConstructNodeMovesCPU(root);
  moveBoards.resize(root.moves.size());
  if (isWhite) {
    CPU::generateWhiteBoards(root.moves.data(), root.moves.size(),
                             root.boards.data(), root.moveToBoardIdx.data(),
                             moveBoards.data());
  } else {
    CPU::generateBlackBoards(root.moves.data(), root.moves.size(),
                             root.boards.data(), root.moveToBoardIdx.data(),
                             moveBoards.data());
  }
  return root;
}

std::vector<Move> GameTree::GetAllMovesFrom(const Board& board, bool isWhite) {
  TreeNode node(board, isWhite, 1);
  ConstructNodeMovesCPU(node);
  std::vector<Move> moves(node.moves.begin(), node.moves.end());
  return moves;
}

/// <summary>
/// Wez n pozycji z CPU do GPU. Buduj na GPU az wywali pamiêæ albo g³êbokoœæ.
/// Jeœli wywali³o pamiêæ to przenieœ wszystko na CPU i wywo³aj ponownie z nowym
/// ptr do values. Na koniec przenieœ z powrotem do GPU razem z values. Odpal
/// propagacje w dó³ i zwróæ values.
/// </summary>
///

// True if reached max depth false if not
Move GetBestMove(uint8_t* d_Buffer,
                 uint32_t d_BufferSize,
                 const Board& board,
                 bool isWhite,
                 uint16_t depth,
                 uint16_t maxDepth) {
  Board* d_Board = (Board*)d_Buffer;
  cudaMemcpy(d_Board, &board, sizeof(Board), cudaMemcpyHostToDevice);
  uint8_t* bufferBegin = d_Buffer + sizeof(Board);
  uint8_t* bufferEnd = d_Buffer + d_BufferSize;
  Move* movesPtr = (Move*)(bufferEnd);
  uint32_t* offsetsPtr = (uint32_t*)bufferBegin;
  std::vector<uint32_t> layerSize;
  layerSize.push_back(1);
  // To count how much moves it will generate we need
  // (size+1) * sizeof(uint32_t) + 3 * 3 * size * sizeof(uint32_t) +
  // allMovesSize * depth * sizeof(Move)
  uint32_t occupiedMemmory = layerSize.back() * depth * sizeof(Move);
  while (depth < maxDepth) {
    occupiedMemmory += (layerSize.back() + 1) * sizeof(uint32_t);
    if (occupiedMemmory + 3 * 3 * layerSize.back() * sizeof(uint32_t) >=
        d_BufferSize) {
      break;
    }
    uint32_t* tmpBitboardsPtr = offsetsPtr + layerSize.back() + 1;
    // Count next moves
    isWhite ? GPU::countWhiteMoves(layerSize.back(), depth, d_Board, movesPtr,
                                   offsetsPtr, tmpBitboardsPtr)
            : GPU::countBlackMoves(layerSize.back(), depth, d_Board, movesPtr,
                                   offsetsPtr, tmpBitboardsPtr);
    GPU::prefixSum(offsetsPtr, layerSize.back() + 1);
    uint32_t nextLayerSize = 0;
    cudaMemcpy(&nextLayerSize, offsetsPtr + layerSize.back(), sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    occupiedMemmory += nextLayerSize * (depth + 1) * sizeof(Move);
    if (occupiedMemmory + 3 * 3 * layerSize.back() * sizeof(uint32_t) >=
        d_BufferSize) {
      break;
    }

    Move* newMovesPtr = movesPtr - nextLayerSize * (depth + 1);
    isWhite
        ? GPU::generateWhiteMoves(layerSize.back(), depth, d_Board, movesPtr,
                                  offsetsPtr, tmpBitboardsPtr, newMovesPtr)
        : GPU::generateBlackMoves(layerSize.back(), depth, d_Board, movesPtr,
                                  offsetsPtr, tmpBitboardsPtr, newMovesPtr);

    offsetsPtr += layerSize.back() + 1;
    movesPtr = newMovesPtr;
    layerSize.push_back(nextLayerSize);
    isWhite = !isWhite;
    depth++;
    std::cout << "Generated: " << layerSize.back()
              << " positions on depth: " << depth << std::endl;
  }
  // After filling up space evaluate boards
  // We can place values in place of last values
  int16_t* valuesPtr = (int16_t*)movesPtr;
  uint32_t valuesOffset = layerSize.back();
  GPU::evaluateBoards(layerSize.back(), depth, d_Board, movesPtr, valuesPtr);
  // We can use board memmory for index
  uint32_t* bestIndex = (uint32_t*)d_Board;
  // Collect values from upper layers
  for (uint16_t d = depth; d > 0; d--) {
    isWhite = !isWhite;
    offsetsPtr -= layerSize[d - 1] + 1;
    isWhite ? GPU::gatherValuesMax(layerSize[d - 1], d, offsetsPtr, valuesPtr,
                                   valuesPtr + valuesOffset, bestIndex)
            : GPU::gatherValuesMin(layerSize[d - 1], d, offsetsPtr, valuesPtr,
                                   valuesPtr + valuesOffset, bestIndex);
    valuesPtr += valuesOffset;
  }
  int16_t bestValue;
  cudaMemcpy(&bestValue, valuesPtr, sizeof(int16_t), cudaMemcpyDeviceToHost);
  uint32_t h_bestIndex;
  cudaMemcpy(&h_bestIndex, bestIndex, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  Move bestMove;
  movesPtr = (Move*)(bufferEnd - layerSize[1] * sizeof(Move));
  cudaMemcpy(&bestMove, movesPtr + h_bestIndex, sizeof(Move),
             cudaMemcpyDeviceToHost);
  std::cout << "Done" << std::endl;
  return bestMove;
}

}  // namespace engine
}  // namespace shogi