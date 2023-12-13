#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "GameTree.h"
#include "cpuInterface.h"
#include "gpuInterface.h"
#include <thrust/extrema.h>

namespace shogi {
namespace engine {
struct TreeNode {
  thrust::host_vector<Board> boards;
  thrust::host_vector<Move> moves;
  thrust::host_vector<uint32_t> moveOffsets;
  thrust::host_vector<uint32_t> moveToBoardIdx;
  bool isWhite;
  uint32_t depth;
};

void ConstructNodeMovesCPU(TreeNode& node) {
  node.moveOffsets.resize(node.boards.size() + 1);
  thrust::host_vector<Bitboard> validMoves(node.boards.size());
  thrust::host_vector<Bitboard> attackedByEnemy(node.boards.size());
  if (node.isWhite) {
    CPU::countWhiteMoves(node.boards.data(), node.boards.size(),
                         validMoves.data(), attackedByEnemy.data(),
                         node.moveOffsets.data() + 1);
  } else {
    CPU::countBlackMoves(node.boards.data(), node.boards.size(),
                         validMoves.data(), attackedByEnemy.data(),
                         node.moveOffsets.data() + 1);
  }
  CPU::prefixSum(node.moveOffsets.data(), node.moveOffsets.size());
  uint32_t movesCount = node.moveOffsets.back();
  node.moves.resize(movesCount);
  node.moveToBoardIdx.resize(movesCount);
  if (node.isWhite) {
    CPU::generateWhiteMoves(node.boards.data(), node.boards.size(),
                            validMoves.data(), attackedByEnemy.data(),
                            node.moveOffsets.data(), node.moves.data(),
                            node.moveToBoardIdx.data());
  } else {
    CPU::generateBlackMoves(node.boards.data(), node.boards.size(),
                            validMoves.data(), attackedByEnemy.data(),
                            node.moveOffsets.data(), node.moves.data(),
                            node.moveToBoardIdx.data());
  }
}

void ConstructNodeMovesGPU(TreeNode& node) {
  thrust::device_vector<uint32_t> moveOffsets(node.boards.size() + 1);
  thrust::device_vector<Bitboard> validMoves(node.boards.size());
  thrust::device_vector<Bitboard> attackedByEnemy(node.boards.size());
  thrust::device_vector<Board> boards = node.boards;
  if (node.isWhite) {
    GPU::countWhiteMoves(boards.data(), boards.size(), validMoves.data(),
                         attackedByEnemy.data(), moveOffsets.data() + 1);
  } else {
    GPU::countBlackMoves(boards.data(), boards.size(), validMoves.data(),
                         attackedByEnemy.data(), moveOffsets.data() + 1);
  }
  GPU::prefixSum(moveOffsets.data(), moveOffsets.size());
  uint32_t movesCount;
  thrust::copy(moveOffsets.end() - 1, moveOffsets.end(), &movesCount);
  thrust::device_vector<Move> moves(movesCount);
  thrust::device_vector<uint32_t> moveToBoardIdx(movesCount);
  if (node.isWhite) {
    GPU::generateWhiteMoves(boards.data(), boards.size(), validMoves.data(),
                            attackedByEnemy.data(), moveOffsets.data(),
                            moves.data(), moveToBoardIdx.data());
  } else {
    GPU::generateBlackMoves(boards.data(), boards.size(), validMoves.data(),
                            attackedByEnemy.data(), moveOffsets.data(),
                            moves.data(), moveToBoardIdx.data());
  }
  node.moves = moves;
  node.moveOffsets = moveOffsets;
  node.moveToBoardIdx = moveToBoardIdx;
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

int16_t EvaluateMovesCPU(thrust::host_vector<Board> boards) {
  thrust::host_vector<int16_t> values(boards.size());
  CPU::evaluateBoards(boards.data(), boards.size(), values.data());
  return *thrust::max_element(thrust::host, values.data(),
                             values.data() + values.size());
}

int16_t EvaluateMovesGPU(thrust::host_vector<Board> boards) {
  thrust::device_vector<Board> d_boards = boards;
  thrust::device_vector<int16_t> values(boards.size());
  GPU::evaluateBoards(d_boards.data(), d_boards.size(), values.data());
  return *thrust::max_element(thrust::device, values.data(),
                              values.data() + values.size());
}

int16_t GameTree::SearchNode(TreeNode& node) {
  positionsSearched[node.depth] += node.boards.size();
  if (node.depth == 0) {
    return (node.boards.size() >= m_minBoardsGPU
                ? EvaluateMovesGPU(node.boards)
                : EvaluateMovesCPU(node.boards));
  }
  node.boards.size() >= m_minBoardsGPU ? ConstructNodeMovesGPU(node)
                                       : ConstructNodeMovesCPU(node);
  uint32_t movesProcessed = 0;
  thrust::host_vector<int16_t> values(
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
  return *std::max_element(values.begin(), values.end());
}

Move GameTree::FindBestMove() {
  GPU::initLookUpArrays();
  TreeNode root;
  root.isWhite = m_startingIsWhite;
  root.depth = m_maxDepth;
  root.boards = {m_startingBoard};
  positionsSearched = std::vector<uint32_t>(m_maxDepth + 1, 0);
  ConstructNodeMovesCPU(root);
  TreeNode avaliableMoves;
  GenerateNewNodeCPU(root, avaliableMoves, root.moves.size(), 0);
  int16_t maxVal = INT16_MIN;
  size_t maxIdx = 0;
  for (int i = 0; i < avaliableMoves.boards.size(); i++) {
    TreeNode moveRoot;
    moveRoot.depth = 1;
    moveRoot.isWhite = avaliableMoves.isWhite;
    moveRoot.boards = {avaliableMoves.boards[i]};
    int16_t moveValue = SearchNode(moveRoot);
    if (moveValue > maxVal) {
      maxVal = moveValue;
      maxIdx = i;
    }
  }
  for (int i = m_maxDepth; i >= 0; i--) {
    std::cout << "Positions on depth " << i << " : " << positionsSearched[i]
              << std::endl;
  }
  return root.moves[maxIdx];
}
}  // namespace engine
}  // namespace shogi