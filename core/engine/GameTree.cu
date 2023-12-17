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

struct TreeNode2 {
  size_t size;
  std::vector<Move> moves;
  std::vector<uint32_t> childrenOffsets;
  std::vector<uint32_t> layersSize;
  std::vector<int16_t> values;
  bool isWhite;
  uint16_t depth;
};

struct Node {
  size_t size;
  Move* moves;
  uint32_t* childrenOffsets;
  std::vector<uint32_t> layerSize;
  bool isWhite;
  uint16_t depth;
};

uint8_t* GPUbuffer;
uint32_t GPUBufferSize;
const Board& startingBoard;
void PropagateValues(bool isWhite,
                     uint16_t depth,
                     std::vector<uint32_t>& layerChildrenOffsetsOffset,
                     uint32_t size) {
  Move* movesPtr = ((Move*)(GPUbuffer + GPUBufferSize)) - size;
  uint32_t moveLayerSize = size;
  uint32_t* childrenOffsetsPtr =
      (uint32_t*)(GPUbuffer +
                  layerChildrenOffsetsOffset.back() * sizeof(uint32_t));

  do {
    // isWhite ? GPU::chooseValuesMax(size, depth, startingBoard, movesPtr, childrenOffsetsPtr, 
  } while (!layerChildrenOffsetsOffset.empty());
}

void SearchNodesGPU(bool isWhite,
                    uint32_t size,
                    uint16_t currentDepth,
                    uint16_t maxDepth,
                    int16_t* values) {
  static uint32_t GPUminSize = 10000;
  bool outOfMemmory = false;
  std::vector<uint32_t> layerChildrenOffsetsOffset;
  std::stack<Node> nodes;
  uint32_t* childrenOffsetsPtr = (uint32_t*)GPUbuffer;
  uint32_t layerSizeSum = 0;
  layerChildrenOffsetsOffset.push_back(0);
  layerSizeSum += size + 1;
  Move* movesPtr = ((Move*)(GPUbuffer + GPUBufferSize)) - size;
  while (currentDepth < maxDepth && !outOfMemmory) {
    uint32_t* tmpBitboards = childrenOffsetsPtr + size;
    isWhite ? GPU::countWhiteMoves(size, currentDepth, startingBoard, movesPtr,
                                   childrenOffsetsPtr, tmpBitboards)
            : GPU::countBlackMoves(size, currentDepth, startingBoard, movesPtr,
                                   childrenOffsetsPtr, tmpBitboards);
    // Check if mate
    GPU::prefixSum(childrenOffsetsPtr, size + 1);
    // First move old moves
    Move* oldMovesPtr = (Move*)(tmpBitboards + 3 * 3 * size);
    cudaMemcpy(oldMovesPtr, movesPtr, size * sizeof(Move),
               cudaMemcpyDeviceToDevice);
    // Get newMoves size
    uint32_t newSize;
    cudaMemcpy(&newSize, currentChildrenOffsetsPtr + size, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    // Find ptr of new moves
    movesPtr = ((Move*)(buffer + GPUBufferSize)) - newSize;

    GPU::generateWhiteMoves(size, currentDepth, startingBoard, oldMovesPtr,
                            currentChildrenOffsetsPtr, tmpBitboards, movesPtr);
  }
  if (outOfMemmory) {
    // Dump current state to CPU and continue
    uint32_t* h_childrenOffsetsPtr = new uint32_t[layerSizeSum];
    cudaMemcpy(h_childrenOffsetsPtr, GPUBuffer, layerSizeSum * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    Move* h_movesPtr = new Move[size];
    uint32_t currentMovesOffset = size - GPUminSize;
    int16_t* values = new int16_t[size];
    uint32_t currentValuesOffset = size - GPUminSize;
    // Gather all values
    do {
      SearchNodesGPU(GPUBuffer, GPUBufferSize, startingBoard, isWhite,
                     GPUminSize, currentDepth, maxDepth,
                     values + currentValuesOffset);
      cudaMemcpy(h_movesPtr + currentMovesOffset,
                 GPUBuffer - GPUminSize * sizeof(Move),
                 GPUminSize * sizeof(Move), cudaMemcpyHostToDevice);
      currentMovesOffset -= GPUminSize;
      if (currentMovesOffset < 0) {
        currentMovesOffset = 0;
      }
    } while (currentMovesOffset >= 0);

    // If all values gathered move data back to GPU to propagate values higher
    cudaMemcpy(GPUBuffer, h_childrenOffsetsPtr, layerSizeSum * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(movesPtr, h_movesPtr, size * sizeof(Move),
               cudaMemcpyHostToDevice);
    // propagateValues();
  }
}

void SearchNodes(uint32_t GPUBufferSize,
                 const Board& startingBoard,
                 bool isWhite,
                 Move* moves,
                 uint32_t size,
                 uint16_t currentDepth,
                 uint16_t maxDepth,
                 std::vector<int16_t>& values) {
  uint32_t GPUminSize;
  // allocate whole buffer on GPU
  uint8_t* buffer;
  cudaMalloc(&buffer, GPUBufferSize);
  std::stack<Node> stack;
  bool maxDepthReached = false;
  std::vector<uint32_t> childrenOffsetsLayerBegin;
  // Left side of buffer are childrenOffsets of each tree Level
  // Right side of buffer is for storing moves
  // You can write an read moves at the same time cause it is not possible to
  // place move of a child after its parent
  uint32_t* childrenOffsetsPtr = (uint32_t*)buffer;
  childrenOffsetsLayerBegin.push_back(0);
  // Copy whole move buffer to GPU buffer
  // Moves are stored in continuous memmory layer next to layer.
  uint32_t movesSize = currentDepth * size;
  Move* movesPtr = ((Move*)(buffer + GPUBufferSize)) - movesSize;
  cudaMemcpy(movesPtr, moves, movesSize * sizeof(Move), cudaMemcpyHostToDevice);
  // Generate next levels of tree untill desired or memmory shortage
  while (maxDepthReached && stack.empty()) {
    // To count next moves we need:
    // currentMovesSize * 4 (moveOffsets) + currentMovesSize * 3 * 3 * 4
    // (tmpBitboards)
    if ((uint8_t*)movesPtr - ((uint8_t*)childrenOffsetsPtr + movesSize * 4) <
        movesSize * (4 + 3 * 3 * 4)) {
      // Not enough space
      // Save to CPU stack
      Node node;
      node.isWhite = isWhite;
      node.depth = currentDepth;
      node.size = movesSize - GPUminSize;
      node.moves.resize(node.size);
      node.childrenOffsetsLayerBegin = childrenOffsetsLayerBegin;
      node.childrenOffsets.resize(childrenOffsetsLayerBegin.back() + movesSize);
      cudaMemcpy(node.moves.data(), movesPtr, node.size * sizeof(Move),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(node.childrenOffsets.data(), childrenOffsetsPtr,
                 node.childrenOffsets.size() * sizeof(uint32_t),
                 cudaMemcpyDeviceToHost);

      movesSize = GPUminSize;
      movesPtr = ((Move*)(buffer + GPUBufferSize)) - movesSize;
      childrenOffsetsPtr = (uint32_t*)buffer;
    }
    // 3 * size of uint32_t is for storing bitboard. We need 3.
    uint32_t* tmpBitboards = childrenOffsetsPtr + size;
    isWhite ? GPU::countWhiteMoves(size, currentDepth, startingBoard, movesPtr,
                                   currentChildrenOffsetsPtr, tmpBitboards)
            : GPU::countBlackMoves(size, currentDepth, startingBoard, movesPtr,
                                   currentChildrenOffsetsPtr, tmpBitboards);
    // Check if mate
    GPU::prefixSum(currentChildrenOffsetsPtr, size + 1);

    // First move old moves
    Move* oldMovesPtr = (Move*)(tmpBitboards + 3 * 3 * size);
    cudaMemcpy(oldMovesPtr, movesPtr, size * sizeof(Move),
               cudaMemcpyDeviceToDevice);
    // Get newMoves size
    uint32_t newSize;
    cudaMemcpy(&newSize, currentChildrenOffsetsPtr + size, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    // Find ptr of new moves
    movesPtr = ((Move*)(buffer + GPUBufferSize)) - newSize;

    GPU::generateWhiteMoves(size, currentDepth, startingBoard, oldMovesPtr,
                            currentChildrenOffsetsPtr, tmpBitboards, movesPtr);
  }
}

void GameTree::SearchNode2(const Board& startBoard,
                           TreeNode2& node,
                           std::vector<int16_t> values) {
  // If max depth evaluate values an return
  if (node.depth >= m_maxDepth) {
    CPU::evaluateBoards(node.size, node.depth, startBoard, node.moves.data(),
    // values.data());
    return;
  }

  // Count number of total moves
  thrust::host_vector<Bitboard> validMoves(node.size);
  thrust::host_vector<Bitboard> attackedByEnemy(node.size);
  thrust::host_vector<Bitboard> pinned(node.size);
  // countMoves(node.size, node.depth, startBoard, node.moves.data(),
  // node.children.data()+1, validMoves.data(), attackedByEnemy.data(),
  // pinned.data()); prefixSum(children.data(), node.children.size()); Generate
  // next moves
  uint32_t numberOfProcesedMoves = 0;
  uint32_t numberOfTotalMoves = node.size;
  thrust::host_vector<int16_t> childrenValues(m_maxProcessedSize);
  while (numberOfProcesedMoves < numberOfTotalMoves) {
    uint32_t movesToProcess = std::min(
        numberOfTotalMoves - numberOfProcesedMoves, m_maxProcessedSize);

    TreeNode2 newNode;
    newNode.depth = node.depth + 1;
    newNode.isWhite = !node.isWhite;
    newNode.size = movesToProcess;
    newNode.moves = thrust::host_vector<Move>(newNode.size * newNode.depth);
    newNode.children =
        thrust::host_vector<uint32_t>(newNode.size * newNode.depth);
    // Generate moves
    // generateMoves(node.size, node.depth, startBoard, node.moves.data(),
    // newNode.moves.data())
    SearchNode2(startBoard, newNode, childrenValues);
    // Gather values from children
    // gatherValues(movesToProcess, node.depth, node.children.data() +
    // movestoProcess, childrenValues.data(), values.data());
  }
}

}  // namespace engine
}  // namespace shogi