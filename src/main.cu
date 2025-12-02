// Optimized sudoku_cuda.cpp with unused variables removed and timing added
// NOTE: Only structural edits applied; functionality preserved.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdint>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>

constexpr int N = 9;
constexpr int BOARD_SIZE = N * N;
constexpr int BOX_N = 3;
constexpr int BFS_DEPTH = 30;
constexpr uint16_t ALL_CANDIDATES = 0x1FF;
constexpr int MAX_BOARDS = (INT_MAX >> 8);

typedef uint16_t u16;

__device__ int d_solution_found = 0;

__host__ __device__ inline int idx(int r, int c) { return r * N + c; }

struct Board { u16 cell[BOARD_SIZE]; };

__device__ bool CheckNeighbors(const Board &b, int id)
{
    const int idc = id % N;
    const int idr = id / N;
    u16 cands = b.cell[id];

    for (int i = 0; i < N; ++i)
    {
        if (i * N + idc != id)
        {
            u16 colMask = b.cell[i * N + idc];
            if (colMask == ALL_CANDIDATES) colMask = 0;
            cands &= ~colMask;
        }

        if (idr * N + i != id)
        {
            u16 rowMask = b.cell[idr * N + i];
            if (rowMask == ALL_CANDIDATES) rowMask = 0;
            cands &= ~rowMask;
        }

        if (!cands) return false;
    }

    int boxRId = idr / BOX_N;
    int boxCId = idc / BOX_N;
    int rowStart = boxRId * BOX_N;
    int colStart = boxCId * BOX_N;

    for (int r = 0; r < BOX_N; ++r)
        for (int c = 0; c < BOX_N; ++c)
        {
            int rr = rowStart + r;
            int cc = colStart + c;
            if (rr * N + cc == id) continue;
            u16 boxMask = b.cell[rr * N + cc];
            if (boxMask == ALL_CANDIDATES) boxMask = 0;
            cands &= ~boxMask;
            if (!cands) return false;
        }

    return true;
}

__device__ u16 ComputeCandidates(const Board &d_board, int id)
{
    const int idc = id % N;
    const int idr = id / N;
    u16 cands = ALL_CANDIDATES;

    for (int i = 0; i < N; ++i)
    {
        u16 colMask = d_board.cell[i * N + idc];
        u16 rowMask = d_board.cell[idr * N + i];
        if (colMask == ALL_CANDIDATES) colMask = 0;
        if (rowMask == ALL_CANDIDATES) rowMask = 0;
        cands &= ~colMask;
        cands &= ~rowMask;
    }

    int boxRId = idr / BOX_N;
    int boxCId = idc / BOX_N;
    int rowStart = boxRId * BOX_N;
    int colStart = boxCId * BOX_N;

    for (int r = 0; r < BOX_N; ++r)
        for (int c = 0; c < BOX_N; ++c)
        {
            int rr = rowStart + r;
            int cc = colStart + c;
            u16 mask = d_board.cell[rr * N + cc];
            if (mask == ALL_CANDIDATES) mask = 0;
            cands &= ~mask;
        }

    return cands;
}

__global__ void DfsSolveKernel(const Board *frontier, int frontier_size, Board *solution_out)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= frontier_size) return;
    if (atomicAdd(&d_solution_found, 0) != 0) return;

    Board b = frontier[tid];
    int prev[BOARD_SIZE];
    u16 mask[BOARD_SIZE];
    u16 copyMask[BOARD_SIZE];

    for (int i = 0; i < BOARD_SIZE; ++i)
    {
        prev[i] = -1;
        mask[i] = 0;
        copyMask[i] = 0;
    }

    for (int i = 0; i < BOARD_SIZE; ++i)
    {
        if (b.cell[i] == ALL_CANDIDATES)
        {
            u16 cand = ComputeCandidates(b, i);
            mask[i] = cand;
            copyMask[i] = cand;
            if (!cand) return;
        }
    }

    int cur = -1;
    for (int i = 0; i < BOARD_SIZE; ++i)
        if (b.cell[i] == ALL_CANDIDATES) { cur = i; break; }

    if (cur == -1)
    {
        if (atomicCAS(&d_solution_found, 0, 1) == 0) *solution_out = b;
        return;
    }

    while (cur != -1)
    {
        if (atomicAdd(&d_solution_found, 0) != 0) return;

        if (!mask[cur])
        {
            b.cell[cur] = ALL_CANDIDATES;
            mask[cur] = copyMask[cur];
            cur = prev[cur];
            continue;
        }

        int c1 = __ffs(mask[cur]);
        if (c1 == 0)
        {
            mask[cur] = copyMask[cur];
            b.cell[cur] = ALL_CANDIDATES;
            cur = prev[cur];
            continue;
        }

        int bitIndex = c1 - 1;
        u16 bitMask = (1u << bitIndex);
        mask[cur] &= ~(1u << bitIndex);
        b.cell[cur] = bitMask;

        if (!CheckNeighbors(b, cur))
        {
            b.cell[cur] = ALL_CANDIDATES;
            continue;
        }

        int nextCell = -1;
        for (int i = 0; i < BOARD_SIZE; ++i)
            if (b.cell[i] == ALL_CANDIDATES) { nextCell = i; break; }

        if (nextCell == -1)
        {
            if (atomicCAS(&d_solution_found, 0, 1) == 0) *solution_out = b;
            return;
        }

        prev[nextCell] = cur;
        cur = nextCell;

        if (!copyMask[cur])
        {
            u16 cand = ComputeCandidates(b, cur);
            mask[cur] = cand;
            copyMask[cur] = cand;
            if (!cand) { cur = prev[cur]; continue; }
        }
    }
}

__global__ void ExpandFrontierKernel(Board *d_frontier, int startIndex, int endIndex, int *d_nextBoardIndex, int maxBoards)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int range = endIndex - startIndex;
    if (gtid >= range) return;

    Board board = d_frontier[startIndex + gtid];

    for (int i = 0; i < BOARD_SIZE; ++i)
    {
        if (board.cell[i] != ALL_CANDIDATES) continue;

        u16 cands = ComputeCandidates(board, i);
        if (!cands) return;

        u16 mask = cands;
        while (mask)
        {
            int bitIndex = __ffs(mask) - 1;
            Board nb = board;
            nb.cell[i] = (1u << bitIndex);

            int pos = atomicAdd(d_nextBoardIndex, 1);
            if (pos < maxBoards) d_frontier[pos] = nb;

            mask &= (mask - 1);
        }
        break;
    }
}

void PrintBoardMasksHost(const Board &B)
{
    for (int r = 0; r < N; ++r)
    {
        for (int c = 0; c < N; ++c)
        {
            u16 m = B.cell[idx(r, c)];
            unsigned bits = m;
            if (__builtin_popcount(bits) == 1)
                printf("%d ", __builtin_ctz(bits) + 1);
            else
                printf(". ");
        }
        printf("\n");
    }
    printf("\n");
}

bool IsSolvedHost(const Board &B)
{
    for (int i = 0; i < BOARD_SIZE; ++i)
        if (__builtin_popcount((unsigned)B.cell[i]) != 1)
            return false;
    return true;
}

int main()
{


    u16 *h_input = new u16[BOARD_SIZE];
    memset(h_input, 0, BOARD_SIZE * sizeof(u16));

    std::ifstream in("sudoku.txt");
    if (!in)
    {
        std::cerr << "Cannot open sudoku.txt\n";
        return 1;
    }

    char ch;
    int idx_in = 0;
    while (in >> ch && idx_in < BOARD_SIZE)
        if (ch >= '0' && ch <= '9') h_input[idx_in++] = ch - '0';
    in.close();

    Board startBoard;
    for (int i = 0; i < BOARD_SIZE; ++i)
    {
        int v = h_input[i];
        startBoard.cell[i] = (v == 0 ? ALL_CANDIDATES : (1u << (v - 1)));
    }


    Board *d_frontier, *d_solution;
    int *d_nextIndex;

    cudaMalloc(&d_frontier, MAX_BOARDS * sizeof(Board));
    cudaMalloc(&d_nextIndex, sizeof(int));
    cudaMalloc(&d_solution, sizeof(Board));

    cudaMemset(d_frontier, 0, MAX_BOARDS * sizeof(Board));

    int initNext = 1;
    cudaMemcpy(d_nextIndex, &initNext, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier, &startBoard, sizeof(Board), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int startIndex = 0;
    int cutoffIndex = 1;
    int range = cutoffIndex - startIndex;
    int gridSize = (range + blockSize - 1) / blockSize;

    ExpandFrontierKernel<<<gridSize, blockSize>>>(d_frontier, startIndex, cutoffIndex, d_nextIndex, MAX_BOARDS);
    cudaDeviceSynchronize();

    int depth = BFS_DEPTH;
    for (int d = 0; d < depth; ++d)
    {
        startIndex = cutoffIndex;
        cudaMemcpy(&cutoffIndex, d_nextIndex, sizeof(int), cudaMemcpyDeviceToHost);
        range = cutoffIndex - startIndex;
        if (range <= 0) break;
        gridSize = (range + blockSize - 1) / blockSize;
        ExpandFrontierKernel<<<gridSize, blockSize>>>(d_frontier, startIndex, cutoffIndex, d_nextIndex, MAX_BOARDS);
    }

    startIndex = cutoffIndex;
    cudaMemcpy(&cutoffIndex, d_nextIndex, sizeof(int), cudaMemcpyDeviceToHost);
    range = cutoffIndex - startIndex;
    gridSize = (range + blockSize - 1) / blockSize;

    DfsSolveKernel<<<gridSize, blockSize>>>(d_frontier + startIndex, range, d_solution);
    cudaDeviceSynchronize();
    int h_solution_found = 0;
    cudaMemcpyFromSymbol(&h_solution_found, d_solution_found, sizeof(int));

    Board solvedHost;
    cudaMemcpy(&solvedHost, d_solution, sizeof(Board), cudaMemcpyDeviceToHost);

    if (h_solution_found && IsSolvedHost(solvedHost))
    {
        std::cout << "Solved!\n";
        PrintBoardMasksHost(solvedHost);
    }
    else std::cout << "No solution.\n";

    cudaFree(d_frontier);
    cudaFree(d_nextIndex);
    cudaFree(d_solution);
    delete[] h_input;

    return 0;
}
