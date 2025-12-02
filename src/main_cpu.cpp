// sudoku_cpu.cpp
// Pure C++ port of the provided CUDA program.
// Build: g++ -O3 -std=c++17 sudoku_cpu.cpp -o sudoku_cpu

#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>

constexpr int N = 9;
constexpr int BOARD_SIZE = N * N;
constexpr int BOX_N = 3;
constexpr int BFS_DEPTH = 30;
constexpr uint16_t ALL_CANDIDATES = 0x1FF; // bits 0..8 set
constexpr int MAX_BOARDS = (INT_MAX >> 8);

using u16 = uint16_t;

inline int idx(int r, int c) { return r * N + c; }

struct Board {
    u16 cell[BOARD_SIZE];
};

inline int popcount(unsigned x) { return __builtin_popcount(x); }
inline int ctz(unsigned x) { return __builtin_ctz(x); }
inline int ffs_int(unsigned x) { return __builtin_ffs(x); } // returns 1-based index or 0

// Compute candidates for cell id given board
u16 ComputeCandidates(const Board &b, int id) {
    const int idc = id % N;
    const int idr = id / N;
    u16 cands = ALL_CANDIDATES;

    // remove row/col
    for (int i = 0; i < N; ++i) {
        u16 colMask = b.cell[i * N + idc];
        u16 rowMask = b.cell[idr * N + i];
        if (colMask == ALL_CANDIDATES) colMask = 0;
        if (rowMask == ALL_CANDIDATES) rowMask = 0;
        cands &= ~colMask;
        cands &= ~rowMask;
    }

    // remove 3x3 box
    int boxRId = idr / BOX_N;
    int boxCId = idc / BOX_N;
    int rowStart = boxRId * BOX_N;
    int colStart = boxCId * BOX_N;
    for (int r = 0; r < BOX_N; ++r)
        for (int c = 0; c < BOX_N; ++c) {
            int rr = rowStart + r;
            int cc = colStart + c;
            u16 mask = b.cell[rr * N + cc];
            if (mask == ALL_CANDIDATES) mask = 0;
            cands &= ~mask;
        }

    return cands;
}

// CheckNeighbors: ensure cell id has at least one candidate after removing neighbors' fixed values
bool CheckNeighbors(const Board &b, int id) {
    const int idc = id % N;
    const int idr = id / N;
    u16 cands = b.cell[id];
    if (cands == ALL_CANDIDATES) cands = ALL_CANDIDATES; // keep sentinel

    for (int i = 0; i < N; ++i) {
        if (i * N + idc != id) {
            u16 colMask = b.cell[i * N + idc];
            if (colMask == ALL_CANDIDATES) colMask = 0;
            cands &= ~colMask;
        }
        if (idr * N + i != id) {
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
        for (int c = 0; c < BOX_N; ++c) {
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

void PrintBoardMasksHost(const Board &B) {
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            u16 m = B.cell[idx(r, c)];
            unsigned bits = m;
            if (popcount(bits) == 1)
                std::cout << (ctz(bits) + 1) << ' ';
            else
                std::cout << ". ";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

bool IsSolvedHost(const Board &B) {
    for (int i = 0; i < BOARD_SIZE; ++i)
        if (popcount((unsigned)B.cell[i]) != 1)
            return false;
    return true;
}

// Depth-first search (backtracking) on a single board until a solution is found or exhaustion.
// Returns true and writes solution into out if found. Stops early when found.
bool DfsSolveSingle(Board curBoard, Board &out) {
    // prev, mask, copyMask arrays
    int prev[BOARD_SIZE];
    u16 mask[BOARD_SIZE];
    u16 copyMask[BOARD_SIZE];

    for (int i = 0; i < BOARD_SIZE; ++i) {
        prev[i] = -1;
        mask[i] = 0;
        copyMask[i] = 0;
    }

    for (int i = 0; i < BOARD_SIZE; ++i) {
        if (curBoard.cell[i] == ALL_CANDIDATES) {
            u16 cand = ComputeCandidates(curBoard, i);
            mask[i] = cand;
            copyMask[i] = cand;
            if (!cand) return false;
        }
    }

    int cur = -1;
    for (int i = 0; i < BOARD_SIZE; ++i)
        if (curBoard.cell[i] == ALL_CANDIDATES) { cur = i; break; }

    if (cur == -1) {
        out = curBoard;
        return true;
    }

    while (cur != -1) {
        if (!mask[cur]) {
            curBoard.cell[cur] = ALL_CANDIDATES;
            mask[cur] = copyMask[cur];
            cur = prev[cur];
            continue;
        }

        int c1 = ffs_int(mask[cur]);
        if (c1 == 0) {
            mask[cur] = copyMask[cur];
            curBoard.cell[cur] = ALL_CANDIDATES;
            cur = prev[cur];
            continue;
        }

        int bitIndex = c1 - 1;
        u16 bitMask = (1u << bitIndex);
        mask[cur] &= ~(1u << bitIndex);
        curBoard.cell[cur] = bitMask;

        if (!CheckNeighbors(curBoard, cur)) {
            curBoard.cell[cur] = ALL_CANDIDATES;
            continue;
        }

        int nextCell = -1;
        for (int i = 0; i < BOARD_SIZE; ++i)
            if (curBoard.cell[i] == ALL_CANDIDATES) { nextCell = i; break; }

        if (nextCell == -1) {
            out = curBoard;
            return true;
        }

        prev[nextCell] = cur;
        cur = nextCell;

        if (!copyMask[cur]) {
            u16 cand = ComputeCandidates(curBoard, cur);
            mask[cur] = cand;
            copyMask[cur] = cand;
            if (!cand) { cur = prev[cur]; continue; }
        }
    }

    return false;
}

int main() {
    using clk = std::chrono::high_resolution_clock;
    auto t_start = clk::now();

    // -------- Initialization (read input) ----------
    std::vector<u16> h_input(BOARD_SIZE, 0);
    std::ifstream in("sudoku.txt");
    if (!in) {
        std::cerr << "Cannot open sudoku.txt\n";
        return 1;
    }
    char ch;
    int idx_in = 0;
    while (in >> ch && idx_in < BOARD_SIZE) {
        if (ch >= '0' && ch <= '9') h_input[idx_in++] = ch - '0';
    }
    in.close();

    Board startBoard;
    for (int i = 0; i < BOARD_SIZE; ++i) {
        int v = (i < (int)h_input.size() ? (int)h_input[i] : 0);
        startBoard.cell[i] = (v == 0 ? ALL_CANDIDATES : (1u << (v - 1)));
    }

    auto t_init = clk::now();

    // -------- "Copy / Setup" (prepare frontier) ----------
    // We'll represent frontier as vectors of Boards
    std::vector<Board> frontier;
    frontier.reserve(1024);
    frontier.push_back(startBoard);

    auto t_copy = clk::now();

    // -------- BFS phase (breadth expansion) ----------
    auto t_bfs_start = clk::now();
    int depth = BFS_DEPTH;
    int total_generated = (int)frontier.size();

    for (int d = 0; d < depth; ++d) {
        std::vector<Board> nextFrontier;
        nextFrontier.reserve(frontier.size() * 8 + 16); // heuristic

        for (size_t bi = 0; bi < frontier.size(); ++bi) {
            const Board &board = frontier[bi];

            // find first empty cell and expand
            int cellToExpand = -1;
            for (int i = 0; i < BOARD_SIZE; ++i)
                if (board.cell[i] == ALL_CANDIDATES) { cellToExpand = i; break; }

            if (cellToExpand == -1) {
                // board already solved - we can directly report
                std::cout << "Solved during BFS!\n";
                PrintBoardMasksHost(board);
                auto t_bfs_end = clk::now();
                auto t_end = clk::now();
                std::cout << "Initialization: " << std::chrono::duration<double>(t_init - t_start).count() << "s\n";
                std::cout << "Setup: " << std::chrono::duration<double>(t_copy - t_init).count() << "s\n";
                std::cout << "BFS phase: " << std::chrono::duration<double>(t_bfs_end - t_bfs_start).count() << "s\n";
                std::cout << "DFS phase: 0s\n";
                std::cout << "Total: " << std::chrono::duration<double>(t_end - t_start).count() << "s\n";
                return 0;
            }

            u16 cands = ComputeCandidates(board, cellToExpand);
            if (!cands) continue;

            u16 mask = cands;
            while (mask) {
                int bitIndex = ctz(mask); // ctz gives zero-based
                Board nb = board;
                nb.cell[cellToExpand] = (1u << bitIndex);
                if ((int)nextFrontier.size() < MAX_BOARDS) nextFrontier.push_back(nb);
                // clear lowest set bit
                mask &= (mask - 1);
            }

            if ((int)nextFrontier.size() >= MAX_BOARDS) break;
        }

        total_generated += (int)nextFrontier.size();
        if (nextFrontier.empty()) break;

        // move to next level; but to mimic CUDA code's single big frontier array,
        // append nextFrontier onto frontier (so frontier grows). However to keep memory small
        // we can set frontier = move(nextFrontier)
        frontier = std::move(nextFrontier);

        if ((int)frontier.size() > MAX_BOARDS) frontier.resize(MAX_BOARDS);
    }

    auto t_bfs_end = clk::now();

    // -------- DFS phase: try each board in last frontier until solution found ----------
    auto t_dfs_start = clk::now();

    bool found = false;
    Board solved;
    for (size_t i = 0; i < frontier.size(); ++i) {
        if (DfsSolveSingle(frontier[i], solved)) {
            found = true;
            break;
        }
    }

    auto t_dfs_end = clk::now();
    auto t_end = clk::now();

    std::cout << "Initialization: " << std::chrono::duration<double>(t_init - t_start).count() << "s\n";
    std::cout << "Setup: " << std::chrono::duration<double>(t_copy - t_init).count() << "s\n";
    std::cout << "BFS phase: " << std::chrono::duration<double>(t_bfs_end - t_bfs_start).count() << "s\n";
    std::cout << "DFS phase: " << std::chrono::duration<double>(t_dfs_end - t_bfs_end).count() << "s\n";
    std::cout << "Total: " << std::chrono::duration<double>(t_end - t_start).count() << "s\n";

    if (found && IsSolvedHost(solved)) {
        std::cout << "Solved!\n";
        PrintBoardMasksHost(solved);
    } else {
        std::cout << "No solution.\n";
    }

    return 0;
}
