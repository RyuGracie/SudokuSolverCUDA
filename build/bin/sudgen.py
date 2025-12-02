import random
import copy

SIZE = 9
BOX = 3

def find_empty(g):
    for r in range(SIZE):
        for c in range(SIZE):
            if g[r][c] == 0:
                return r, c
    return None

def is_valid(g, r, c, val):
    if any(g[r][j] == val for j in range(SIZE)):
        return False
    if any(g[i][c] == val for i in range(SIZE)):
        return False
    br = (r // BOX) * BOX
    bc = (c // BOX) * BOX
    for i in range(br, br + BOX):
        for j in range(bc, bc + BOX):
            if g[i][j] == val:
                return False
    return True

def solve_backtrack(g, limit_solutions=None):
    empty = find_empty(g)
    if not empty:
        return 1 if limit_solutions else True
    r, c = empty
    solutions = 0
    for val in range(1, 10):
        if is_valid(g, r, c, val):
            g[r][c] = val
            res = solve_backtrack(g, limit_solutions)
            if limit_solutions is None:
                if res:
                    return True
            else:
                solutions += res
                if solutions >= limit_solutions:
                    g[r][c] = 0
                    return solutions
            g[r][c] = 0
    return solutions if limit_solutions else False

def generate_full_grid():
    grid = [[0]*9 for _ in range(9)]
    nums = list(range(1, 10))

    def fill():
        empty = find_empty(grid)
        if not empty:
            return True
        r, c = empty
        random.shuffle(nums)
        for v in nums:
            if is_valid(grid, r, c, v):
                grid[r][c] = v
                if fill():
                    return True
                grid[r][c] = 0
        return False

    fill()
    return grid

def make_puzzle(full_grid, clues=30):
    puzzle = copy.deepcopy(full_grid)
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)

    removed = 0
    to_remove = 81 - clues

    for r, c in cells:
        if removed >= to_remove:
            break
        backup = puzzle[r][c]
        puzzle[r][c] = 0

        grid_copy = copy.deepcopy(puzzle)
        if solve_backtrack(grid_copy, limit_solutions=2) != 1:
            puzzle[r][c] = backup  # restore if not unique
        else:
            removed += 1

    return puzzle

def save_to_txt(grid, filename="sudoku.txt"):
    with open(filename, "w") as f:
        for row in grid:
            f.write("".join(str(x) for x in row) + "\n")

if __name__ == "__main__":
    random.seed()

    full = generate_full_grid()
    puzzle = make_puzzle(full, clues=15)

    save_to_txt(puzzle)  # creates sudoku.txt
    print("Sudoku puzzle saved to sudoku.txt")
