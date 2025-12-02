# SudokuSolverCUDA

CUDA‑accelerated Sudoku solver (with a CPU version for comparison) — currently supports 9×9 puzzles.

## 🚀 What is this

SudokuSolverCUDA is a solver for standard 9×9 Sudoku boards. It uses NVIDIA CUDA to leverage GPU parallelism for solving Sudoku puzzles — and includes a CPU version so you can compare performance.

## Features

* ✅ Solve 9×9 Sudoku boards using GPU (CUDA).
* ✅ CPU implementation alongside CUDA version (for benchmarking / fallback).
* ✅ Built using C++ and CUDA.
* ✅ Cross‑platform build via CMake.
* ✅ Easy to set up and run (see Build & Run instructions below).

## Requirements

* NVIDIA GPU with CUDA support
* CUDA Toolkit (matching your GPU / driver)
* C++ compiler (supporting C++11 or later)
* CMake
* (Optionally) Python — for helper board creator

## Project structure

```
/SudokuSolverCUDA
  |-- src/               # Source code (C++, CUDA)  
  |-- build/             # Build artifacts / output directory  
  |-- CMakeLists.txt     # Build configuration  
  |-- .gitignore         
  |-- (optional) Python scripts / helpers  
```

## Build & Run

```bash
cd build
cmake ..
make
```

This should compile both the CPU and CUDA versions of the solver.

To run:

```bash
# Example: run the CUDA solver
./bin/main_gpu
```

You can also run the CPU version for comparison.


```bash
./bin/main_cpu
```

## Usage

* The solver expects a 9×9 Sudoku board.
* After execution, the solver prints the solved board (if solvable) or indicates if no solution exists.

## Future / To‑do

* ➕ Support arbitrary Sudoku sizes (e.g. 16×16, or other variants)
