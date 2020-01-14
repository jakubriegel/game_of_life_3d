#pragma once

#include "project.hpp"
#include "game.hpp"

#define CUDA_PRINTF_BUFFER_SIZE 2000000

__device__ __host__
unsigned cell_index(const cell & c);

__device__ __host__
unsigned cell_index(const unsigned & x, const unsigned & y, const unsigned & z);

__device__ __host__
void revive_cell(const unsigned & i, bool * const arena);

__device__ __host__
void revive_cell(const cell & c, bool * const arena);

__device__ __host__
void kill_cell(const unsigned & i, bool * const arena);

__device__ __host__
void kill_cell(const cell & c, bool * const arena);

__global__
void mature_cells_kernel(const unsigned n, bool * const even_arena, bool * const odd_arena, unsigned * const kernel_gen);

void cuda_init();

void cuda_game(const unsigned & n, bool * const arena, bool * const old, unsigned * const kernel_gen);

void cuda_finalize();
