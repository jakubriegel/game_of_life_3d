#pragma once

#include "../inc/project.hpp"

#define ARENA_DIM 10

#define KILL_LOW 2
#define KILL_HIGH 3
#define REVIVE 3

__device__ __host__
unsigned int cell_index(const unsigned int & x, const unsigned int & y);

__device__ __host__
void revive_cell(const unsigned int & x, const unsigned int & y, bool * const arena);

__device__ __host__
void kill_cell(const unsigned int & x, const unsigned int & y, bool * const arena);

__device__ __host__
unsigned int count_neighbours(const unsigned int & x, const unsigned int & y, const bool * const old);

__device__ __host__
void mature_cell(const unsigned int & x, const unsigned int & y, bool * const arena, const bool * const old);

__global__
void mature_cells(bool * const arena, const bool * const old);

class game {
    private:
        unsigned int generation = 0;

        void new_arena();
        void init_cells();

    public:
        bool * arena;

        game();
        ~game();

        void next_generation();
        void print_arena();
};
