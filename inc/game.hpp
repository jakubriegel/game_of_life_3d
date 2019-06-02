#pragma once

#include "../inc/project.hpp"

#define ARENA_DIM 100

#define KILL_LOW 2
#define KILL_HIGH 7
#define REVIVE_LOW 3
#define REVIVE_HIGH 6

struct cell {
    unsigned x;
    unsigned y;
    unsigned z;
};

__device__ __host__
unsigned cell_index(const cell & c);

__device__ __host__
unsigned cell_index(const unsigned & x, const unsigned & y, const unsigned & z);


__device__ __host__
void revive_cell(const cell & c);

__device__ __host__
void kill_cell(const cell & c);

__device__ __host__
unsigned count_neighbours(const cell & c);

__device__ __host__
void mature_cell(
    const cell & c,
    bool * const arena, const bool * const old
);

__global__
void mature_cells(bool * const arena, const bool * const old);

class game {
    private:
        unsigned generation = 0;

        void init_arena();
        void switch_arena();

        void init_cells();

    public:
        bool * arena;
        bool * old_arena;

        game();
        ~game();

        void next_generation();
        void print_arena() const;
};
