#include "../inc/game.hpp"

__device__ __host__
unsigned int cell_index(const unsigned int & x, const unsigned int & y) {
    return ARENA_DIM * y + x;
}

__device__ __host__
void revive_cell(const unsigned int & x, const unsigned int & y, bool * const arena) { 
    arena[cell_index(x, y)] = true;
}

__device__ __host__
void kill_cell(const unsigned int & x, const unsigned int & y, bool * const arena) { 
    arena[cell_index(x, y)] = false;
}

__device__ __host__
unsigned int count_neighbours(const unsigned int & x, const unsigned int & y, const bool * const old) {
    unsigned int neighbours = 0;
    for (int i = x-1; i <= x+1; i++) if(i > 0 && i < ARENA_DIM) 
        for (unsigned int j = y-1; j <= y+1; j++) if(j > 0 && j < ARENA_DIM)
            if (!(x == i && y == j)) if (old[cell_index(i, j)]) neighbours++;
    return neighbours;
}

__device__ __host__
void mature_cell(const unsigned int & x, const unsigned int & y, bool * const arena, const bool * const old) {
    const auto neighbours = count_neighbours(x, y, old);
    const auto index = cell_index(x, y);
    if (old[index]) { 
        if (neighbours < KILL_LOW || neighbours > KILL_HIGH) kill_cell(x, y, arena);
    }
    else {
        if (neighbours == REVIVE) revive_cell(x, y, arena);
    }
}

__global__
void mature_cells(bool * const arena, const bool * const old) {
    const int index = threadIdx.x;
    const int block = blockIdx.x;
    const int dim = blockDim.x;
    // blockI

    for (unsigned int x = index; x < ARENA_DIM; x += dim) {
        for (unsigned int y = 0; y < ARENA_DIM; y++) 
            mature_cell(x, y, arena, old);
    }
}

game::game() {
    new_arena();
    init_cells();
    print_arena();
}

game::~game() { cudaFree(arena); };

void game::new_arena() {
    const auto full_size = ARENA_DIM * ARENA_DIM;
    cudaMallocManaged(&arena, full_size*sizeof(bool));
    for (int i = 0; i < ARENA_DIM; i++) arena[i] = false;
}

void game::init_cells() {
    const auto center = ARENA_DIM / 2;
    revive_cell(0 + center, 1 + center, arena);
    revive_cell(1 + center, 2 + center, arena);
    revive_cell(2 + center, 0 + center, arena);
    revive_cell(2 + center, 1 + center, arena);
    revive_cell(2 + center, 2 + center, arena);
}

void game::next_generation() {
    bool * old = arena;
    new_arena();
    for(unsigned int i = 0; i < ARENA_DIM * ARENA_DIM; i++) if (old[i]) arena[i] = true;
    
    mature_cells<<<1, 1>>>(arena, old);
    cudaDeviceSynchronize();
    
    generation++;
}

void game::print_arena() {
    for (unsigned int x = 0; x < ARENA_DIM; x++) {
        for (unsigned int y = 0; y < ARENA_DIM; y++) 
            printf("%d ", arena[cell_index(x, y)]);
        printf("\n");
    }
}

// const unsigned int ARENA_DIM = 10;
// const unsigned int KILL_LOW = 2;
// const unsigned int KILL_HIGH = 3;
// const unsigned int REVIVE = 3;
