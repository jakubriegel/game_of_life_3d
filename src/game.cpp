#include "../inc/game.hpp"

__device__ __host__
inline unsigned cell_index(const cell & c) { return cell_index(c.x, c.y, c.z); }

__device__ __host__
unsigned cell_index(const unsigned & x, const unsigned & y, const unsigned & z) {
    const unsigned z_index = ARENA_DIM * ARENA_DIM * z;
    const unsigned y_index = ARENA_DIM * y;
    return z_index + y_index + x;
}

__device__ __host__
inline void revive_cell(const cell & c, bool * const arena) { arena[cell_index(c)] = true; }

__device__ __host__
inline void kill_cell(const cell & c, bool * const arena) { arena[cell_index(c)] = false; }

__device__ __host__
unsigned count_neighbours(const cell & c, const bool * const old) {
    unsigned neighbours = 0;

    auto it = [](const unsigned & i) -> unsigned { return i > 0 ? i-1 : 0; };
    for (unsigned z = it(c.z); z <= c.z+1; z++) if (z < ARENA_DIM)
        for (unsigned x = it(c.x); x <= c.x+1; x++) if (x < ARENA_DIM)
            for (unsigned y = it(c.y); y <= c.y+1; y++) if(y < ARENA_DIM)
                if (!(c.x == x && c.y == y && c.z == z))
                    if (old[cell_index(x, y, z)]) neighbours++;

    return neighbours;
}

__device__ __host__
void mature_cell(
    const cell & c,
    bool * const arena, const bool * const old) 
{
    const auto neighbours = count_neighbours(c, old);
    const auto index = cell_index(c);
    if (old[index]) { 
        if (neighbours < KILL_LOW || neighbours > KILL_HIGH) kill_cell(c, arena);
        else revive_cell(c, arena);
    }
    else {
        if (neighbours >= REVIVE_LOW && neighbours <= REVIVE_HIGH) revive_cell(c, arena);
        else kill_cell(c, arena);
    }
}

__global__
void mature_cells(bool * const arena, const bool * const old) {
    const unsigned y = threadIdx.x;
    const unsigned z = blockIdx.x;
    
    for (unsigned x = 0; x < ARENA_DIM; x++) {
        const cell c{x, y, z};
        mature_cell(c, arena, old);
    }
}

game::game() {
    init_arena();
    init_cells();
}

game::~game() { cudaFree(arena); };

void game::init_arena() {
    const auto full_size = ARENA_DIM * ARENA_DIM * ARENA_DIM;
    const auto arena_size = full_size * sizeof(bool);
    cudaMallocManaged(&arena, arena_size);
    cudaMallocManaged(&old_arena, arena_size);
}

void game::switch_arena() {
    bool * const temp = arena;
    arena = old_arena;
    old_arena = temp;
}

void game::init_cells() {
    const auto center = ARENA_DIM / 2;
    const cell start[7] = {
        {0 + center, 0 + center, 5}, 
        {1 + center, 0 + center, 5},
        {0 + center, 1 + center, 5}, 
        {-1 + center, 0 + center, 5},
        {0 + center, -1 + center, 5},
        {0 + center, 0 + center, 6}, 
        {0 + center, 0 + center, 4}
    };

    for (const cell & c : start) revive_cell(c, arena);
}

void game::next_generation() {
    switch_arena();
    mature_cells<<<ARENA_DIM, ARENA_DIM>>>(arena, old_arena);
    cudaDeviceSynchronize();
    generation++;
}

void game::print_arena() const {
    printf("generation: %d:\n\n", generation);
    for (unsigned z = 0; z < ARENA_DIM; z++) {
        printf("%d\n", z);
        for (unsigned x = 0; x < ARENA_DIM; x++) {
            for (unsigned y = 0; y < ARENA_DIM; y++) printf("%d ", arena[cell_index(x, y, z)]);
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}
