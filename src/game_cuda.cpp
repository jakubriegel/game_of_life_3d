#include "../inc/game_cuda.hpp"

__device__ __host__
inline unsigned cell_index(const cell & c) { return cell_index(c.x, c.y, c.z); }

__device__ __host__
unsigned cell_index(const unsigned & x, const unsigned & y, const unsigned & z) {
    const unsigned z_index = ARENA_DIM * ARENA_DIM * z;
    const unsigned y_index = ARENA_DIM * y;
    return z_index + y_index + x;
}

__device__ __host__
void revive_cell(const unsigned & i, bool * const arena) { arena[i] = true; };

__device__ __host__
void revive_cell(const cell & c, bool * const arena) { revive_cell(cell_index(c), arena); };

__device__ __host__
void kill_cell(const unsigned & i, bool * const arena) { arena[i] = false; };

__device__ __host__
void kill_cell(const cell & c, bool * const arena) { kill_cell(cell_index(c), arena); };

__device__ __host__
void wait_for_others(const unsigned * const my_gen, const unsigned * const kernel_gen) {
    bool con = true;
    do { 
        con = true;
        for(unsigned i = 0; i < ARENA_DIM * ARENA_DIM; i++) if (kernel_gen[i] < *my_gen) con = false; 
    } while(!con);
}

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
inline void mature_cell(
    const cell & c,
    bool * const arena, const bool * const old
) 
{
    const auto neighbours = count_neighbours(c, old);
    const auto index = cell_index(c);
    if (old[index]) { 
        if (neighbours < KILL_LOW || neighbours > KILL_HIGH) kill_cell(index, arena);
        else revive_cell(index, arena);
    }
    else {
        if (neighbours >= REVIVE_LOW && neighbours <= REVIVE_HIGH) revive_cell(index, arena);
        else kill_cell(index, arena);
    }
}

__device__ __host__
void row_next_gen(
    const bool & even_gen, const unsigned & y, const unsigned & z, 
    bool * const even_arena, bool * const odd_arena
) {
    for (unsigned x = 0; x < ARENA_DIM; x++) {
        const cell c{x, y, z};
        if (even_gen) mature_cell(c, even_arena, odd_arena);
        else mature_cell(c, odd_arena, even_arena);
    }
}

__device__ __host__
void mature_cells(
    unsigned * const my_gen, const unsigned * const kernel_gen,
    const unsigned & y, const unsigned & z, 
    bool * const even_arena, bool * const odd_arena
) {
    wait_for_others(my_gen, kernel_gen);
    row_next_gen(*my_gen % 2 == 0, y, z, even_arena, odd_arena);
}

// __device__
// void p(const bool * const a) {
//      // if (y == 0 && z == 0) {
//     //     if (kernel_gen[index] % 2 == 0) p(arena);
//     //     else p(old);
//     // }
//     for (unsigned z = 0; z < ARENA_DIM; z++) {
//             printf("%d\n", z);
//             for (unsigned x = 0; x < ARENA_DIM; x++) {
//                 for (unsigned y = 0; y < ARENA_DIM; y++) printf("%d ", a[cell_index(x, y, z)]);
//                 printf("\n");
//             }
//             printf("\n");
//         }
//         printf("\n");
// }

__global__
void mature_cells_kernel(const unsigned n, bool * const even_arena, bool * const odd_arena, unsigned * const kernel_gen) {
    const unsigned y = threadIdx.x;
    const unsigned z = blockIdx.x;

    const auto index = ARENA_DIM * z + y;
    unsigned * const my_gen = &kernel_gen[index];
    
    while (*my_gen < n) {
        mature_cells(my_gen, kernel_gen, y, z, even_arena, odd_arena);
        atomicAdd(my_gen, (unsigned) 1);
    }
}

void cuda_init() {
    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, CUDA_PRINTF_BUFFER_SIZE);
}

void cuda_game(const unsigned & n, bool * const arena, bool * const old, unsigned * const kernel_gen) {
    mature_cells_kernel<<<ARENA_DIM, ARENA_DIM>>>(n, arena, old, kernel_gen);
}

void cuda_finalize() { 
    cudaDeviceSynchronize(); 
    // cudaProfilerStop();
}