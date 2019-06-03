#include "../inc/game.hpp"

game::game() {
    init_arena();
    init_kernel_gen();
    init_cells();

    cuda_init();
}

game::~game() { 
    cudaFree(even_arena); 
    cudaFree(odd_arena);
};

void game::init_arena() {
    const auto arena_size = game::arena_size();
    cudaMallocManaged(&even_arena, arena_size);
    cudaMallocManaged(&odd_arena, arena_size);
}

void game::switch_arena() {
    bool * const temp = even_arena;
    even_arena = odd_arena;
    odd_arena = temp;
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

    for (const cell & c : start) revive_cell(c, odd_arena);
}

void game::init_kernel_gen() { cudaMallocManaged(&kernel_gen, game::kernel_number() * sizeof(unsigned)); }

void game::start(const unsigned & n) {
    // fflush(stdout);
    
    cuda_game(n, even_arena, odd_arena, kernel_gen);
    cuda_finalize();
    
    generation += n;
}

void game::print_arena() const {
    const bool * const arena = ([&]() {
        if (generation % 2 == 0) return odd_arena;
        else return even_arena;
    })();

    printf("generation: %d\n\n", generation);
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

void game::print_kernel_gen() const {
    for (unsigned i = 0; i < game::kernel_number(); i++) {
        printf("%d ", kernel_gen[i]);
    }
    printf("\n");
}

unsigned game::full_size() { return ARENA_DIM * ARENA_DIM * ARENA_DIM; }
unsigned game::arena_size() { return game::full_size() * sizeof(bool); }
unsigned game::kernel_number() { return ARENA_DIM * ARENA_DIM; }
