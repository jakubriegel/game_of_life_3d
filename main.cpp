#include "inc/project.hpp"
#include "inc/game.hpp"

int main(void)
{
    printf("it works");
    auto g = game();
    printf("\n\n");
    for (long long i = 0; i < 10; i++) {
        g.next_generation();
        cudaDeviceSynchronize();
        g.print_arena();
        printf("\n\n");
    }
    
    // g.next_generation();
    // g.print_arena();
    // printf("\n\n");
    // g.next_generation();
    // g.print_arena();
    // game<<<5, 100>>>(ARENA_DIM, arena);

    cudaDeviceSynchronize();
    
    return 0;
}

