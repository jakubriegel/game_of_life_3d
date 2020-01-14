#include "inc/project.hpp"
#include "inc/game.hpp"

int main(int argc, char **argv) {
    const unsigned generations = std::atoi(argv[1]);
    
    auto g = game();    
    g.start(generations);
    // g.print_arena(); 
    
    return 0;
}

