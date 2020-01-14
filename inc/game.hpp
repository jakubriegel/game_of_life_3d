#pragma once

#include "project.hpp"
#include "game_cuda.hpp"

#define ARENA_DIM 75

#define KILL_LOW 2
#define KILL_HIGH 7
#define REVIVE_LOW 3
#define REVIVE_HIGH 6

class game {
    private:
        unsigned generation = 0;

        void init_arena();
        void switch_arena();

        void init_cells();
        void init_kernel_gen();

    public:
        bool * even_arena;
        bool * odd_arena;
        unsigned * kernel_gen;
        game();
        ~game();

        void start(const unsigned & n);

        void print_arena() const;
        void print_kernel_gen() const;

        static unsigned full_size();
        static unsigned arena_size();
        static unsigned kernel_number();
};
