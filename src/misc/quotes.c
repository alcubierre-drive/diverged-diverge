/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "quotes.h"
#include "mpi_functions.h"

static unsigned random_index( unsigned max ) {
    unsigned num = 0;
    FILE* f = fopen("/dev/random", "rb");
    if (f) {
        fread(&num, sizeof(num), 1, f);
        fclose(f);
    } else {
        num = (uint64_t)(diverge_mpi_wtime()*1.e9);
    }
    return num % max;
}

static const char quotes[][1024] = {
    "One of my most productive days was throwing away 1,000 lines of code.\n"
    "                                                 -- Ken Thompson\n",
    "When in doubt, use brute force.\n"
    "                                                 -- Ken Thompson\n",
    "FORTRAN was the language of choice for the same reason that three-legged\n"
    "races are popular.\n"
    "                                                 -- Ken Thompson\n",
    "If you want to go somewhere, goto is the best way to get there.\n"
    "                                                 -- Ken Thompson\n",
    "I just don't know what to do with myself.\n"
    "                                            -- The White Stripes\n",
};

const char* diverge_random_quote( void ) { return quotes[random_index(sizeof(quotes)/sizeof(quotes[0]))]; }

