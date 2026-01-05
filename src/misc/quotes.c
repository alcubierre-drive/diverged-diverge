/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include "quotes.h"

static int has_srand_setup = 0;
static void do_srand_setup( void );
static int random_index( int max );

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
};

const char* random_quote( void ) {
    return quotes[random_index( sizeof(quotes)/sizeof(quotes[0]) )];
}

static int random_index( int max ) {
    do_srand_setup();
    return rand() % max;
}

static void do_srand_setup( void ) {
    if (has_srand_setup) return;
    has_srand_setup = 1;
    time_t currentTime = time(NULL);
    unsigned char *pByte = (unsigned char *)&currentTime;
    unsigned seed = 0;
    for (size_t index = 0; index < sizeof(currentTime); index++) {
        seed = seed * (UCHAR_MAX + 2U) + pByte[index];
    }
    srand(seed);
}

