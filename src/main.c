/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge.h"

int main( int argc, char** argv ) {
    diverge_init( &argc, &argv );
    int r = diverge_run_tests( argc, argv );
    diverge_finalize();
    return r;
}
