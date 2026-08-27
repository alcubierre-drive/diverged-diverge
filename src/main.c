/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge.h"
#include "diverge_constants.h"

/** File that is compiled to an executable when using ``make test`` */

int main( int argc, char** argv ) {
    if (argc > 1 && !strcmp(argv[1], "-printconstants")) {
        diverge_constants_print(0);
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "-printheader")) {
        diverge_constants_print(1);
        return 0;
    }
    diverge_init( &argc, &argv );
    diverge_compilation_status();
    int r = diverge_run_tests( argc, argv );
    diverge_finalize();
    return r;
}
