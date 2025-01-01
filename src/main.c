/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge.h"

// Namespace: main
// File that is compiled to an executable when making the tests via
// === Code ===
// make test
// ============

int main( int argc, char** argv ) {
    diverge_init( &argc, &argv );
    diverge_compilation_status();
    int r = diverge_run_tests( argc, argv );
    diverge_finalize();
    return r;
}
