/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "misc/compilation_status.h"
#include "misc/mpi_functions.h"
#include "diverge_run_tests.h"

int main( int argc, char** argv ) {

    #if INC_TEST
    diverge_init( &argc, &argv );
    mpi_usr_printf( "Wer testet hat's nötig...\n" );
    int retv = diverge_run_tests( argc, argv );
    diverge_finalize();
    return retv;
    #endif

    diverge_init( &argc, &argv );
    diverge_compilation_status();
    mpi_usr_printf( "I just don't know what to do with myself. -- White Stripes\n" );
    diverge_finalize();
    return 42;
}
