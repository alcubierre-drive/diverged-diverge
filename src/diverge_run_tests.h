/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Function: diverge_run_tests
// run the tests, including control over which tests to run etc. via command
// line arguments. Usually, a call would look like
// --- C/C++ ---
// #include <diverge.h>
// 
// int main( int argc, char** argv ) {
//     diverge_init( &argc, &argv );
//     diverge_compilation_status();
//     int r = diverge_run_tests( argc, argv );
//     diverge_finalize();
//     return r;
// }
// -------------
// which is exactly what is done in <src/main.c at main>, i.e., the file that
// is compiled to the divERGe executable when making the 'test' target.
//
// Parameters:
// argc - number of command line arguments
// argv - command line argument vector
int diverge_run_tests( int argc, char** argv );

#ifdef __cplusplus
}
#endif
