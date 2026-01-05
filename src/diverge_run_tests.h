/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * run the tests, including control over which tests to run etc. via command
 * line arguments. Usually, a call would look like
 *
 * .. sourcecode:: C
 *
 *  #include <diverge.h>
 *
 *  int main( int argc, char** argv ) {
 *      diverge_init( &argc, &argv );
 *      diverge_compilation_status();
 *      int r = diverge_run_tests( argc, argv );
 *      diverge_finalize();
 *      return r;
 *  }
 *
 * which is exactly what is done in ``src/main.c``, i.e., the file that
 * is compiled to the divERGe executable when making the 'test' target.
 *
 * :param int argc: number of command line arguments
 * :param char** argv: command line argument vector
 */
int diverge_run_tests( int argc, char** argv );

#ifdef __cplusplus
}
#endif
