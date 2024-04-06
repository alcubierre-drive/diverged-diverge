/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/** prints compilation status to stderr */
void diverge_compilation_status( void );

/** return 1 if MPI was compiled in, else 0 */
int diverge_compilation_status_mpi( void );

/**
 * return 1 if CUDA was compiled in, else 0. output the CUDA version string to
 * str. ignore str if NULL (None)
 */
int diverge_compilation_status_cuda( char* str );

/**
 * return 1 if we're on a clean git branch, else 0. output the git version
 * string to str. ignore str if NULL (None)
 */
int diverge_compilation_status_version( char* str );

/** return 1 if number representations follow IEC and endianness is little, else 0 */
int diverge_compilation_status_numbers( void );

#ifdef __cplusplus
}
#endif
