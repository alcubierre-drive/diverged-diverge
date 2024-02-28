/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Function: diverge_compilation_status
void diverge_compilation_status( void );

// Function: diverge_compilation_status_mpi
// return 1 if MPI was compiled in, else 0
int diverge_compilation_status_mpi( void );

// Function: diverge_compilation_status_cuda
// return 1 if CUDA was compiled in, else 0. output the CUDA version string to
// str. ignore str if NULL (None)
int diverge_compilation_status_cuda( char* str );

// Function: diverge_compilation_status_version
// return 1 if we're on a clean git branch, else 0. output the git version
// string to str. ignore str if NULL (None)
int diverge_compilation_status_version( char* str );

// Function: diverge_compilation_status_numbers
// return 1 if number representations follow IEC and endianness is little, else
// 0
int diverge_compilation_status_numbers( void );

#ifdef __cplusplus
}
#endif
