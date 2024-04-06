/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Memory that is shared among all ranks on a single node. Must be enabled with
 * the compilation flags -DUSE_MPI and -DUSE_SHARED_MEM (see :ref:`Compilation`).
 * reasonably recent linux kernel + headers required (memfd); as well as the
 * uthash.h library somewhere in your CPATH.
 */

// init/deinit functions called by diverge internally, do not use (no symbols)
void shared_malloc_init( int more_colors );
void shared_malloc_finalize( void );

/**
 * allocate rank shared memory on a single node. signature equivalent to
 * stdlib's malloc
 */
void* shared_malloc( long long sz );

/**
 * same as :c:func:`shared_malloc` but zero out the memory. signature equivalent
 * to stdlib's calloc
 */
void* shared_calloc( long long cnt, long long sz );

/**
 * free memory allocated with :c:func:`shared_malloc` or
 * :c:func:`shared_calloc`. signature equivalent to stdlib's free
 */
void shared_free( void* ptr );

/**
 * enter a region that is exlusively executed by one rank per node. Usage:
 *
 * .. sourcecode:: C
 *
 *  if (shared_exclusive_enter(ptr) == 1) {
 *    // do something exclusively
 *  }
 *  shared_exclusive_wait(ptr);
 */
int shared_exclusive_enter( void* ptr );

/**
 * wait for an operation entered via :c:func:`shared_exclusive_enter` to finish
 * on all ranks.
 */
void shared_exclusive_wait( void* ptr );

/**
 * return the rank on one node. useful for worksharing when operating on shared
 * memory.
 */
int shared_malloc_rank( void );
/**
 * return the number of ranks on one node. useful for worksharing when operating
 * on shared memory.
 */
int shared_malloc_size( void );
/**
 * barrier that must be reached by each rank on one node before execution
 * continues
 */
void shared_malloc_barrier( void );

#ifdef __cplusplus
}
#endif

