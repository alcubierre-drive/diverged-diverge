/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"
#include "../misc/mpi_functions.h"

#ifndef MAGIC_NUMBER_POST_PROCESSING
#define MAGIC_NUMBER_POST_PROCESSING 'P'
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct grid_loop_t grid_loop_t;
typedef struct grid_vertex_t grid_vertex_t;
typedef struct grid_postproc_t grid_postproc_t;

grid_postproc_t* grid_postproc_init( grid_vertex_t* V, grid_loop_t* L );
void grid_postproc_free( grid_postproc_t* p );

void grid_postproc_susc( grid_postproc_t* p, double T, char chan );
void grid_postproc_lingap( grid_postproc_t* p, double T, char chan, const char* vertex_file );

void grid_postproc_set_num_singular_values( grid_postproc_t* p, uindex_t n );
void grid_postproc_set_loop_usage( grid_postproc_t* p, bool loop );

// only call communicate after all calculations!
void grid_postproc_communicate( grid_postproc_t* p );

/* header is index_t[64], and all displacement/size information is in
 * units of 64bit (i.e. double); must be freed by user
 *
 * header[0] - MAGIC_NUMBER_POST_PROCESSING
 * header[60-61] - file format version string [displ, size]. if both are
 *                 zero, default to v0.4 for diverge.output.read().
 *
 * header[1] - nk
 * header[2] - nb
 * header[3] - nff
 * header[4] - SU2
 * header[5-6] - formfactors [displ, size]
 * header[7-8] - susc P [displ, size]
 * header[9-10] - lin gap P [displ, size]
 * header[11-12] - susc C [displ, size]
 * header[13-14] - lin gap C [displ, size]
 * header[15-16] - susc D [displ, size]
 * header[17-18] - lin gap D [displ, size]
 * header[19] - lin gap num values/vectors
 * header[20] - lin gap matrix size (one leg, i.e., nk*nb*nb)
 *
 * header[62] - MAGIC_NUMBER_POST_PROCESSING
 * header[63] - total file size (in 64bit units)
 */
const index_t* grid_postproc_to_file( grid_postproc_t* p, const char* fname );
void grid_postproc_print_header( const index_t* header );

#ifdef __cplusplus
}
#endif
