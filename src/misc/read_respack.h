/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rs_vertex_t rs_vertex_t;

/**
 * read vertices in RESPACK format; both U and J. Note that frequencies are
 * *not* taken care of, and also the detection whether the file is J or W (i.e.,
 * channels). Spin is not accounted for.
 *
 * You can pass :c:macro:`DIVERGE_W90_READER_SWAP_ORBITALS` in ``*len`` to swap
 * the orbital convention.
 *
 * .. note:: The orbital convention might be wrong. It is hard to check because
 *           RESPACK does not preserve symmetries either way.
 */
rs_vertex_t* diverge_read_respack_vertices( const char* filename, index_t* len );

#ifdef __cplusplus
}
#endif
