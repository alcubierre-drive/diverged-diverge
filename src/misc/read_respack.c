/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "read_respack.h"
#include "../diverge_model.h"
#include "text_io.h"

#include "read_W90.h"

#define LINALG_NO_LAPACKE
#include "linalg.h"
LINALG_IMPLEMENT


#define VRB_ASSERT( cond ) { \
    int err = !(cond); \
    assert( err == 0 ); \
    if (err) { mpi_err_printf( "%s\n", #cond ); } \
}

rs_vertex_t* diverge_read_respack_vertices( const char* filename, index_t* len ) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        mpi_err_printf("failed to open data file '%s'\n", filename);
        return NULL;
    }

    int swap_orbitals = 0;
    if (len && *len == DIVERGE_W90_READER_SWAP_ORBITALS) {
        mpi_log_printf( "read_respack: swapping orbital order\n" );
        swap_orbitals = 1;
    }

    char line[DIVERGE_GETLINE_MAX] = {0};
    char line_cpy[DIVERGE_GETLINE_MAX] = {0};
    int nread = 0;

    int R[3] = {0};
    int line_num = 0;
    VECTOR_INIT(rs_vertex_t, verts);
    while (diverge_getline(line, f) != -1) {
        line_num++;
        diverge_chomp_inplace(line);
        if (line[0] == '#') continue; // comments are skipped

        int is_R_line = 0;
        int R_test[3] = {0};
        nread = sscanf(line, "%i %i %i", &R_test[0], &R_test[1], &R_test[2]);
        if (nread == 3) {
            sprintf(line_cpy, "%i %i %i", R_test[0], R_test[1], R_test[2]);
            if (!strcmp(line_cpy, line)) {
                R[0] = R_test[0]; R[1] = R_test[1]; R[2] = R_test[2];
                is_R_line = 1;
            }
        }
        if (!is_R_line) {
            rs_vertex_t V = {.chan='R'};
            nread = sscanf(line, "%li %li %lf %lf", &V.o1, &V.o2, ((double*)&V.V), ((double*)&V.V)+1);
            V.o1 -= 1; V.o2 -= 1; // one-based indexing
            if (nread == 4) {
                V.R[0] = R[0]; V.R[1] = R[1]; V.R[2] = R[2];
                if (swap_orbitals) XORSWAP(V.o1, V.o2);
                VECTOR_PUSH_BACK(verts, V);
            } else {
                if (strcmp(line, ""))
                    mpi_wrn_printf( "non-empty line no.%i ('%s') not parsed\n", line_num, line );
            }
        }
    }
    VECTOR_PUSH_BACK(verts, (rs_vertex_t){0});
    VECTOR_SHRINK_TO_FIT(verts);
    if (len) *len = verts_sz-1;

    fclose(f);
    return verts;
}
