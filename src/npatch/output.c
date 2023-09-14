/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "output.h"
#include "vertex.h"
#include "loop.h"
#include "../misc/mpi_functions.h"

#define VERTEX_FILE_MAGIC_NUMBER 'V'

int npatch_output_post( npatch_vertex_t* v, npatch_loop_t* l, const npatch_output_post_config_t* ocfg, const char* fname ) {

    if (diverge_mpi_comm_rank() != 0) return 0;

    FILE* f = fopen(fname, "w");
    if (!f) {
        mpi_err_printf("could not open file %s. aborting\n", fname );
        return 1;
    }

    index_t header[128];
    memset( header, 0, sizeof(header) );

    int i=0;
    header[i++] = VERTEX_FILE_MAGIC_NUMBER;
    // v->V
    header[i] = 128; i++;
    header[i++] = ocfg->V ? v->size*2 : 0;
    // l->Lp
    header[i] = header[i-1] + header[i-2]; i++;
    header[i++] = ocfg->Lp ? v->size*2 : 0;
    // l->Lm
    header[i] = header[i-1] + header[i-2]; i++;
    header[i++] = ocfg->Lm ? v->size*2 : 0;

    // v->dV
    header[i] = header[i-1] + header[i-2]; i++;
    header[i++] = ocfg->dV ? v->size*2 : 0;

    // prepare the vertices in q matrix format
    uindex_t* qmat_mem[3];
    index_t qmat_mem_sz[3];
    index_t qmat_nq[3];
    char chans[3] = {'P', 'C', 'D'};

    if (ocfg->q_matrices) {
        mpi_vrb_printf("generating q matrices…\n");
        double time_per_chan[3];
        for (int c=0; c<3; ++c) {
            double tick = diverge_mpi_wtime();
            index_t nq, nq_sorted;

            npatch_qvert_mat_t* qmat = npatch_qvert_mat_from_vertex( v, l, chans[c],
                    ocfg->q_matrices_use_dV, ocfg->q_matrices_max_rel, &nq, &nq_sorted );

            npatch_qvert_mat_eigen( qmat, nq_sorted, ocfg->q_matrices_nv, ocfg->q_matrices_eigen_which );

            uindex_t* mem = NULL;
            index_t mem_sz = 0;

            for (index_t q=0; q<nq_sorted; ++q) {
                index_t size;
                uindex_t* qmem = npatch_qvert_mat_serialize( qmat+q, &size );
                mem_sz += size;
                mem = realloc( mem, sizeof(uindex_t) * mem_sz );
                memcpy( mem + mem_sz - size, qmem, sizeof(uindex_t) * size );
                free(qmem);
            }
            npatch_qvert_mat_free( qmat, nq );

            qmat_mem[c] = mem;
            qmat_mem_sz[c] = mem_sz;
            qmat_nq[c] = nq_sorted;
            double tock = diverge_mpi_wtime();
            time_per_chan[c] = tock - tick;
        }
        mpi_tim_printf("q-matrices: %.2fs (P), %.2fs (C), %.2fs (D)\n", time_per_chan[0], time_per_chan[1], time_per_chan[2]);
    } else {
        for (int c=0; c<3; ++c) {
            qmat_nq[c] = 0;
            qmat_mem[c] = NULL;
            qmat_mem_sz[c] = 0;
        }
    }

    // vertex_in_q_channel
    for (int c=0; c<3; ++c) {
        header[i++] = chans[c];
        header[i++] = ocfg->q_matrices_use_dV;
        header[i++] = qmat_nq[c];
        header[i] = header[i-4] + header[i-5]; i++;
        header[i++] = qmat_mem_sz[c];
    }

    fwrite(header, sizeof(header), 1, f);

    if (ocfg->V)
        fwrite(v->V, sizeof(complex128_t), v->size, f);
    if (ocfg->Lp)
        fwrite(l->Lp, sizeof(complex128_t), v->size, f);
    if (ocfg->Lm)
        fwrite(l->Lm, sizeof(complex128_t), v->size, f);
    if (ocfg->dV)
        fwrite(v->dV, sizeof(complex128_t), v->size, f);

    for (int c=0; c<3; ++c) {
        if (qmat_mem[c] != NULL) {
            fwrite(qmat_mem[c], sizeof(uindex_t), qmat_mem_sz[c], f);
            free(qmat_mem[c]);
        }
    }

    fclose(f);
    return 0;
}

