/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "vertex.h"
#include "vertex_helper.h"
#include "loop.h"
#include "../misc/eigen.h"
#include "../misc/batched_eigen.h"
#include "../misc/mpi_functions.h"

#include "../diverge_model.h"
#include "../diverge_internals_struct.h"

#define POW4(x) ((x)*(x)*(x)*(x))

npatch_vertex_t* npatch_vertex_init( diverge_model_t* dmod ) {
    index_t np = dmod->patching->n_patches;
    index_t nb = dmod->internals->patch_nb;
    npatch_vertex_t* result = (npatch_vertex_t*)malloc(sizeof(npatch_vertex_t));
    result->size = POW3(np) * POW4(nb);
    result->V = (complex128_t*)malloc(sizeof(complex128_t)*result->size);
    result->dV = (complex128_t*)malloc(sizeof(complex128_t)*result->size);
    return result;
}

void npatch_vertex_free( npatch_vertex_t* vert ) {
    free(vert->V);
    free(vert->dV);
    free(vert);
}

npatch_qvert_mat_t* npatch_qvert_mat_from_vertex( const npatch_vertex_t* vert,
        const npatch_loop_t* loop, char chan, bool dV, double max_q_rel,
        index_t* pnq, index_t* pnq_sorted ) {
    complex128_t* qvert;
    qkkp_t* qvindex;
    index_t* qcounts;
    vhelper_vertex_in_q_channel( vert, loop, chan, dV, &qvert, &qvindex, &qcounts, pnq );
    npatch_qvert_mat_t* result = vhelper_q_vertex_matrix_from_vertex( qvert, qvindex, qcounts, *pnq, loop->nb );

    *pnq_sorted = *pnq;
    if (max_q_rel > 0 && max_q_rel < 1)
        vhelper_q_vertex_sort_max( result, pnq_sorted, max_q_rel );

    free( qvert );
    free( qvindex );
    free( qcounts );
    return result;
}

void npatch_qvert_mat_free( npatch_qvert_mat_t* qvm, index_t nq ) {
    for (npatch_qvert_mat_t* qvm_idx = qvm; qvm_idx != qvm + nq; qvm_idx++) {
        free( qvm_idx->data );
        free( qvm_idx->k_kp_descr );
        if (qvm_idx->values != NULL && qvm_idx->eigen && qvm_idx->nv > 0)
            free( qvm_idx->values );
    }
    free( qvm );
}

void npatch_qvert_mat_eigen( npatch_qvert_mat_t* qvm, index_t nq, index_t nv, char which ) {
    if (nv == -1) {
        for (int q=0; q<nq; ++q) {
            qvm[q].eigen = false;
            qvm[q].nv = -1;
            qvm[q].values = NULL;
        }
    } else {
        index_t Nmax = 0;
        for (int q=0; q<nq; ++q) {
            index_t N = qvm[q].nb * qvm[q].nb * qvm[q].n_k_kp;
            Nmax = Nmax <= N ? N : Nmax;
        }
        mpi_vrb_printf("vertex to eigenbasis (nq=%lli, Nmax=%lli)…\n", nq, Nmax);
        complex128_t* vector = vhelper_q_vertex_matrices_to_buffer( qvm, nq, Nmax );
        double* valueR = (double*)calloc(Nmax*nq,sizeof(double));
        batched_eigen_shut_up(1);
        batched_eigen_r( NULL, -1, vector, valueR, Nmax, nq );
        batched_eigen_shut_up(0);
        #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
        for (int q=0; q<nq; ++q) {
            single_eigen_sort( vector+q*Nmax*Nmax, valueR+q*Nmax, Nmax, which );

            index_t N = qvm[q].nb * qvm[q].nb * qvm[q].n_k_kp;
            qvm[q].eigen = true;
            qvm[q].nv = (nv > N || nv == 0) ? N : nv;
            qvm[q].values = realloc(qvm[q].values, sizeof(double)*qvm[q].nv);
            for (index_t v=0; v<qvm[q].nv; ++v) {
                qvm[q].values[v] = valueR[v+q*Nmax];
                for (index_t i=0; i<N; ++i)
                    qvm[q].data[i+v*N] = vector[q*Nmax*Nmax + i + v*Nmax];
            }
        }
        free(valueR);
        free(vector);
    }
}

uindex_t* npatch_qvert_mat_serialize( npatch_qvert_mat_t* qvm, index_t* psize ) {

    *psize = ( qvm->eigen ? (qvm->nv * qvm->n_k_kp * POW2(qvm->nb) * 2 + qvm->nv) :
                            (POW2(qvm->n_k_kp) * POW4(qvm->nb) * 2) )
             + qvm->n_k_kp + 5;

    uindex_t* result = malloc( sizeof(uindex_t) * *psize );
    index_t* ihead = (index_t*)result;
    *ihead = qvm->n_k_kp; ihead++;
    *ihead = qvm->nb; ihead++;
    *ihead = qvm->q; ihead++;
    *ihead = qvm->eigen; ihead++;
    *ihead = qvm->nv; ihead++;

    memcpy( ihead, qvm->k_kp_descr, sizeof(index_t)*qvm->n_k_kp );
    ihead += qvm->n_k_kp;

    if (qvm->eigen) {
        memcpy( ihead, qvm->values, qvm->nv * sizeof(double) );
        ihead += qvm->nv;
        memcpy( ihead, qvm->data, qvm->nv * qvm->n_k_kp * POW2(qvm->nb) * sizeof(complex128_t) );
        ihead += qvm->nv * qvm->n_k_kp * POW2(qvm->nb) * 2;
    } else {
        memcpy( ihead, qvm->data, POW2(qvm->n_k_kp) * POW4(qvm->nb) * sizeof(complex128_t) );
        ihead += POW2(qvm->n_k_kp) * POW4(qvm->nb) * 2;
    }

    return result;
}

npatch_qvert_mat_t* npatch_qvert_mat_deserialize( uindex_t* mem ) {
    npatch_qvert_mat_t* result = (npatch_qvert_mat_t*)malloc(sizeof(npatch_qvert_mat_t));
    index_t* ihead = (index_t*)mem;
    result->n_k_kp = *ihead; ihead++;
    result->nb = *ihead; ihead++;
    result->q = *ihead; ihead++;
    result->eigen = *ihead; ihead++;
    result->nv = *ihead; ihead++;

    result->k_kp_descr = (index_t*)malloc(sizeof(index_t) * result->n_k_kp);
    memcpy( result->k_kp_descr, ihead, sizeof(index_t) * result->n_k_kp );

    ihead += result->n_k_kp;

    result->data = (complex128_t*)malloc(POW2(result->n_k_kp) * POW4(result->nb) * sizeof(complex128_t) );
    memcpy( result->data, ihead, POW2(result->n_k_kp) * POW4(result->nb) * sizeof(complex128_t) );

    ihead += POW2(result->n_k_kp) * POW4(result->nb) * 2;

    if (result->eigen && result->nv > 0) {
        result->values = (double*)malloc( result->nv * sizeof(double) );
        memcpy( ihead, result->values, result->nv * sizeof(double) );
    }

    return result;
}

