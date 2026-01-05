/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "kmesh_to_bands.h"
#include "qnan.h"
#include "mpi_functions.h"
#include "../diverge_model.h"
#include "../diverge_internals_struct.h"
#include "../diverge_momentum_gen.h"

#include <omp.h>

#define LINALG_NO_LAPACKE
#include "linalg.h"
LINALG_IMPLEMENT

static void find_bandstructure_segment_indices( const double start[3],
        const double stop[3], const double* kfmesh, index_t nktot, index_t**
        indices, index_t* n_indices );
static index_t* diverge_kmesh_to_bands_both( diverge_model_t* m, index_t**
        indices, index_t* n_indices, bool fine );
static inline index_t argmax( const double* ary, index_t sz );
static inline int compar_kidx_tval_t( const void* vA, const void* vB );
typedef struct {
    index_t kidx;
    double tval;
} kidx_tval_t;

index_t* diverge_kmesh_to_bands( diverge_model_t* m, index_t** indices, index_t* n_indices ) {
    return diverge_kmesh_to_bands_both( m, indices, n_indices, true );
}
index_t* diverge_kmesh_to_bands_crs( diverge_model_t* m, index_t** indices, index_t* n_indices ) {
    return diverge_kmesh_to_bands_both( m, indices, n_indices, false );
}

static void find_bandstructure_segment_indices( const double start_[3], const double stop_[3],
        const double* kfmesh, index_t nktot, index_t** indices, index_t* n_indices ) {

    claV3d_t start = claV3d_map( start_ ), stop = claV3d_map( stop_ );

    // the line is given by L(t) = A + t * B
    claV3d_t A = start, B = claV3d_sub(stop,start);
    double B2 = claV3d_norm2(B);

    claV3d_t generator = claV3d_zero(); generator.v[(argmax(B.v,3)+1)%3] = 1.0;
    claV3d_t Bo1 = claV3d_cross(generator, B);
    claV3d_t Bo2 = claV3d_cross(B, Bo1);

    VECTOR_INIT( kidx_tval_t, kts_a );

    int nthr = diverge_omp_num_threads();
    index_t* k_counts = diverge_distribute( nktot, nthr );
    index_t* k_displs = k_counts + nthr;
    #pragma omp parallel num_threads(nthr)
    {
        int t = omp_get_thread_num();
        VECTOR_INIT( kidx_tval_t, kts );
        for (index_t kc=0; kc<k_counts[t]; ++kc) {
            index_t k = k_displs[t] + kc;
            claV3d_t K = claV3d_map(kfmesh + 3*k);
            claV3d_t KA = claV3d_sub(K,A);
            double tval = claV3d_dot( KA, B ) / B2;
            short ortho_1 = (fabs(claV3d_dot(KA,Bo1)) < DIVERGE_EPS_MESH),
                  ortho_2 = (fabs(claV3d_dot(KA,Bo2)) < DIVERGE_EPS_MESH);
            if (ortho_1 && ortho_2 && tval > (-DIVERGE_EPS_MESH) && tval < (1.+DIVERGE_EPS_MESH)) {
                VECTOR_PUSH_BACK( kts, (kidx_tval_t){.kidx=k, .tval=tval} );
            }
        }
        #pragma omp critical
        for (index_t k=0; k<kts_sz; ++k) {
            VECTOR_PUSH_BACK( kts_a, kts[k] );
        }
        free( kts );
    }
    free( k_counts );

    qsort( kts_a, kts_a_sz, sizeof*kts_a, &compar_kidx_tval_t );
    *indices = malloc( kts_a_sz * sizeof*indices );
    *n_indices = kts_a_sz;
    for (index_t i=0; i<kts_a_sz; ++i)
        (*indices)[i] = kts_a[i].kidx;
    free( kts_a );
}

static index_t* diverge_kmesh_to_bands_both( diverge_model_t* m, index_t**
        indices, index_t* n_indices, bool fine ) {
    if (m->n_ibz_path <= 1) {
        *n_indices = -1;
        *indices = NULL;
        mpi_err_printf( "cannot find IBZ path indices if n_ibz_path <= 1\n" );
        return NULL;
    }
    index_t *result = NULL, sresult = 0;
    double gvec[3][3];
    diverge_model_generate_mom_basis( m->lattice[0], gvec[0] );
    claM3d_t Gvec = claM3d_map(gvec[0]);
    index_t* n_per_segment = (index_t*)malloc(sizeof(index_t)*m->n_ibz_path-1);
    for (index_t segment=0; segment<m->n_ibz_path-1; ++segment) {

        // find whether we should skip this segment
        int skip_segment = 0;
        for (int add=0; add<2; ++add) {
            if (qnan_isnan( m->ibz_path[segment+add][0] )) {
                skip_segment = (qnan_get( m->ibz_path[segment+add][0] ) ==
                                qnan_get(diverge_model_ibz_path_skip_segment()));
                if (!skip_segment) {
                    mpi_wrn_printf( "segment contains qnan-skip (%u), but not the right one (%u)\n",
                            qnan_get( m->ibz_path[segment+add][0] ),
                            qnan_get( diverge_model_ibz_path_skip_segment() ) );
                }
            }
        }

        if (!skip_segment) {
            claV3d_t start_crystal = claV3d_map( m->ibz_path[segment] ),
                     stop_crystal = claV3d_map( m->ibz_path[segment+1] );

            claV3d_t start = claM3d_vecmat( start_crystal, Gvec ),
                     stop = claM3d_vecmat( stop_crystal, Gvec );

            if (claV3d_norm(claV3d_sub(start,stop)) < DIVERGE_EPS_MESH)
                mpi_err_printf( "each segment must cover nonzero distance\n" );

            index_t *sres = NULL, nsres = 0;
            if (fine)
                find_bandstructure_segment_indices( start.v, stop.v, m->internals->kfmesh,
                        kdimtot(m->nk, m->nkf), &sres, &nsres );
            else
                find_bandstructure_segment_indices( start.v, stop.v, m->internals->kmesh,
                        kdim(m->nk), &sres, &nsres );
            result = (index_t*)realloc( result, sizeof(index_t) * (sresult + nsres) );
            memcpy( result + sresult, sres, sizeof(index_t)*nsres );
            sresult += nsres;
            free( sres );
            n_per_segment[segment] = nsres;
        } else {
            n_per_segment[segment] = 0;
        }
    }
    *indices = result;
    *n_indices = sresult;
    return n_per_segment;
}

static inline index_t argmax( const double* ary, index_t sz ) {
    if (sz <= 0) return 0;
    index_t arg = 0;
    double val = fabs(ary[arg]);
    for (index_t i=1; i<sz; ++i)
        if (fabs(ary[i]) > val) {
            val = fabs(ary[i]);
            arg = i;
        }
    return arg;
}

static inline int compar_kidx_tval_t( const void* vA, const void* vB ) {
    const kidx_tval_t *A = vA, *B = vB;
    if (A->tval > B->tval) return 1;
    if (A->tval < B->tval) return -1;
    return 0;
}

