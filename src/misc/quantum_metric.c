/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "quantum_metric.h"
#include "../diverge_model.h"
#include "../diverge_model_internals.h"
#include "../diverge_internals_struct.h"
#include "init_internal_libs.h"

#define LINALG_NO_LAPACKE
#include "linalg.h"
LINALG_IMPLEMENT

static double* diverge_qgt_impl_grad( const double* lattice_, complex128_t* U,
        index_t nb, index_t* nk, index_t* which_bands, index_t n_which_bands ) {
    claM3cd_t lattice = {0};
    for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j)
        lattice.m[i][j] = lattice_[IDX2(i,j,3)];
    claM3cd_t AAT = claM3cd_matmul( lattice, claM3cd_transpose(lattice) );
    const index_t nk_tot = nk[0]*nk[1]*nk[2];
    const int oneX = nk[0] > 1 ? 1 : 0,
              oneY = nk[1] > 1 ? 1 : 0,
              oneZ = nk[2] > 1 ? 1 : 0;
    const index_t dkx = IDX3(oneX,0,0,nk[1],nk[2]),
                  dky = IDX3(0,oneY,0,nk[1],nk[2]),
                  dkz = IDX3(0,0,oneZ,nk[1],nk[2]);
    const index_t dk[] = { dkx, dky, dkz };

    double* result = (double*)calloc(nk_tot, sizeof(double));

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk_tot; ++k) {
        double dP_trace = 0;
        for (index_t o1=0; o1<nb; ++o1)
        for (index_t o2=0; o2<nb; ++o2) {
            claV3cd_t f;
            for (uint8_t dim=0; dim<3; ++dim) {
                index_t k_dk = k1p2( k, dk[dim], nk );
                complex128_t P_o1_o2_k = 0,
                             P_o1_o2_k_dk = 0;
                for (index_t ww=0; ww<n_which_bands; ++ww) {
                    index_t bb=which_bands[ww];
                    P_o1_o2_k += U[IDX3( k, bb, o1, nb, nb )] * conj(U[IDX3( k, bb, o2, nb, nb )]);
                    P_o1_o2_k_dk += U[IDX3( k_dk, bb, o1, nb, nb )] * conj(U[IDX3( k_dk, bb, o2, nb, nb )]);
                }
                complex128_t dP_o1_o2 = P_o1_o2_k_dk - P_o1_o2_k;
                f.v[dim] = dP_o1_o2;
            }
            dP_trace += creal(claV3cd_dot( f, claM3cd_matvec( AAT, f ) ));
        }
        result[k] = 0.5 * dP_trace;
    }

    return result;
}

static double* diverge_qgt_impl( complex128_t* U, index_t nb, index_t* nk,
        index_t* which_bands, index_t n_which_bands ) {
    claM3d_t lattice = claM3d_id(); 
    return diverge_qgt_impl_grad( lattice.m[0], U, nb, nk, which_bands, n_which_bands );
}

double* diverge_qgt( diverge_model_t* m, complex128_t* U_, index_t nb_,
        index_t* nk_, index_t* which_bands, index_t n_which_bands,
        int non_ortho_grad ) {

    index_t nk[3] = {0};
    complex128_t* U = NULL;
    index_t nb = 0;

    // using model data as default
    if (m) {
        for (int x=0; x<3; ++x) {
            nk[x] = m->nk[x] * m->nkf[x];
            if (nk[x] == 0) nk[x] = 1;
        }
        if (!U)
            U = diverge_model_internals_get_U(m);
        nb = m->n_orb * m->n_spin;
        mpi_log_printf( "obtaining nk=(%li,%li,%li), nb=%li, U(k,b,o) from model\n", nk[0], nk[1], nk[2], nb );
    }

    if (nk_) {
        memcpy(nk, nk_, sizeof(index_t)*3);
        for (int x=0; x<3; ++x) if (nk[x] == 0) nk[x] = 1;
        mpi_log_printf( "using user supplied nk=(%li,%li,%li)\n", nk[0], nk[1], nk[2] );
    }
    if (U_) {
        U = U_;
        mpi_log_printf( "using user supplied U(k,b,o)\n" );
    }
    if (nb_ > 0) {
        nb = nb_;
        mpi_log_printf( "using user supplied nb=%li\n", nb );
    }


    int errors = 0;
    if (nb <= 0) {
        mpi_err_printf( "found nb=%li\n", nb );
        errors++;
    }
    if (U == NULL) {
        mpi_err_printf( "found U(k,b,o)=NULL\n" );
        errors++;
    }
    if (!which_bands) {
        mpi_err_printf( "which bands cannot be NULL\n" );
        errors++;
    }
    if (n_which_bands < 0 || n_which_bands >= nb) {
        mpi_err_printf( "# of which bands must be within %li,%li\n", 0, nb );
        errors++;
    }

    if (errors) return NULL;

    if (!m && non_ortho_grad) {
        mpi_wrn_printf( "cannot use non-orthogonal algorithm whem model is not supplied\n" );
        non_ortho_grad = 0;
    }

    if (!non_ortho_grad)
        return diverge_qgt_impl( U, nb, nk, which_bands, n_which_bands );
    else
        return diverge_qgt_impl_grad( m->lattice[0], U, nb, nk, which_bands, n_which_bands );

}
