/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "quantum_metric.h"
#include "../diverge_model.h"
#include "../diverge_model_internals.h"
#include "../diverge_internals_struct.h"
#include "../diverge_momentum_gen.h"
#include "mpi_functions.h"

#define LINALG_NO_LAPACKE
#define LINALG_EXPORT static inline
#include "linalg.h"
LINALG_IMPLEMENT

typedef struct {
    double* lattice;
    double* pos;
    double* kfmesh;
    index_t no;
    double sgn;
} qgt_improper_gauge_t;

static double* diverge_qgt_impl_grad( double* lattice_, complex128_t* U,
        index_t nb, index_t* nk, index_t* which_bands, index_t n_which_bands,
        const qgt_improper_gauge_t* ig ) {

    claM3cd_t lattice = {0};
    for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j)
        lattice.m[i][j] = lattice_[IDX2(i,j,3)];

    double rlattice[3][3] = {0};
    if (ig)
        diverge_model_generate_mom_basis( ig->lattice, rlattice[0] );
    else
        diverge_model_generate_mom_basis( lattice_, rlattice[0] );

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

    #pragma omp parallel for num_threads(diverge_omp_num_threads()) collapse(3)
    for (index_t kx=0; kx<nk[0]; ++kx)
    for (index_t ky=0; ky<nk[1]; ++ky)
    for (index_t kz=0; kz<nk[2]; ++kz) {
        index_t ks[] = {kx, ky, kz};
        index_t k = IDX3(kx, ky, kz, nk[1], nk[2]);
        claV3d_t kvec = {0};
        if (ig)
            kvec = claV3d_map(ig->kfmesh + 3*k);
        double dP_trace = 0;
        for (index_t o1=0; o1<nb; ++o1)
        for (index_t o2=0; o2<nb; ++o2) {
            claV3d_t o1vec = {0};
            claV3d_t o2vec = {0};
            if (ig) {
                o1vec = claV3d_map(ig->pos + 3*(o1 % ig->no));
                o2vec = claV3d_map(ig->pos + 3*(o2 % ig->no));
            }
            claV3cd_t f;
            for (uint8_t dim=0; dim<3; ++dim) {
                // wrap around
                claV3d_t dkvec = {0};
                index_t k_dk = k1p2( k, dk[dim], nk );
                if (ig)
                    dkvec = claV3d_map(ig->kfmesh + 3*k_dk);
                if (ks[dim] == (nk[dim]-1))
                    dkvec = claV3d_add(dkvec, claV3d_map(rlattice[dim]));
                complex128_t P_o1_o2_k = 0,
                             P_o1_o2_k_dk = 0;
                for (index_t ww=0; ww<n_which_bands; ++ww) {
                    index_t bb=which_bands[ww];
                    complex128_t fac = 1.0;
                    if (ig)
                        fac = cexp( ig->sgn*I*claV3d_dot(kvec, claV3d_sub(o1vec,o2vec)) );
                    P_o1_o2_k += U[IDX3( k, bb, o1, nb, nb )] * conj(U[IDX3( k, bb, o2, nb, nb )]) * fac;
                    if (ig)
                        fac = cexp( ig->sgn*I*claV3d_dot(dkvec, claV3d_sub(o1vec,o2vec)) );
                    P_o1_o2_k_dk += U[IDX3( k_dk, bb, o1, nb, nb )] * conj(U[IDX3( k_dk, bb, o2, nb, nb )]) * fac;
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
        index_t* which_bands, index_t n_which_bands, qgt_improper_gauge_t* ig ) {
    claM3d_t lattice = claM3d_id(); 
    return diverge_qgt_impl_grad( lattice.m[0], U, nb, nk, which_bands, n_which_bands, ig );
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
    if (n_which_bands < 0 || n_which_bands > nb) {
        mpi_err_printf( "# of which bands must be within %li,%li\n", 0, nb );
        errors++;
    }

    if (errors) return NULL;

    if (!m && non_ortho_grad) {
        mpi_wrn_printf( "cannot use non-orthogonal algorithm whem model is not supplied\n" );
        non_ortho_grad = 0;
    }

    qgt_improper_gauge_t ig = {0};
    ig.sgn = -1.0;

    int improper_gauge = 0;
    for (int i=0; i<3; ++i) {
        if (nk[i] < 0) {
            improper_gauge = 1;
            nk[i] *= -1;
            ig.sgn *= -1;
        }
    }

    if (!m && improper_gauge) {
        improper_gauge = 0;
        mpi_wrn_printf( "improper gauge requires model\n" );
    } else if (improper_gauge) {
        if (!m->internals->has_common_internals) {
            improper_gauge = 0;
            mpi_wrn_printf( "improper gauge requires common internals\n" );
        } else {
            // actuall improper gauge, no errors
            ig.pos = m->positions[0];
            ig.kfmesh = m->internals->kfmesh;
            ig.lattice = m->lattice[0];
            ig.no = m->n_orb;
        }
    }

    if (!non_ortho_grad)
        return diverge_qgt_impl( U, nb, nk, which_bands, n_which_bands,
                improper_gauge ? &ig : NULL );
    else
        return diverge_qgt_impl_grad( m->lattice[0], U, nb, nk, which_bands, n_which_bands,
                improper_gauge ? &ig : NULL );

}
