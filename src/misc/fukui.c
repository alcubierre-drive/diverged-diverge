/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "fukui.h"
#include "../diverge_model.h"
#include "../diverge_model_internals.h"
#include "../diverge_internals_struct.h"
#include "../diverge_momentum_gen.h"
#include "mpi_functions.h"
#include "invert.h"

#define LINALG_NO_LAPACKE
#include "linalg.h"
LINALG_IMPLEMENT

#ifdef CLOG_MISSING
complex128_t clog(complex128_t z);
#endif

double* diverge_fukui( diverge_model_t* m, complex128_t* U_, index_t nb_, index_t* nk_ ) {

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

    if (errors) return NULL;

    double sgn = -1.0;
    int improper_gauge = 0;
    for (int i=0; i<3; ++i) {
        if (nk[i] < 0) {
            improper_gauge = 1;
            nk[i] *= -1;
            sgn *= -1;
        }
    }
    double* pos = NULL;
    double* kfmesh = NULL;
    double rlattice[3][3] = {0};
    index_t no = 0, ns = 0;
    if (!m && improper_gauge) {
        improper_gauge = 0;
        mpi_wrn_printf( "improper gauge requires model\n" );
    } else if (improper_gauge) {
        if (!m->internals->has_common_internals) {
            improper_gauge = 0;
            mpi_wrn_printf( "improper gauge requires common internals\n" );
        } else {
            // actuall improper gauge, no errors
            pos = m->positions[0];
            kfmesh = m->internals->kfmesh;
            diverge_model_generate_mom_basis(m->lattice[0], rlattice[0]);
            no = m->n_orb;
            ns = m->n_spin;
        }
    }

    index_t nb2 = nb*nb;
    index_t nk_tot = nk[0]*nk[1]*nk[2];
    double* out = calloc( nk_tot * nb * 3, sizeof(double) );

    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
    complex128_t* U_cube = calloc(8*nb2, sizeof(complex128_t));
    #pragma omp for collapse(3)
    for (index_t kx=0; kx<nk[0]; ++kx)
    for (index_t ky=0; ky<nk[1]; ++ky)
    for (index_t kz=0; kz<nk[2]; ++kz) {

        index_t k = IDX3(kx, ky, kz, nk[1], nk[2]);

        const int wraps[3] = {kx == (nk[0]-1), ky == (nk[1]-1), kz == (nk[2]-1)};
        int wraps_add[3] = {0};

        index_t cube_idx = 0;
        index_t k_idx = 0;

        #define U_CUBE_PUSH_BACK() \
            if (!improper_gauge) { \
                for (index_t bo=0; bo<nb2; ++bo) { \
                    U_cube[IDX2(cube_idx, bo, nb2)] = U[IDX2(k_idx, bo, nb2)]; \
                } \
            } else { \
                for (index_t b=0; b<nb; ++b) \
                for (index_t s=0; s<ns; ++s) \
                for (index_t o=0; o<no; ++o) { \
                    index_t U_cube_idx = IDX4(cube_idx, b, s, o, nb, ns, no); \
                    U_cube[U_cube_idx] = U[IDX4(k_idx, b, s, o, nb, ns, no)]; \
                    claV3d_t kvec = claV3d_map( kfmesh + k_idx*3 ); \
                    claV3d_t ovec = claV3d_map( pos + o*3 ); \
                    for (int i=0; i<3; ++i) { \
                        if (wraps_add[i] && wraps[i]) \
                            kvec = claV3d_add(kvec, claV3d_map(rlattice[i])); \
                    } \
                    U_cube[U_cube_idx] *= cexp( sgn*I*claV3d_dot(kvec,ovec) ); \
                } \
            } \
            cube_idx++;

        int oneX = nk[0] > 1 ? 1 : 0,
            oneY = nk[1] > 1 ? 1 : 0,
            oneZ = nk[2] > 1 ? 1 : 0;
        k_idx = k1p2(k, IDX3(0,   0,   0,   nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        wraps_add[2] = 1;
        k_idx = k1p2(k, IDX3(0,   0,   oneZ,nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        wraps_add[1] = 1, wraps_add[2] = 0;
        k_idx = k1p2(k, IDX3(0,   oneY,0,   nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        wraps_add[2] = 1;
        k_idx = k1p2(k, IDX3(0,   oneY,oneZ,nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        wraps_add[1] = wraps_add[2] = 0; wraps_add[0] = 1;
        k_idx = k1p2(k, IDX3(oneX,0,   0,   nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        wraps_add[2] = 1;
        k_idx = k1p2(k, IDX3(oneX,0,   oneZ,nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        wraps_add[1] = 1; wraps_add[2] = 0;
        k_idx = k1p2(k, IDX3(oneX,oneY,0,   nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        wraps_add[2] = 1;
        k_idx = k1p2(k, IDX3(oneX,oneY,oneZ,nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        for (index_t b1=0; b1<nb; ++b1)
        for (index_t dim=0; dim<3; ++dim) {

            complex128_t prod = 1.0;
            index_t cube_sel[4] = {0};
            switch (dim) {
                case 0: cube_sel[1] = 1; cube_sel[2] = 3; cube_sel[3] = 2; break;
                case 1: cube_sel[1] = 1; cube_sel[2] = 4; cube_sel[3] = 5; break;
                case 2: cube_sel[1] = 4; cube_sel[2] = 6; cube_sel[3] = 2; break;
                default: break;
            }
            for (int plaq=0; plaq<4; ++plaq) {
                complex128_t scalar_product = 0.0;
                for (index_t o=0; o<nb; ++o) {
                    index_t c1 = IDX3(cube_sel[(plaq+0)%4],b1,o,nb,nb),
                            c2 = IDX3(cube_sel[(plaq+1)%4],b1,o,nb,nb);
                    scalar_product += U_cube[c1] * conj(U_cube[c2]);
                }
                prod *= scalar_product;
            }
            out[IDX3( k, b1, dim, nb, 3 )] = carg(prod) / 2. / M_PI;
        }
    }
    free(U_cube);
    } // omp

    return out;
}

// Eq. (16) from [J. Phys. Soc. Jpn. 74, pp. 1674-1677 (2005)]
// https://doi.org/10.1143/JPSJ.74.1674
// Fukui/Hatsugai/Suzuki
double* diverge_fukui_non_abelian( diverge_model_t* m, complex128_t* U_,
        index_t nb_, index_t* nk_, index_t* which_bands, index_t n_which_bands ) {

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

    if (errors) return NULL;

    double sgn = -1.0;
    int improper_gauge = 0;
    for (int i=0; i<3; ++i) {
        if (nk[i] < 0) {
            improper_gauge = 1;
            nk[i] *= -1;
            sgn *= -1;
        }
    }
    double* pos = NULL;
    double* kfmesh = NULL;
    double rlattice[3][3] = {0};
    index_t no = 0, ns = 0;
    if (!m && improper_gauge) {
        improper_gauge = 0;
        mpi_wrn_printf( "improper gauge requires model\n" );
    } else if (improper_gauge) {
        if (!m->internals->has_common_internals) {
            improper_gauge = 0;
            mpi_wrn_printf( "improper gauge requires common internals\n" );
        } else {
            // actuall improper gauge, no errors
            pos = m->positions[0];
            kfmesh = m->internals->kfmesh;
            diverge_model_generate_mom_basis(m->lattice[0], rlattice[0]);
            no = m->n_orb;
            ns = m->n_spin;
        }
    }

    index_t nb2 = nb*nb;
    index_t nk_tot = nk[0]*nk[1]*nk[2];
    double* out = calloc( nk_tot * 3, sizeof(double) );

    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
    complex128_t* U_cube = calloc(8*nb2, sizeof(complex128_t));
    complex128_t* U_non_abel = calloc(POW2(n_which_bands), sizeof*U_non_abel);
    #pragma omp for collapse(3)
    for (index_t kx=0; kx<nk[0]; ++kx)
    for (index_t ky=0; ky<nk[1]; ++ky)
    for (index_t kz=0; kz<nk[2]; ++kz) {

        index_t k = IDX3(kx, ky, kz, nk[1], nk[2]);

        const int wraps[3] = {kx == (nk[0]-1), ky == (nk[1]-1), kz == (nk[2]-1)};
        int wraps_add[3] = {0};

        index_t cube_idx = 0;
        index_t k_idx = 0;

        #define U_CUBE_PUSH_BACK() \
            if (!improper_gauge) { \
                for (index_t bo=0; bo<nb2; ++bo) { \
                    U_cube[IDX2(cube_idx, bo, nb2)] = U[IDX2(k_idx, bo, nb2)]; \
                } \
            } else { \
                for (index_t b=0; b<nb; ++b) \
                for (index_t s=0; s<ns; ++s) \
                for (index_t o=0; o<no; ++o) { \
                    index_t U_cube_idx = IDX4(cube_idx, b, s, o, nb, ns, no); \
                    U_cube[U_cube_idx] = U[IDX4(k_idx, b, s, o, nb, ns, no)]; \
                    claV3d_t kvec = claV3d_map( kfmesh + k_idx*3 ); \
                    claV3d_t ovec = claV3d_map( pos + o*3 ); \
                    for (int i=0; i<3; ++i) { \
                        if (wraps_add[i] && wraps[i]) \
                            kvec = claV3d_add(kvec, claV3d_map(rlattice[i])); \
                    } \
                    U_cube[U_cube_idx] *= cexp( sgn*I*claV3d_dot(kvec,ovec) ); \
                } \
            } \
            cube_idx++;

        int oneX = nk[0] > 1 ? 1 : 0,
            oneY = nk[1] > 1 ? 1 : 0,
            oneZ = nk[2] > 1 ? 1 : 0;
        k_idx = k1p2(k, IDX3(0,   0,   0,   nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        wraps_add[2] = 1;
        k_idx = k1p2(k, IDX3(0,   0,   oneZ,nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        wraps_add[1] = 1, wraps_add[2] = 0;
        k_idx = k1p2(k, IDX3(0,   oneY,0,   nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        wraps_add[2] = 1;
        k_idx = k1p2(k, IDX3(0,   oneY,oneZ,nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        wraps_add[1] = wraps_add[2] = 0; wraps_add[0] = 1;
        k_idx = k1p2(k, IDX3(oneX,0,   0,   nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        wraps_add[2] = 1;
        k_idx = k1p2(k, IDX3(oneX,0,   oneZ,nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        wraps_add[1] = 1; wraps_add[2] = 0;
        k_idx = k1p2(k, IDX3(oneX,oneY,0,   nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        wraps_add[2] = 1;
        k_idx = k1p2(k, IDX3(oneX,oneY,oneZ,nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        for (index_t dim=0; dim<3; ++dim) {
            index_t cube_sel[4] = {0};
            switch (dim) {
                case 0: cube_sel[1] = 1; cube_sel[2] = 3; cube_sel[3] = 2; break;
                case 1: cube_sel[1] = 1; cube_sel[2] = 4; cube_sel[3] = 5; break;
                case 2: cube_sel[1] = 4; cube_sel[2] = 6; cube_sel[3] = 2; break;
                default: break;
            }
            complex128_t prod = 1.0;
            for (int plaq=0; plaq<4; ++plaq) {
                for (index_t ib1=0; ib1<n_which_bands; ++ib1)
                for (index_t ib2=0; ib2<n_which_bands; ++ib2) {
                    U_non_abel[IDX2(ib1,ib2,n_which_bands)] = 0.0;
                    index_t b1 = which_bands[ib1],
                            b2 = which_bands[ib2];
                    for (index_t o=0; o<nb; ++o) {
                        index_t c1 = IDX3(cube_sel[(plaq+0)%4],b1,o,nb,nb),
                                c2 = IDX3(cube_sel[(plaq+1)%4],b2,o,nb,nb);
                        U_non_abel[IDX2(ib1,ib2,n_which_bands)] += U_cube[c1] * conj(U_cube[c2]);
                    }
                }
                complex128_t det = cmatrix_determinant( U_non_abel, n_which_bands );
                prod *= det;
            }
            out[IDX2( k, dim, 3 )] = carg(prod) / 2. / M_PI;
        }
    }
    free(U_cube);
    free(U_non_abel);
    } // omp

    return out;
}
