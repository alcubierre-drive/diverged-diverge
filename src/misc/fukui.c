#include "fukui.h"
#include "../diverge_model.h"
#include "../diverge_model_internals.h"
#include "../diverge_internals_struct.h"
#include "init_internal_libs.h"

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

    index_t nb2 = nb*nb;
    index_t nk_tot = nk[0]*nk[1]*nk[2];
    double* out = calloc( nk_tot * nb * 3, sizeof(double) );

    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
    complex128_t* U_cube = calloc(8*nb2, sizeof(complex128_t));
    #pragma omp for
    for (index_t k=0; k<nk_tot; ++k) {
        index_t cube_idx = 0;
        index_t k_idx = 0;
        #define U_CUBE_PUSH_BACK() \
            for (index_t bo=0; bo<nb2; ++bo) { \
                U_cube[IDX2(cube_idx, bo, nb2)] = U[IDX2(k_idx, bo, nb2)]; \
            } \
            cube_idx++;

        int oneX = nk[0] > 1 ? 1 : 0,
            oneY = nk[1] > 1 ? 1 : 0,
            oneZ = nk[2] > 1 ? 1 : 0;
        k_idx = k1p2(k, IDX3(0,   0,   0,   nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        k_idx = k1p2(k, IDX3(0,   0,   oneZ,nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        k_idx = k1p2(k, IDX3(0,   oneY,0,   nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        k_idx = k1p2(k, IDX3(0,   oneY,oneZ,nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        k_idx = k1p2(k, IDX3(oneX,0,   0,   nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        k_idx = k1p2(k, IDX3(oneX,0,   oneZ,nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
        k_idx = k1p2(k, IDX3(oneX,oneY,0,   nk[1],nk[2]), nk); U_CUBE_PUSH_BACK();
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
