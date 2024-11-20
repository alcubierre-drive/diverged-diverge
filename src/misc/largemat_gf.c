#include "largemat_gf.h"
#include "gemm.h"
#include "mpi_functions.h"
#include "../diverge_model.h"
#include "../diverge_internals_struct.h"
#include "../diverge_model_internals.h"

static void transpose_memcpy( complex128_t* out, const complex128_t* in, index_t nb ) {
    for (index_t o1=0; o1<nb; ++o1)
    for (index_t o2=0; o2<nb; ++o2)
        out[IDX2(o1,o2,nb)] = in[IDX2(o2,o1,nb)];
}

greensfunc_op_t diverge_largemat_gf( const diverge_model_t* model, complex128_t Lambda, complex128_t* buf ) {

    const index_t nkf = kdimtot(model->nk, model->nkf);
    const index_t nb = model->n_spin * model->n_orb;

    complex128_t* U = diverge_model_internals_get_U(model);
    double* E = diverge_model_internals_get_E(model);

    for (index_t s=0; s<2; ++s) {
        complex128_t L = s ? Lambda : conj(Lambda);
        #pragma omp parallel num_threads(diverge_omp_num_threads())
        {
            complex128_t* scratch1 = calloc(nb*nb, sizeof(complex128_t));
            complex128_t* scratch2 = calloc(nb*nb, sizeof(complex128_t));
            #pragma omp for
            for (index_t k=0; k<nkf; ++k) {
                memset( scratch1, 0, sizeof(complex128_t)*nb*nb );
                for (index_t b=0; b<nb; ++b) scratch1[b*nb+b] = 1./(L - E[IDX2(k,b,nb)]);
                single_square_gemm( U+k*nb*nb, 1, scratch1, 0, scratch2, nb );
                single_square_gemm( scratch2, 0, U+k*nb*nb, 0, scratch1, nb );
                transpose_memcpy( buf + (s*nkf+k)*nb*nb, scratch1, nb );
            }
            free(scratch1);
            free(scratch2);
        }
    }

    return greensfunc_op_cpu;
}

