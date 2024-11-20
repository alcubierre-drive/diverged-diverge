#include "subspace_gf.h"
#include "shared_mem.h"
#include "mpi_functions.h"

#ifndef USE_CUDA

struct subspace_gfill_multi_t {
    double* E;
    complex128_t* U;
    index_t nk;
    index_t nb;
    index_t* orbs;
    index_t n_orbs;
};

subspace_gfill_multi_t* subspace_gfill_multi_init( double* E, complex128_t* U,
        index_t nk, const index_t* orbs, index_t n_orbs, index_t nb, index_t b_per_bb ) {
    (void)b_per_bb;
    mpi_wrn_printf( "subspace GF works on CPU, but won't be much faster...\n" );
    subspace_gfill_multi_t* s = calloc(1, sizeof(subspace_gfill_multi_t));
    s->E = E;
    s->U = U;
    s->nk = nk;
    s->nb = nb;
    s->n_orbs = n_orbs;
    s->orbs = calloc(s->n_orbs, sizeof(*s->orbs));
    memcpy( s->orbs, orbs, sizeof(*s->orbs)*s->n_orbs );
    return s;
}

void subspace_gfill_multi_free( subspace_gfill_multi_t* m ) {
    free( m->orbs );
    free( m );
}

void subspace_gfill_multi_exec( subspace_gfill_multi_t* m, complex128_t* dest, complex128_t Lambda ) {
    subspace_gfill_multi_exec_mu( m, dest, Lambda, 0.0 );
}

void subspace_gfill_multi_exec_mu( subspace_gfill_multi_t* m, complex128_t* dest, complex128_t Lambda, double mu_add ) {
    const index_t no = m->n_orbs,
                  *orbs = m->orbs,
                  nk = m->nk,
                  nb = m->nb;
    const complex128_t* U = m->U;
    const double* E = m->E;

    for (int sign=0; sign<2; ++sign) {
        complex128_t L = sign ? Lambda : conj(Lambda);
        #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
        for (index_t k=0; k<nk; ++k)
        for (index_t o=0; o<no; ++o)
        for (index_t p=0; p<no; ++p) {
            complex128_t tmp = 0.0;
            for (index_t b=0; b<nb; ++b)
                tmp += U[IDX3(k,b,orbs[o],nb,nb)] * conj(U[IDX3(k,b,orbs[p],nb,nb)]) / (L - E[IDX2(k,b,nb)] + mu_add);
            dest[IDX4(sign, k, o, p, nk, no, no)] = tmp;
        }
    }
}

#else // USE_CUDA

#include "subspace_gf.cu.h"

subspace_gfill_multi_t* subspace_gfill_multi_init( double* E, complex128_t* U,
        index_t nk, const index_t* orbs, index_t n_orbs, index_t nb, index_t b_per_bb ) {
    return subspace_gfill_multi_init_cu( E, U, nk, orbs, n_orbs, nb, b_per_bb );
}
void subspace_gfill_multi_free( subspace_gfill_multi_t* m ) {
    subspace_gfill_multi_free_cu( m );
}
void subspace_gfill_multi_exec( subspace_gfill_multi_t* m, complex128_t* dest, complex128_t Lambda ) {
    subspace_gfill_multi_exec_mu( m, dest, Lambda, 0.0 );
}
void subspace_gfill_multi_exec_mu( subspace_gfill_multi_t* m, complex128_t* dest, complex128_t Lambda, double mu_add ) {
    subspace_gfill_multi_exec_cu( m, dest, Lambda, mu_add );
}

#endif // USE_CUDA
