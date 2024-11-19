#include "ham_at_k.h"
#include "../diverge_model.h"
#include "../diverge_internals_struct.h"

#define LINALG_NO_LAPACKE
#include "linalg.h"
LINALG_IMPLEMENT

void diverge_model_ham_at_kpt( const diverge_model_t* m, const double* k, complex128_t* buf ) {
    const index_t no = m->n_orb,
                  ns = m->n_spin,
                  nb = m->n_orb * m->n_spin;
    memset( buf, 0, sizeof*buf * POW2(nb) );
    const claV3d_t kvec = claV3d_map((double*)k);
    const claM3d_t Lvec = claM3d_map((double*)m->lattice[0]);
    for (const rs_hopping_t* h=m->hop; h!=m->hop+m->n_hop; ++h) {
        claV3d_t hR = {{h->R[0], h->R[1], h->R[2]}};
        claV3d_t Rvec = claM3d_vecmat( hR, Lvec );
        buf[IDX4(h->s1, h->o1, h->s2, h->o2, no, ns, no)] += cexp(-I*claV3d_dot(Rvec,kvec)) * h->t;
    }
}

void diverge_model_ham_at_kcidx( const diverge_model_t* m, index_t k, complex128_t* buf ) {
    if (!m->internals->has_common_internals) {
        mpi_err_printf( "need common internals for accessing kmesh via index\n" );
        return;
    }
    diverge_model_ham_at_kpt( m, m->internals->kmesh+3*k, buf );
}

void diverge_model_ham_at_kfidx( const diverge_model_t* m, index_t k, complex128_t* buf ) {
    if (!m->internals->has_common_internals) {
        mpi_err_printf( "need common internals for accessing kfmesh via index\n" );
        return;
    }
    diverge_model_ham_at_kpt( m, m->internals->kfmesh+3*k, buf );
}
