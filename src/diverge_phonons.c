#include "diverge_phonons.h"
#include "diverge_common.h"
#include "diverge_model.h"
#include "misc/mpi_functions.h"

#include "misc/shared_mem.h"
#include "misc/bitary.h"
#include "diverge_model.h"
#include "diverge_model_internals.h"
#include "diverge_internals_struct.h"
#include <assert.h>

rs_hopping_t* diverge_phonons_asr( rs_hopping_t* ph_hop, index_t* n_ph_hop ) {
    mpi_log_printf( "apply ASR to phonon hoppings in format o1/o2 -> (site, direction)\n" );
    index_t n_hop = *n_ph_hop;

    index_t n_orb = 0;
    for (index_t i=0; i<n_hop; ++i) {
        index_t om = MAX(ph_hop[i].o1, ph_hop[i].o2);
        n_orb = MAX(n_orb, om);
    }
    n_orb = (n_orb+1)/3;
    mpi_vrb_printf( "found (%li,3) orbitals\n", n_orb );

    complex128_t* self_forces = calloc(n_orb*9, sizeof*self_forces);
    for (index_t i=0; i<n_hop; ++i) {
        int ho1 = ph_hop[i].o1/3, hr1 = ph_hop[i].o1%3, hr2 = ph_hop[i].o2%3;
        self_forces[IDX3(ho1,hr1,hr2, 3,3)] += ph_hop[i].t;
    }

    ph_hop = realloc(ph_hop, (*n_ph_hop = (n_hop + (n_orb*9))) * sizeof*ph_hop);
    for (index_t o=0; o<n_orb; ++o)
    for (index_t r1=0; r1<3; ++r1)
    for (index_t r2=0; r2<3; ++r2) {
        ph_hop[n_hop + IDX3(o,r1,r2, 3,3)] = (rs_hopping_t){
            .o1=IDX2(o,r1,3), .o2=IDX2(o,r2,3),
            .t=-self_forces[IDX3(o,r1,r2, 3,3)],
        };
    }

    free(self_forces);
    return ph_hop;
}

void diverge_phonons_apply_sgnsqrt( diverge_model_t* m ) {
    assert( m->n_spin == 1 );
    double* E = diverge_model_internals_get_E(m);
    if (!m->internals->model_phonons_imag_freq)
        m->internals->model_phonons_imag_freq = bitary_alloc(kdimtot(m->nk,m->nkf)*m->n_orb);
    int nimag = 0;
    if (shared_exclusive_enter(E)) for (index_t k=0; k<kdimtot(m->nk,m->nkf)*m->n_orb; ++k) {
        int isimag = (E[k] < -DIVERGE_EPS_MESH);
        if (isimag) {
            bitary_set(m->internals->model_phonons_imag_freq, k);
            nimag++;
        }
        E[k] = (isimag?-1.0:1.0) * sqrt(fabs(E[k]));
    }
    shared_exclusive_wait(E);
    diverge_mpi_allreduce_int_max_inplace( &nimag, 1 );
    if (nimag)
        mpi_wrn_printf( "found %li imaginary phonon frequencies\n", nimag );
}

void* diverge_phonons_get_ifrq( diverge_model_t* m ) {
    return m->internals->model_phonons_imag_freq;
}
