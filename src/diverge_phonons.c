#include "diverge_phonons.h"
#include "diverge_common.h"
#include "diverge_model.h"
#include "misc/mpi_functions.h"

rs_hopping_t* diverge_phonons_asr( rs_hopping_t* ph_hop, index_t* n_ph_hop ) {
    mpi_log_printf( "apply ASR to phonon hoppings in format o1/o2 -> (site, direction)\n" );
    index_t n_hop = *n_ph_hop;

    index_t n_orb = 0;
    #pragma omp parallel for reduction(max:n_orb) num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<n_hop; ++i) {
        index_t om = MAX(ph_hop[i].o1, ph_hop[i].o2);
        n_orb = MAX(n_orb, om);
    }
    n_orb = (n_orb+1)/3;
    mpi_vrb_printf( "found (%li,3) orbitals\n", n_orb );

    complex128_t* self_forces = calloc(n_orb*9, sizeof*self_forces);

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<n_hop; ++i) {
        int ho1 = ph_hop[i].o1/3, hr1 = ph_hop[i].o1%3, hr2 = ph_hop[i].o2%3;
        self_forces[IDX3(ho1,hr1,hr2, 3,3)] += ph_hop[i].t;
    }

    ph_hop = realloc(ph_hop, (*n_ph_hop = (n_hop + (n_orb*9))) * sizeof*ph_hop);
    #pragma omp parallel for num_threads(diverge_omp_num_threads()) collapse(3)
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

