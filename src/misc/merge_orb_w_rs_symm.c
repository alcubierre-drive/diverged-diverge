/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "merge_orb_w_rs_symm.h"

#define LINALG_NO_LAPACKE
#include "linalg.h"
LINALG_IMPLEMENT

void merge_rs_orb(diverge_model_t* model) {
    if (!model->orb_symmetries || model->n_sym == 0) return;
    complex128_t* orb_symmetries = model->orb_symmetries;
    index_t n_orb = model->n_orb;
    index_t n_spin = model->n_spin;
    index_t n_sym = model->n_sym;
    if (n_sym < 0) n_sym *= -1;
    index_t n_mi = n_orb * n_spin;

    complex128_t* orb_cpy = malloc( POW2(n_mi)*n_sym*sizeof(*orb_cpy) );
    memcpy( orb_cpy, orb_symmetries, POW2(n_mi)*n_sym*sizeof(*orb_cpy) );
    memset( orb_symmetries, 0, POW2(n_mi)*n_sym*sizeof(*orb_symmetries) );

    int periodic_iter[3];
    periodic_iter[0] = model->internals->periodic_direction[0] ? 3 : 0;
    periodic_iter[1] = model->internals->periodic_direction[1] ? 3 : 0;
    periodic_iter[2] = model->internals->periodic_direction[2] ? 3 : 0;

    claM3d_t UC = claM3d_map(model->lattice[0]);

    for (index_t s=0; s<n_sym; ++s)
    for (index_t o1=0; o1<n_orb; ++o1) {
        claM3d_t sym_rs = claM3d_map(model->rs_symmetries[s][0]);
        claV3d_t from = claV3d_map(model->positions[o1]);
        claV3d_t to = claM3d_matvec(sym_rs, from);
        for (index_t o2=0; o2<n_orb; ++o2) {
            int found = 0;
            claV3d_t ro2 = claV3d_map(model->positions[o2]);
            for (int in=-periodic_iter[0]; in<=periodic_iter[0]; in++)
            for (int im=-periodic_iter[1]; im<=periodic_iter[1]; im++)
            for (int iq=-periodic_iter[2]; iq<=periodic_iter[2]; iq++) {
                claV3d_t trial = claV3d_add(ro2, claM3d_vecmat((claV3d_t){{in,im,iq}},UC));
                if (claV3d_norm2(claV3d_sub(trial,to)) < DIVERGE_EPS_MESH) found = 1;
            }
            if (found)
                for (index_t s1=0; s1<n_spin; ++s1)
                for (index_t s2=0; s2<n_spin; ++s2) {
                    orb_symmetries[IDX5(s,s2,o2,s1,o1,n_spin,n_orb,n_spin,n_orb)] =
                           orb_cpy[IDX5(s,s2,o2,s1,o1,n_spin,n_orb,n_spin,n_orb)];
                }
        }
    }
    free(orb_cpy);
}

