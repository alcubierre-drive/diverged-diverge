#include "all.h"
#include "../helper_functions.h"
#include "../../src/diverge_model_internals.h"

#ifndef DIVERGE_SKIP_TESTS

diverge_model_t* gen_random(index_t nkx, index_t nky) {
    diverge_model_t* mod = diverge_model_init();
    mod->n_orb = 2;
    mod->n_spin = 2;
    mod->SU2 = false; // enforce exchange symmetry

    random_seed(12345); // the seed is important for reproducibility

    double lattice_norm = 0.0;
    for (int i=0; i<2; ++i)
    for (int j=0; j<2; ++j) {
        mod->lattice[i][j] = fabs(random_dbl());
        lattice_norm += POW2(mod->lattice[i][j]);
    }
    for (int i=0; i<2; ++i) for (int j=0; j<2; ++j) mod->lattice[i][j] /= sqrt(lattice_norm);
    mod->lattice[2][2] = 1.0;
    for (int i=0; i<2; ++i)
    for (int j=0; j<2; ++j)
        mod->positions[i][j] = fabs(random_dbl());

    mod->hop = calloc(1000, sizeof(rs_hopping_t));
    for (int o=0; o<2; ++o)
    for (int s=0; s<2; ++s)
    for (int Rx=-1; Rx<=1; ++Rx)
    for (int Ry=-1; Ry<=1; ++Ry) {
        complex128_t r_c;

        r_c = random_cmplx();
        mod->hop[mod->n_hop++] = (rs_hopping_t){ { Rx, Ry,0}, o,o, s,s, r_c };
        mod->hop[mod->n_hop++] = (rs_hopping_t){ {-Rx,-Ry,0}, o,o, s,s, conj(r_c) };

        r_c = random_cmplx();
        mod->hop[mod->n_hop++] = (rs_hopping_t){ { Rx, Ry,0}, o,!o, s,s, r_c };
        mod->hop[mod->n_hop++] = (rs_hopping_t){ {-Rx,-Ry,0}, !o,o, s,s, conj(r_c) };

        r_c = random_cmplx();
        mod->hop[mod->n_hop++] = (rs_hopping_t){ { Rx, Ry,0}, o,o, s,!s, r_c };
        mod->hop[mod->n_hop++] = (rs_hopping_t){ {-Rx,-Ry,0}, o,o, !s,s, conj(r_c) };

        r_c = random_cmplx();
        mod->hop[mod->n_hop++] = (rs_hopping_t){ { Rx, Ry,0}, o,!o, s,!s, r_c };
        mod->hop[mod->n_hop++] = (rs_hopping_t){ {-Rx,-Ry,0}, !o,o, !s,s, conj(r_c) };
    }

    mod->nk[0] = nkx, mod->nk[1] = nky;
    mod->nkf[0] = 1, mod->nkf[1] = 1;

    mod->vert = calloc(1000, sizeof(rs_vertex_t));

    for (int o=0; o<2; ++o)
    for (int p=0; p<2; ++p)
    for (int Rx=-1; Rx<=1; ++Rx)
    for (int Ry=-1; Ry<=1; ++Ry) {
        complex128_t r_c;
        r_c = random_cmplx();
        mod->vert[mod->n_vert++] = (rs_vertex_t){'D', { Rx, Ry,0}, o,p, -1,0,0,0, r_c};
        r_c = random_cmplx();
        mod->vert[mod->n_vert++] = (rs_vertex_t){'C', { Rx, Ry, 0}, o,p, -1,0,0,0, r_c};
        r_c = random_cmplx();
        mod->vert[mod->n_vert++] = (rs_vertex_t){'P', { Rx, Ry, 0}, o,p, -1,0,0,0, r_c};
    }
    diverge_model_internals_common( mod );
    return mod;
}

#endif // DIVERGE_SKIP_TESTS
