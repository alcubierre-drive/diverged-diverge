#ifndef DIVERGE_SKIP_TESTS


#include "model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../../src/diverge_Eigen3.hpp"
#include "../../src/misc/generate_symmetries.h"
#include <random>
diverge_model_t* gen_random(index_t nkx, index_t nky) {
    diverge_model_t* mod = diverge_model_init();
    mod->n_orb = 2;
    mod->n_spin = 2;
    mod->SU2 = false; // enforce exchange symmetry

    std::mt19937_64 hr_gen(12345);
    uint64_t hr_zero = 0x0;
    uint64_t hr_max = ~hr_zero;
    auto hr_double = [&hr_gen, hr_max]()->double{
        return (double)hr_gen()/(double)hr_max;
    };

    Map<Mat3d> lat((double*)(mod->lattice));
    for (int i=0; i<2; ++i)
    for (int j=0; j<2; ++j)
        lat(i,j) = hr_double();
    lat.block<2,2>(0,0) /= lat.block<2,2>(0,0).norm();
    lat(2,2) = -1.0;
    Map<Mat3d> pos((double*)(mod->positions));
    for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j)
        pos(i,j) = hr_double();
    pos.row(2) *= 0;
    pos.col(2) *= 0;
    if (lat.inverse().norm() > 1.e+5)
        mpi_wrn_printf("not a useful system, lattice is not invertible\n");

    auto rand_real = [&hr_double]()->double{
        return (hr_double() - 0.5)*2;
    };
    auto rand_cmplx = [&rand_real]()->complex128_t{
        return complex128_t(rand_real(), rand_real());
    };

    mod->hop = (rs_hopping_t*)calloc(1000, sizeof(rs_hopping_t));
    for (int o=0; o<2; ++o)
    for (int s=0; s<2; ++s)
    for (int Rx=-1; Rx<=1; ++Rx)
    for (int Ry=-1; Ry<=1; ++Ry) {
        complex128_t r_c;

        r_c = rand_cmplx();
        mod->hop[mod->n_hop++] = rs_hopping_t{ { Rx, Ry,0}, o,o, s,s, r_c };
        mod->hop[mod->n_hop++] = rs_hopping_t{ {-Rx,-Ry,0}, o,o, s,s, std::conj(r_c) };

        r_c = rand_cmplx();
        mod->hop[mod->n_hop++] = rs_hopping_t{ { Rx, Ry,0}, o,!o, s,s, r_c };
        mod->hop[mod->n_hop++] = rs_hopping_t{ {-Rx,-Ry,0}, !o,o, s,s, std::conj(r_c) };

        r_c = rand_cmplx();
        mod->hop[mod->n_hop++] = rs_hopping_t{ { Rx, Ry,0}, o,o, s,!s, r_c };
        mod->hop[mod->n_hop++] = rs_hopping_t{ {-Rx,-Ry,0}, o,o, !s,s, std::conj(r_c) };

        r_c = rand_cmplx();
        mod->hop[mod->n_hop++] = rs_hopping_t{ { Rx, Ry,0}, o,!o, s,!s, r_c };
        mod->hop[mod->n_hop++] = rs_hopping_t{ {-Rx,-Ry,0}, !o,o, !s,s, std::conj(r_c) };

    }

    mod->nk[0] = nkx, mod->nk[1] = nky;
    mod->nkf[0] = 1, mod->nkf[1] = 1;

    mod->vert = (rs_vertex_t*)calloc(1000, sizeof(rs_vertex_t));

    for (int o=0; o<2; ++o)
    for (int p=0; p<2; ++p)
    for (int Rx=-1; Rx<=1; ++Rx)
    for (int Ry=-1; Ry<=1; ++Ry) {
        complex128_t r_c;
        r_c = rand_cmplx();
        if (o != p) r_c = 0.0;
        if (Rx != 0 || Ry != 0) r_c = 0.0;

        mod->vert[mod->n_vert++] = {'D', { Rx, Ry,0}, o,p, -1,0,0,0, 0.0*r_c};
        mod->vert[mod->n_vert++] = {'D', {-Rx,-Ry,0}, o,p, -1,0,0,0, 0.0*std::conj(r_c)};

        mod->vert[mod->n_vert++] = {'D', { Rx, Ry,0}, p,o, -1,0,0,0, 0.0*r_c};
        mod->vert[mod->n_vert++] = {'D', {-Rx,-Ry,0}, p,o, -1,0,0,0, 0.0*std::conj(r_c)};
        r_c = rand_cmplx();
        mod->vert[mod->n_vert++] = {'P', { Rx, Ry,0}, o,p, -1,0,0,0, r_c};
        mod->vert[mod->n_vert++] = {'P', {-Rx,-Ry,0}, p,o, -1,0,0,0, std::conj(r_c)};
    }

    diverge_model_internals_common( mod );
    return mod;
}

#endif // DIVERGE_SKIP_TESTS
