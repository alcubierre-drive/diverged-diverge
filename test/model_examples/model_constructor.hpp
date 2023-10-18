#ifndef DIVERGE_SKIP_TESTS

#pragma once
#include "../../src/diverge_model.h"


diverge_model_t* gen_square_lat_hub(index_t nkx, index_t nky, index_t lx=1,
        index_t ly=1, double t=1, double tp=-0.1, double U=3, double mu=-0.5,
        index_t nkxf = 1, index_t nkyf = 1);

diverge_model_t* gen_square_lat_hub_proj( index_t nk, double scale_tb = -0.5 );

diverge_model_t* gen_random(index_t nkx, index_t nky);

diverge_model_t* gen_square_lat_rashba(index_t nkx=6, index_t nky=6, double t=1,
        double tp=0.1, double U=3, double mu=0.2, double alpha=0.5);

diverge_model_t* gen_honeycomb_lat_hub(index_t nkx, index_t nky, index_t lx=1,
        index_t ly=1, double t=1, double tp=0.1, double U=3, double mu=0.2);

diverge_model_t* gen_square_lat_hub_sym(index_t nkx, index_t nky,
        index_t nkxf=1, index_t nkyf=1,
        double t=1, double tp=-0.1, double U=3, double mu=-0.5);

diverge_model_t* gen_square_lat_rashba_sym(index_t nkx=6, index_t nky=6,
        index_t nkxf=1, index_t nkyf=1, double t=1,
        double tp=0.1, double U=3, double mu=0.2, double alpha=0.5);

diverge_model_t* gen_honeycomb_lat_hub_sym(index_t nkx, index_t nky,
        index_t nkxf=1, index_t nkyf=1,
        double t=1, double tp=0.1, double U=3, double mu=0.2);

diverge_model_t* gen_triangular_pxpy_sym(index_t nk, bool to_ppm);

diverge_model_t* gen_triangular_lat_rashba(index_t nk);

diverge_model_t* gen_honeycomb_lat_rash(index_t nkx, index_t nky, index_t lx=1,
        index_t ly=1, index_t nkxf = 1, index_t nkyf = 1);
diverge_model_t* gen_kagome_model( index_t nk, index_t nkf , double U , double V, double VC, double VP);

#endif // DIVERGE_SKIP_TESTS
