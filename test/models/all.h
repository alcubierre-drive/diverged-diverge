#ifndef DIVERGE_SKIP_TESTS

#pragma once
#include "../../src/diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KWARGS(fun, ...) KWARGS_##fun (&(model_constructor_opts_t){ __VA_ARGS__ })

typedef struct model_constructor_opts_t model_constructor_opts_t;
struct model_constructor_opts_t {
    index_t nkx;
    index_t nky;
    double U;
    double V;
    double VC;
    double VP;
    double alpha;
    double mu;
    double scale_tb;
    double t;
    double tp;
    index_t lx;
    index_t ly;
    index_t nkxf;
    index_t nkyf;
    index_t nsym;
    int to_ppm;
};

#ifndef __cplusplus
static const model_constructor_opts_t model_constructor_opts_defaults = {
    .nkx = 6,
    .nky = 6,
    .U = 3,
    .V = 1,
    .VC = -0.7,
    .VP = 0.5,
    .alpha = 0.5,
    .mu = -0.5,
    .scale_tb = -0.5,
    .t = 1,
    .tp = -0.1,
    .lx = 1,
    .ly = 1,
    .nkxf = 1,
    .nkyf = 1,
    .nsym = 1,
    .to_ppm = 1,
};
#endif //__cplusplus
model_constructor_opts_t model_constructor_opts_defaults_CPP( void );

#define KWARGS_DECL(fun, ...) \
    KWARGS_##fun ( model_constructor_opts_t* kwargs ); \
    diverge_model_t* fun ( __VA_ARGS__ )

diverge_model_t* KWARGS_DECL(gen_square_lat_hub, index_t nkx, index_t nky,
        index_t lx, index_t ly, double t, double tp, double U, double mu, index_t nkxf, index_t nkyf);
diverge_model_t* KWARGS_DECL(gen_square_lat_hub_proj, index_t nkx, index_t nky,
        double scale_tb);
diverge_model_t* KWARGS_DECL(gen_random, index_t nkx, index_t nky);
diverge_model_t* KWARGS_DECL(gen_square_lat_rashba, index_t nkx, index_t nky,
        double t, double tp, double U, double mu, double alpha);
diverge_model_t* KWARGS_DECL(gen_honeycomb_lat_hub, index_t nkx, index_t nky,
        index_t lx, index_t ly, double t, double tp, double U, double mu);
diverge_model_t* KWARGS_DECL(gen_square_lat_hub_sym, index_t nkx, index_t nky,
        index_t nkxf, index_t nkyf, double t, double tp, double U, double mu);
diverge_model_t* KWARGS_DECL(gen_square_lat_rashba_sym, index_t nkx, index_t nky,
        index_t nkxf, index_t nkyf, double t, double tp, double U, double mu, double alpha);
diverge_model_t* KWARGS_DECL(gen_honeycomb_lat_hub_sym, index_t nkx, index_t nky,
        index_t nkxf, index_t nkyf, double t, double tp, double U, double mu,
        index_t nsym);
diverge_model_t* KWARGS_DECL(gen_triangular_pxpy_sym, index_t nkx, index_t nky,
        int to_ppm);
diverge_model_t* KWARGS_DECL(gen_triangular_lat_rashba, index_t nkx, index_t nky);
diverge_model_t* KWARGS_DECL(gen_honeycomb_lat_rash, index_t nkx, index_t nky,
        index_t lx, index_t ly, index_t nkxf, index_t nkyf);
diverge_model_t* KWARGS_DECL(gen_kagome_model, index_t nkx, index_t nky,
        double U, double V, double VC, double VP, index_t nsym);

#ifdef __cplusplus
}
#endif

#endif // DIVERGE_SKIP_TESTS
