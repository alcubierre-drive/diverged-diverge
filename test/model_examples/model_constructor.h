#ifndef DIVERGE_SKIP_TESTS

#pragma once
#include "../../src/diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

diverge_model_t* C_gen_square_lat_hub(index_t nkx, index_t nky);
diverge_model_t* C_gen_square_lat_hub_proj(index_t nk, double scale_tb);
diverge_model_t* C_gen_random(index_t nkx, index_t nky);
diverge_model_t* C_gen_square_lat_rashba(index_t nkx, index_t nky);
diverge_model_t* C_gen_honeycomb_lat_hub(index_t nkx, index_t nky);
diverge_model_t* C_gen_square_lat_hub_sym(index_t nkx, index_t nky);
diverge_model_t* C_gen_square_lat_rashba_sym(index_t nkx, index_t nky);
diverge_model_t* C_gen_honeycomb_lat_hub_sym(index_t nkx, index_t nky);
diverge_model_t* C_gen_triangular_pxpy_sym(index_t nk, bool to_ppm);
diverge_model_t* C_gen_honeycomb_lat_rash(index_t nkx, index_t nky);
diverge_model_t* C_gen_kagome_model( index_t nkx, index_t nky);

#ifdef __cplusplus
}
#endif

#endif // DIVERGE_SKIP_TESTS
