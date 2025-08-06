#include "model_constructor.hpp"
#include "model_constructor.h"

#ifndef DIVERGE_SKIP_TESTS

diverge_model_t* C_gen_square_lat_hub(index_t nkx, index_t nky) {
    return gen_square_lat_hub(nkx, nky);
}
diverge_model_t* C_gen_square_lat_hub_proj(index_t nkx, index_t nky) {
    (void)nky;
    return gen_square_lat_hub_proj(nkx);
}
diverge_model_t* C_gen_random(index_t nkx, index_t nky) {
    return gen_random(nkx, nky);
}
diverge_model_t* C_gen_square_lat_rashba(index_t nkx, index_t nky) {
    return gen_square_lat_rashba(nkx, nky);
}
diverge_model_t* C_gen_honeycomb_lat_hub(index_t nkx, index_t nky) {
    return gen_honeycomb_lat_hub(nkx, nky);
}
diverge_model_t* C_gen_square_lat_hub_sym(index_t nkx, index_t nky) {
    return gen_square_lat_hub_sym(nkx, nky);
}
diverge_model_t* C_gen_square_lat_rashba_sym(index_t nkx, index_t nky) {
    return gen_square_lat_rashba_sym(nkx, nky);
}
diverge_model_t* C_gen_honeycomb_lat_hub_sym(index_t nkx, index_t nky) {
    return gen_honeycomb_lat_hub_sym(nkx, nky);
}
diverge_model_t* C_gen_triangular_pxpy_sym(index_t nkx, index_t nky) {
    (void)nky;
    return gen_triangular_pxpy_sym(nkx, true);
}
diverge_model_t* C_gen_honeycomb_lat_rash(index_t nkx, index_t nky) {
    return gen_honeycomb_lat_rash(nkx, nky);
}
diverge_model_t* C_gen_kagome_model( index_t nkx, index_t nky) {
    return gen_kagome_model( nkx, nky);
}

#endif // DIVERGE_SKIP_TESTS
