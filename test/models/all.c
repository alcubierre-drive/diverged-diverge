#include "all.h"
#include <string.h>

model_constructor_opts_t model_constructor_opts_defaults_CPP( void ) {
    return model_constructor_opts_defaults;
}

static inline int kwarg_isset(const char* ptr, unsigned sz) {
    char cmp[sz]; for (unsigned i=0; i<sz; ++i) cmp[i] = 0;
    return memcmp(ptr, cmp, sz);
}

#define KWARG_GET(type, id) \
    type id = (kwarg_isset((const char*)&kwargs->id, sizeof kwargs->id) ? \
            kwargs->id : kwargs_defaults.id)

#define KWARG_WRAP(fun) \
diverge_model_t* KWARGS_##fun ( model_constructor_opts_t* kwargs ) { \
    model_constructor_opts_t kwargs_defaults = model_constructor_opts_defaults; \
    kwargs = kwargs ? kwargs : &kwargs_defaults; \
    KWARG_GET(index_t, nkx); \
    KWARG_GET(index_t, nky);

#define KWARG_WRAP_END }

KWARG_WRAP(gen_square_lat_hub)
    KWARG_GET(index_t, lx);
    KWARG_GET(index_t, ly);
    KWARG_GET(double, t);
    KWARG_GET(double, tp);
    KWARG_GET(double, U);
    KWARG_GET(double, mu);
    KWARG_GET(index_t, nkxf);
    KWARG_GET(index_t, nkyf);
    return gen_square_lat_hub(nkx, nky, lx, ly, t, tp, U, mu, nkxf, nkyf);
KWARG_WRAP_END

KWARG_WRAP(gen_square_lat_hub_proj)
    KWARG_GET(double, scale_tb);
    return gen_square_lat_hub_proj(nkx, nky, scale_tb);
KWARG_WRAP_END

KWARG_WRAP(gen_random)
    return gen_random(nkx, nky);
KWARG_WRAP_END

KWARG_WRAP(gen_square_lat_rashba)
    KWARG_GET(double, t);
    KWARG_GET(double, tp);
    KWARG_GET(double, U);
    KWARG_GET(double, mu);
    KWARG_GET(double, alpha);
    return gen_square_lat_rashba(nkx, nky, t, tp, U, mu, alpha);
KWARG_WRAP_END

KWARG_WRAP(gen_honeycomb_lat_hub)
    KWARG_GET(index_t, lx);
    KWARG_GET(index_t, ly);
    KWARG_GET(double, t);
    KWARG_GET(double, tp);
    KWARG_GET(double, U);
    KWARG_GET(double, mu);
    return gen_honeycomb_lat_hub(nkx, nky, lx, ly, t, tp, U, mu);
KWARG_WRAP_END

KWARG_WRAP(gen_square_lat_hub_sym)
    KWARG_GET(index_t, nkxf);
    KWARG_GET(index_t, nkyf);
    KWARG_GET(double, t);
    KWARG_GET(double, tp);
    KWARG_GET(double, U);
    KWARG_GET(double, mu);
    return gen_square_lat_hub_sym(nkx, nky, nkxf, nkyf, t, tp, U, mu);
KWARG_WRAP_END

KWARG_WRAP(gen_square_lat_rashba_sym)
    KWARG_GET(index_t, nkxf);
    KWARG_GET(index_t, nkyf);
    KWARG_GET(double, t);
    KWARG_GET(double, tp);
    KWARG_GET(double, U);
    KWARG_GET(double, mu);
    KWARG_GET(double, alpha);
    return gen_square_lat_rashba_sym(nkx, nky, nkxf, nkyf, t, tp, U, mu, alpha);
KWARG_WRAP_END

KWARG_WRAP(gen_honeycomb_lat_hub_sym)
    KWARG_GET(index_t, nkxf);
    KWARG_GET(index_t, nkyf);
    KWARG_GET(double, t);
    KWARG_GET(double, tp);
    KWARG_GET(double, U);
    KWARG_GET(double, mu);
    return gen_honeycomb_lat_hub_sym(nkx, nky, nkxf, nkyf, t, tp, U, mu);
KWARG_WRAP_END

KWARG_WRAP(gen_triangular_pxpy_sym)
    KWARG_GET(bool, to_ppm);
    return gen_triangular_pxpy_sym(nkx, nky, to_ppm);
KWARG_WRAP_END

KWARG_WRAP(gen_triangular_lat_rashba)
    return gen_triangular_lat_rashba(nkx, nky);
KWARG_WRAP_END

KWARG_WRAP(gen_honeycomb_lat_rash)
    KWARG_GET(index_t, lx);
    KWARG_GET(index_t, ly);
    KWARG_GET(index_t, nkxf);
    KWARG_GET(index_t, nkyf);
    return gen_honeycomb_lat_rash(nkx, nky, lx, ly, nkxf, nkyf);
KWARG_WRAP_END

KWARG_WRAP(gen_kagome_model)
    KWARG_GET(double, U);
    KWARG_GET(double, V);
    KWARG_GET(double, VC);
    KWARG_GET(double, VP);
    KWARG_GET(index_t, nsym);
    return gen_kagome_model(nkx, nky, U, V, VC, VP, nsym);
KWARG_WRAP_END

