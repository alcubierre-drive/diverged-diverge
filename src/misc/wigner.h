#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_model_t diverge_model_t;

typedef struct {
    index_t nsc;
    double sgn;
    double eps;
    int dim;
} diverge_wigner_config_t;
#ifndef __cplusplus
static const diverge_wigner_config_t diverge_wigner_config_defaults = {
    .nsc = 2, .sgn = -1, .eps = 1.e-7, .dim = -1,
};
#endif
diverge_wigner_config_t diverge_wigner_config_defaults_CPP( void );

diverge_model_t* diverge_phonons_read_ifc( const char* fname, const diverge_wigner_config_t* wc );

#ifdef __cplusplus
}
#endif
