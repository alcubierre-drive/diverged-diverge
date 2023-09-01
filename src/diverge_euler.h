#pragma once

#include "diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_euler_t diverge_euler_t;
// Class: diverge_euler_t
struct diverge_euler_t {
    // Variable: Lambda
    double Lambda;
    // Variable: dLambda
    double dLambda;
    // Variable: Lambda_min
    double Lambda_min;
    // Variable: dLambda_min
    double dLambda_min;
    // Variable: dLambda_fac
    double dLambda_fac;
    // Variable: dLambda_fac_scale
    double dLambda_fac_scale;
    // Variable: maxvert
    double maxvert;
    // Variable: maxvert_hard_limit
    double maxvert_hard_limit;

    // Variable: niter
    index_t niter;
    // Variable: maxiter
    index_t maxiter;
    // Variable: consider_maxvert_iter_start
    index_t consider_maxvert_iter_start;
    // Variable: consider_maxvert_lambda
    double consider_maxvert_lambda;
};

// Function: diverge_euler_defaults_CPP
// returns default values for the Euler integrator in C/C++/Python
diverge_euler_t diverge_euler_defaults_CPP( void );
// Function: diverge_euler_next
// performs adaptive step, without calling any <diverge_flow_step_t> related
// routines. example usage:
// --- C/C++ ---
// // diverge_flow_step_t* st = ...;
// diverge_euler_t eu = diverge_euler_defaults_CPP();
// double vertmax = 0.0;
// do {
//     diverge_flow_step_euler( st, eu.Lambda, eu.dLambda );
//     diverge_flow_step_vertmax( st, &vertmax );
// } while (diverge_euler_next(&eu, vertmax));
// -------------
bool diverge_euler_next( diverge_euler_t* de, double Vmax );

#if !defined(__cplusplus) && !defined(CTYPESGEN)
// defaults in plain C
static const diverge_euler_t diverge_euler_defaults = {
    .Lambda = 50.0,
    .dLambda = -5.0,
    .Lambda_min = 1.e-5,
    .dLambda_min = 1.e-6,
    .dLambda_fac = 0.1,
    .dLambda_fac_scale = 1.0,
    .maxvert = 50.0,
    .maxvert_hard_limit = 1.e+4,
    .niter = 0,
    .maxiter = -1,
    .consider_maxvert_iter_start = -1,
    .consider_maxvert_lambda = -1.0
};
#endif

#ifdef __cplusplus
}
#endif
