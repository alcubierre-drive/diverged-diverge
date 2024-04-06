/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_euler_t diverge_euler_t;
/**
 * simple structure to support user-driven Euler integration with adaptive step
 * size
 */
struct diverge_euler_t {
    /** current scale :math:`\Lambda` */
    double Lambda;
    /** current differential scale :math:`d_\Lambda` */
    double dLambda;
    /** minimal scale :math:`\Lambda_\mathrm{min}` */
    double Lambda_min;
    /** absolute minimal differentail scale :math:`d_{\Lambda,\mathrm{min}}` */
    double dLambda_min;
    /** :math:`d_{\Lambda,\mathrm{fac}}` TODO: see arXiv how to document */
    double dLambda_fac;
    /** :math:`d_{\Lambda,\mathrm{fac-scale}}` TODO: see arXiv how to document */
    double dLambda_fac_scale;
    /** maximum vertex value */
    double maxvert;
    /** hard limit for maximum vertex value */
    double maxvert_hard_limit;

    /** number of iterations */
    index_t niter;
    /** maximum number of iterations */
    index_t maxiter;
    /** consider :c:member:`diverge_euler_t.maxvert` only after this iteration */
    index_t consider_maxvert_iter_start;
    /**
     * consider :c:member:`diverge_euler_t.maxvert` only after passing this
     * value of :math:`\Lambda`
     */
    double consider_maxvert_lambda;
};

/** returns default values for the Euler integrator in C/C++/Python */
diverge_euler_t diverge_euler_defaults_CPP( void );

/**
 * performs adaptive step, without calling any :c:type:`diverge_flow_step_t`
 * related routines. example usage:
 *
 * .. sourcecode:: C
 *
 *  // diverge_flow_step_t* st = ...;
 *  diverge_euler_t eu = diverge_euler_defaults_CPP();
 *  double vertmax = 0.0;
 *  do {
 *      diverge_flow_step_euler( st, eu.Lambda, eu.dLambda );
 *      diverge_flow_step_vertmax( st, &vertmax );
 *  } while (diverge_euler_next(&eu, vertmax));
 */
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
