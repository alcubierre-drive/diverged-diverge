/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_euler.h"

bool diverge_euler_next( diverge_euler_t* de, double Vmax ) {

    bool abort = false;

    // only check for vertex max if we are above the limit where we start
    bool lambda_allows_maxvert = (de->Lambda <= de->consider_maxvert_lambda) || (de->consider_maxvert_lambda < 0);

    if ((de->niter >= de->consider_maxvert_iter_start) && lambda_allows_maxvert && (Vmax > de->maxvert))
        abort = true;
    if (de->Lambda < de->Lambda_min)
        abort = true;
    if (Vmax >= de->maxvert_hard_limit)
        abort = true;

    double dLambda_new = de->dLambda_fac_scale * de->Lambda / Vmax;
    if (dLambda_new > de->dLambda_fac * de->Lambda)
        dLambda_new = de->dLambda_fac * de->Lambda;
    if (dLambda_new < de->dLambda_min)
        dLambda_new = de->dLambda_min;

    de->niter++;
    if (de->maxiter >= 0 && de->niter >= de->maxiter)
        abort = true;

    if (!abort)
        de->Lambda += de->dLambda;

    de->dLambda = -dLambda_new;

    return !abort;
}

diverge_euler_t diverge_euler_defaults_CPP( void ) {
    return diverge_euler_defaults;
}
