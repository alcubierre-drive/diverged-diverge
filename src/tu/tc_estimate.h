/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diverge_flow_step_t diverge_flow_step_t;

/** results of a :c:func:`tu_tc_estimate` call */
typedef struct {
    /** DOS at EF */
    double N0;
    /** EPC strength at Lambda=0 */
    double lambda;
    /** lambda-weighted log average */
    double omega_log;
    /** rms phonon freq */
    double omega_rms;
    /** Coulomb pseudopotential at Lambda_star */
    double mu_star_at_Lstar;
    /** Morel-Anderson rescaled mu_star */
    double mu_star;
    /** McMillan / Allen-Dynes; 0 if there is no solution */
    double Tc;

    /** number of points*bands on the FS */
    index_t n_fs;
    /** pair vertex averaged over FS */
    double vpair_fs;
    /** number of modes skipped due to small freq */
    index_t n_modes_skipped;

    /** check for errors */
    int errors;
} tu_tc_result_t;

/**
 * Estimate :math:`T_c` from Allen-Dynes/MacMillan. Usage: Stop the flow at
 * Lambda* (above the phonon scale). Then mu_star_at_Lstar is N0*<<P[V]>>_FS.
 * Residual Morel-Anderson log from Lambda* to omega_log is applied on top.
 *
 * This works with the TU-phonons backend only, and only if the P channel is
 * included in the flow.
 *
 * The flow must be run with the phonons switched *off*.
 * (:c:func:`diverge_flow_step_set_interchannel`, "ph:nnn"),
 * standard intraband weak-coupling assumptions are taken.
 *
 * :param double Lambda_star:  the scale the flow was stopped at
 * :param double smearing:     Fermi surface smearing (gaussian)
 *
 * :returns tu_tc_result_t: results
 */
tu_tc_result_t tu_tc_estimate( diverge_flow_step_t* step, double Lambda_star, double smearing );

/**
 * change the minimum weight that an FS point has to have in order to be
 * considered as part of the FS. returns the old value. Does not do anything if
 * qnan_isnan(w).
 */
double tu_tc_estimate_change_fs_weight_min( double w );

/**
 * change the minimum phonon frequency that is taken into account. returns the
 * old value. Does not do anything if qnan_isnan(w).
 */
double tu_tc_estimate_change_omega_min( double w );

#ifdef __cplusplus
}
#endif
