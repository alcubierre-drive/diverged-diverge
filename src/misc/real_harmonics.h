/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SKIP_CLAV25CD_T
typedef struct claV25cd_t { complex128_t v[25]; } claV25cd_t;
#endif
typedef struct claV25cd_t spherical_harmonic_t;
static const double SQRT2 = 1.41421356237309504876;

static const spherical_harmonic_t real_harmonics_s = {{ 1.,
                                0.,0.,0.,
                                0.,0.,0.,0.,0.,
                                0.,0.,0.,0.,0.,0.,0.,
                                0.,0.,0.,0.,0.,0.,0.,0.,0. }};

// TODO do we actually need a minus sign here? Why would phonon bands otherwise
// transform like (px,py,-pz)?
static const spherical_harmonic_t real_harmonics_pz = {{ 0.,
                                  0.,1.,0.,
                                  0.,0.,0.,0.,0.,
                                  0.,0.,0.,0.,0.,0.,0.,
                                  0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_py = {{ 0.,
                                  I128/SQRT2,0,I128/SQRT2,
                                  0.,0.,0.,0.,0.,
                                  0.,0.,0.,0.,0.,0.,0.,
                                  0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_px = {{ 0.,
                                  1./SQRT2,0,-1./SQRT2,
                                  0.,0.,0.,0.,0.,
                                  0.,0.,0.,0.,0.,0.,0.,
                                  0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_dxy = {{ 0.,
                                0.,0.,0.,
                                I128/SQRT2,0.,0.,0.,-I128/SQRT2,
                                0.,0.,0.,0.,0.,0.,0.,
                                0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_dyz = {{ 0.,
                                0.,0.,0.,
                                0.,I128/SQRT2,0,I128/SQRT2,0.,
                                0.,0.,0.,0.,0.,0.,0.,
                                0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_dz2 = {{ 0.,
                                0.,0.,0.,
                                0.,0.,1.,0.,0.,
                                0.,0.,0.,0.,0.,0.,0.,
                                0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_dxz = {{ 0.,
                                0.,0.,0.,
                                0.,1./SQRT2,0,-1./SQRT2,0.,
                                0.,0.,0.,0.,0.,0.,0.,
                                0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_dx2my2 = {{ 0.,
                                      0.,0.,0.,
                                      1./SQRT2,0.,0.,0.,1./SQRT2,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_f_3_m3 = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      I128/SQRT2,0.,0.,0.,0.,0.,I128/SQRT2,
                                      0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_f_3_m2 = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,I128/SQRT2,0.,0.,0.,-I128/SQRT2,0.,
                                      0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_f_3_m1 = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,I128/SQRT2,0.,I128/SQRT2,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_f_3_m0 = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,1.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_f_3_1 = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,1./SQRT2,0.,-1./SQRT2,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_f_3_2 = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,1./SQRT2,0.,0.,0.,1./SQRT2,0.,
                                      0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_f_3_3 = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      1./SQRT2,0.,0.,0.,0.,0.,-1./SQRT2,
                                      0.,0.,0.,0.,0.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_g_4_m4 = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      I128/SQRT2,0.,0.,0.,0.,0.,0.,0.,-I128/SQRT2 }};

static const spherical_harmonic_t real_harmonics_g_4_m3 = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,I128/SQRT2,0.,0.,0.,0.,0.,I128/SQRT2,0. }};

static const spherical_harmonic_t real_harmonics_g_4_m2 = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,0.,I128/SQRT2,0.,0.,0.,-I128/SQRT2,0.,0. }};

static const spherical_harmonic_t real_harmonics_g_4_m1 = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,0.,0.,I128/SQRT2,0.,I128/SQRT2,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_g_4_0  = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,1.,0.,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_g_4_1  = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,0.,0.,1./SQRT2,0.,-1./SQRT2,0.,0.,0. }};

static const spherical_harmonic_t real_harmonics_g_4_2  = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,0.,1./SQRT2,0.,0.,0.,1./SQRT2,0.,0. }};

static const spherical_harmonic_t real_harmonics_g_4_3  = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,1./SQRT2,0.,0.,0.,0.,0.,-1./SQRT2,0. }};

static const spherical_harmonic_t real_harmonics_g_4_4  = {{ 0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      1./SQRT2,0.,0.,0.,0.,0.,0.,0.,1./SQRT2 }};

#ifdef __cplusplus
}
#endif
