/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../diverge_common.h"

#ifndef __cplusplus
typedef struct {
    complex128_t _data[25];
} spherical_harmonic_t;
#else
#include "../diverge_Eigen3.hpp"
typedef Matrix<complex128_t,25,1> spherical_harmonic_t;
extern "C" {
#endif


static const spherical_harmonic_t real_harmonics_s = {1.,
                                0.,0.,0.,
                                0.,0.,0.,0.,0.,
                                0.,0.,0.,0.,0.,0.,0.,
                                0.,0.,0.,0.,0.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_pz = {0.,
                                  0.,1.,0.,
                                  0.,0.,0.,0.,0.,
                                  0.,0.,0.,0.,0.,0.,0.,
                                  0.,0.,0.,0.,0.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_py = {0.,
                                  I128/sqrt(2.),0,I128/sqrt(2.),
                                  0.,0.,0.,0.,0.,
                                  0.,0.,0.,0.,0.,0.,0.,
                                  0.,0.,0.,0.,0.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_px = {0.,
                                  1./sqrt(2.),0,-1./sqrt(2.),
                                  0.,0.,0.,0.,0.,
                                  0.,0.,0.,0.,0.,0.,0.,
                                  0.,0.,0.,0.,0.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_dxy = {0.,
                                0.,0.,0.,
                                I128/sqrt(2.),0.,0.,0.,-I128/sqrt(2.),
                                0.,0.,0.,0.,0.,0.,0.,
                                0.,0.,0.,0.,0.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_dyz = {0.,
                                0.,0.,0.,
                                0.,I128/sqrt(2.),0,I128/sqrt(2.),0.,
                                0.,0.,0.,0.,0.,0.,0.,
                                0.,0.,0.,0.,0.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_dz2 = {0.,
                                0.,0.,0.,
                                0.,0.,1.,0.,0.,
                                0.,0.,0.,0.,0.,0.,0.,
                                0.,0.,0.,0.,0.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_dxz = {0.,
                                0.,0.,0.,
                                0.,1./sqrt(2.),0,-1./sqrt(2.),0.,
                                0.,0.,0.,0.,0.,0.,0.,
                                0.,0.,0.,0.,0.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_dx2my2 = {0.,
                                      0.,0.,0.,
                                      1./sqrt(2.),0.,0.,0.,1./sqrt(2.),
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.};


static const spherical_harmonic_t real_harmonics_f_3_m3 = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      I128/sqrt(2.),0.,0.,0.,0.,0.,I128/sqrt(2.),
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_f_3_m2 = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,I128/sqrt(2.),0.,0.,0.,-I128/sqrt(2.),0.,
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_f_3_m1 = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,I128/sqrt(2.),0.,I128/sqrt(2.),0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_f_3_m0 = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,1.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_f_3_1 = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,1./sqrt(2.),0.,-1./sqrt(2.),0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_f_3_2 = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,1./sqrt(2.),0.,0.,0.,1./sqrt(2.),0.,
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_f_3_3 = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      1./sqrt(2.),0.,0.,0.,0.,0.,-1./sqrt(2.),
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.};


static const spherical_harmonic_t real_harmonics_g_4_m4 = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      I128/sqrt(2.),0.,0.,0.,0.,0.,0.,0.,-I128/sqrt(2.)};

static const spherical_harmonic_t real_harmonics_g_4_m3 = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,I128/sqrt(2.),0.,0.,0.,0.,0.,I128/sqrt(2.),0.};

static const spherical_harmonic_t real_harmonics_g_4_m2 = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,0.,I128/sqrt(2.),0.,0.,0.,-I128/sqrt(2.),0.,0.};

static const spherical_harmonic_t real_harmonics_g_4_m1 = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,0.,0.,I128/sqrt(2.),0.,I128/sqrt(2.),0.,0.,0.};

static const spherical_harmonic_t real_harmonics_g_4_0  = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,1.,0.,0.,0.,0.};

static const spherical_harmonic_t real_harmonics_g_4_1  = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,0.,0.,1./sqrt(2.),0.,-1./sqrt(2.),0.,0.,0.};

static const spherical_harmonic_t real_harmonics_g_4_2  = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,0.,1./sqrt(2.),0.,0.,0.,1./sqrt(2.),0.,0.};

static const spherical_harmonic_t real_harmonics_g_4_3  = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      0.,1./sqrt(2.),0.,0.,0.,0.,0.,-1./sqrt(2.),0.};

static const spherical_harmonic_t real_harmonics_g_4_4  = {0.,
                                      0.,0.,0.,
                                      0.,0.,0.,0.,0.,
                                      0.,0.,0.,0.,0.,0.,0.,
                                      1./sqrt(2.),0.,0.,0.,0.,0.,0.,0.,1./sqrt(2.)};

#ifdef __cplusplus
}
#endif
