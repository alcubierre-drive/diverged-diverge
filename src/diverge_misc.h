/* divERGe implements various ERG examples */
/* framework for FRG simulations on real materials */
/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl */

/* This program is free software: you can redistribute it and/or modify */
/* it under the terms of the GNU General Public License as published by */
/* the Free Software Foundation, either version 3 of the License, or */
/* (at your option) any later version. */

/* This program is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/* GNU General Public License for more details. */

/* You should have received a copy of the GNU General Public License */
/* along with this program.  If not, see <https://www.gnu.org/licenses/>. */

#pragma once

// (batched) linear algebra
#include "misc/batched_eigen.h"
#include "misc/batched_svd.h"
#include "misc/batched_gemms.h"
#include "misc/batched_gemm_small.h"
#include "misc/batched_zgemd.h"
#include "misc/eigen.h"
#include "misc/gemm.h"
#include "misc/invert.h"
#include "misc/arnoldi.h"

// Some low dimensional linear algebra in C, implement with putting
// 'LINALG_IMPLEMENT' in your source file. If you want decompositions, use
// #define LINALG_LAPACKE in your source file *before* including this file!
#ifndef LINALG_LAPACKE
#define LINALG_NO_LAPACKE
#endif
#include "misc/linalg.h"

// storing vertices during the flow
#include "misc/vertex_store.h"

// greens function generators
#include "misc/subspace_gf.h"
#include "misc/largemat_gf.h"

// useful functions for model generation
#include "misc/unique_distances.h"
#include "misc/ham2hop.h"
#include "misc/bitary.h"
#include "misc/qnan.h"
#include "misc/diverge_qnan.h"

// density of states
#include "misc/dos.h"

// hartree fock
#include "misc/hartree_fock.h"

// rpa
#include "tu/rpa.h"

