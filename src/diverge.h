/* divERGe implements various ERG examples */
/* framework for FRG simulations on real materials */
/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl */

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

#include "misc/mpi_functions.h"
#include "misc/mpi_log.h"
#include "misc/compilation_status.h"
#include "misc/alloc.h"
#include "misc/read_W90.h"
#include "misc/generate_symmetries.h"
#include "misc/kmesh_to_bands.h"
#include "misc/license.h"
#include "misc/shared_mem.h"

#include "diverge_common.h"
#include "diverge_model.h"
#include "diverge_model_internals.h"
#include "diverge_flow_step.h"
#include "diverge_euler.h"
#include "diverge_post.h"

#include "diverge_model_output.h"

#include "diverge_momentum_gen.h"
#include "diverge_filling.h"

#include "diverge_patching.h"
#include "diverge_symmetrize.h"

#include "diverge_run_tests.h"

#include "diverge_hack.h"
