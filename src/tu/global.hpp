#pragma once

#include <cmath>
#include <complex>
#include <math.h>
#include "../diverge_Eigen3.hpp"
#include "../diverge_common.h"
#include <stdio.h>

#include <vector>
#include <array>

using std::vector;
using std::array;
using std::string;
using std::stringstream;

const double abs_err_integrator = 1e-3, rel_err_integrator = 1e-2;
const double absEps = 1e-12, relEps = 1e-10;
const double safety_factor = 0.9;

#include "global_allocation.hpp"
#include "global_macros.hpp"

typedef unsigned int uint;
