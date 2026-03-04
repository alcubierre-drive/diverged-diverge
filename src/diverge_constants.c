#include "diverge_constants.h"
#include "misc/generate_symmetries.h"

diverge_constants_t diverge_constants( void ) {
    return (diverge_constants_t){
        ._MAX_N_ORBS = MAX_N_ORBS,
        ._USE_GF_FLOATS = sizeof(gf_complex_t) != sizeof(complex128_t),
        ._MAX_NAME_LENGTH = MAX_NAME_LENGTH,
        ._MAX_N_SYM = MAX_N_SYM,
        ._MAX_ORBS_PER_SITE = MAX_ORBS_PER_SITE,
        ._DIVERGE_EPS_MESH = DIVERGE_EPS_MESH,
    };
}
