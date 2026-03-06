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

void diverge_constants_print( void ) {
    diverge_constants_t c = diverge_constants();
    printf( "-DMAX_N_ORBS=%li ", c._MAX_N_ORBS );
    printf( "-%cUSE_GF_FLOATS ", c._USE_GF_FLOATS ? 'D' : 'U' );
    printf( "-DMAX_NAME_LENGTH=%li ", c._MAX_NAME_LENGTH );
    printf( "-DMAX_N_SYM=%li ", c._MAX_N_SYM );
    printf( "-DMAX_ORBS_PER_SITE=%li ", c._MAX_ORBS_PER_SITE );
    printf( "-DDIVERGE_EPS_MESH=%.5e\n", c._DIVERGE_EPS_MESH );
}
