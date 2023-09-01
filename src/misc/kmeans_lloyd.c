#include "kmeans_lloyd.h"

static int h_dkm_use_random_seed = 0;

int dkm_use_random_seed( void ) {
    return h_dkm_use_random_seed;
}

void dkm_set_use_random_seed( int use ) {
    h_dkm_use_random_seed = use;
}
