#include "unique_distances.h"

#include "../diverge_model.h"
#include "init_internal_libs.h"
#include "qnan.h"
#define LINALG_NO_LAPACKE
#define LINALG_EXPORT static inline
#include "linalg.h"
LINALG_IMPLEMENT

static double unique_distance_equal = DIVERGE_EPS_MESH;
static int unique_distance_relative = 0;

void diverge_model_unique_distances_set_precision( double prec ) {
    if (qnan_isnan(prec)) {
        unique_distance_equal = DIVERGE_EPS_MESH;
        unique_distance_relative = 0;
    } else if (prec >= 0.0) {
        unique_distance_equal = prec;
        unique_distance_relative = 0;
    } else {
        unique_distance_equal = prec;
        unique_distance_relative = 1;
    }
}

static int compare_dbl( const void* a_, const void* b_ );

double* diverge_model_unique_distances( diverge_model_t* m, int Rmax, int dim ) {
    index_t n_dist = dim == 3 ? POW2(m->n_orb) * POW3(2*Rmax+1) :
                     dim == 2 ? POW2(m->n_orb) * POW2(2*Rmax+1) :
                                POW2(m->n_orb) *     (2*Rmax+1);
    double* dist = calloc( n_dist+1, sizeof(double) );
    index_t didx = 0;
    int Rmax_x = Rmax,
        Rmax_y = dim > 1 ? Rmax : 0,
        Rmax_z = dim > 2 ? Rmax : 0;
    #pragma omp parallel for num_threads(diverge_omp_num_threads()) collapse(5)
    for (int Rx=-Rmax_x; Rx<=Rmax_x; ++Rx)
    for (int Ry=-Rmax_y; Ry<=Rmax_y; ++Ry)
    for (int Rz=-Rmax_z; Rz<=Rmax_z; ++Rz)
    for (index_t o=0; o<m->n_orb; ++o)
    for (index_t p=0; p<m->n_orb; ++p) {
        claV3d_t L = claV3d_add( claV3d_add(
                claV3d_scale( claV3d_map( m->lattice[0] ), Rx ),
                claV3d_scale( claV3d_map( m->lattice[1] ), Ry ) ),
                claV3d_scale( claV3d_map( m->lattice[2] ), Rz ) );
        claV3d_t P = claV3d_sub( claV3d_map( m->positions[o] ), claV3d_map( m->positions[p] ) );
        #pragma omp critical
        dist[didx++] = claV3d_norm( claV3d_add( L, P ) );
    }
    qsort( dist, n_dist, sizeof(double), &compare_dbl );

    double equal = unique_distance_equal * (unique_distance_relative ?
            claV3d_norm(claV3d_map(m->lattice[0])) : 1.0);
    index_t last_unique_idx = 0;
    for (index_t idx=1; idx<n_dist; ++idx) {
        if (fabs(dist[idx] - dist[last_unique_idx]) > equal) {
            dist[++last_unique_idx] = dist[idx];
        }
    }

    dist[last_unique_idx] = qnan_gen(0);
    return dist;
}

static int compare_dbl( const void* a_, const void* b_ ) {
    const double *pa = a_, *pb = b_;
    if (*pa > *pb)
        return 1;
    else if (*pa < *pb)
        return -1;
    else
        return 0;
}

index_t diverge_model_unique_distances_length( double* dists ) {
    index_t n_dists = -1; while (!qnan_isnan(dists[++n_dists])) {}
    return n_dists;
}
