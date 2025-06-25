/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include <assert.h>
#include <time.h>

#define LINALG_NO_LAPACKE
#include "linalg.h"
LINALG_IMPLEMENT

#define SKIP_CLAV2D_T
#include "kmeans_lloyd.h"

#define RANDOM_SEED 123456789ul
static uint64_t random_seed = RANDOM_SEED;
static int helper_use_random_seed = 0;
int dkm_use_random_seed( void ) {
    return helper_use_random_seed;
}
void dkm_set_use_random_seed( int use ) {
    helper_use_random_seed = use;
    if (helper_use_random_seed) {
        srand(time(NULL));
        random_seed = ((uint64_t)rand()) << 32 | (rand());
        mpi_vrb_printf( "kmeans_lloyd using random seed %lu\n", random_seed );
    } else {
        random_seed = RANDOM_SEED;
    }
}

static inline uint64_t LCG_gen( uint64_t* state ); // linear congruent generator
static inline double LCG_genf( uint64_t* state ); // fp64 version
static inline index_t find_bin_irreg( const double* vals, index_t n_vals, double test ); // binary search

// implementation functions
static void closest_distances( double* distances, const claV2d_t* means, index_t k,
        const claV2d_t* pts, index_t n_pts );
static void random_plusplus( claV2d_t* means, const claV2d_t* pts, index_t n_pts,
        index_t k, uint64_t seed );
static inline index_t closest_mean( claV2d_t pt, const claV2d_t* means, index_t k );
static void calc_clusters( index_t* clusters, const claV2d_t* pts, index_t n_pts,
        const claV2d_t* means, index_t k );
static void calculate_means( claV2d_t* means, const claV2d_t* pts, index_t n_pts,
        const index_t* clusters, index_t k, const claV2d_t* old_means );
static double means_maxdiff( const claV2d_t* means1, const claV2d_t* means2, index_t k );

// Using a very simple PRBS generator, parameters selected according to
// https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
static inline uint64_t LCG_gen( uint64_t* state ) {
    #define LCG_A 6364136223846793005
    #define LCG_C 1442695040888963407
    #define LCG_M UINT64_MAX
    return *state = (LCG_A * (*state) + LCG_C) % LCG_M;
}
// floating point version
static inline double LCG_genf( uint64_t* state ) {
    return LCG_gen( state );
}

// binary search
static inline index_t find_bin_irreg( const double* vals, index_t n_vals, double test ) {
    if (test >= vals[n_vals-1]) return n_vals-1;
    if (test <= vals[0]) return -1;
    index_t low = 0,
            high = n_vals-1;
    while (low <= high) {
        index_t mid = low + (high - low)/2;
        if (vals[mid] <= test)
            low = mid + 1;
        else
            high = mid -1;
    }
    return low - 1;
}


// get the squared distance to the closest mean for each point
static void closest_distances( double* distances, const claV2d_t* means, index_t k,
        const claV2d_t* pts, index_t n_pts ) {
    for (index_t p=0; p<n_pts; ++p) {
        for (index_t j=0; j<k; ++j) {
            double dist2 = claV2d_norm2(claV2d_sub(pts[p], means[j]));
            distances[p] = (j==0) ? dist2 : MIN(distances[p], dist2);
        }
    }
}

// initialize means with random points that are distributed by maximal distance
static void random_plusplus( claV2d_t* means, const claV2d_t* pts, index_t n_pts,
        index_t k, uint64_t seed ) {
    double* distances = calloc(n_pts, sizeof*distances);
    for (index_t c=0; c<k; ++c) {
        // first mean is random
        if (!c) {
            means[c] = pts[LCG_gen(&seed)%n_pts];
        } else {
            closest_distances( distances, means, c+1, pts, n_pts );
            // cumsum for the distribution
            double distances_tot = 0;
            for (index_t p=0; p<n_pts; ++p) {
                double dist = distances[p];
                distances[p] = distances_tot;
                distances_tot += dist;
            }
            // find random new index weighted by distances
            index_t gen = find_bin_irreg( distances, n_pts, LCG_genf(&seed)/(double)LCG_M*distances_tot );
            if (gen <= 0) { gen = 0; } if (gen >= n_pts) { gen = n_pts-1; }
            means[c] = pts[gen];
        }
    }
    free( distances );
}

// Calculate the index of the mean a particular data point is closest to (euclidean distance)
static inline index_t closest_mean( claV2d_t pt, const claV2d_t* means, index_t k ) {
    if (k <= 0) return 0;
    double dist = claV2d_norm2(claV2d_sub(pt,means[0]));
    index_t idx = 0;
    for (index_t i=1; i<k; ++i) {
        double t_dist = claV2d_norm2(claV2d_sub(pt,means[i]));
        if (t_dist < dist) {
            dist = t_dist;
            idx = i;
        }
    }
    return idx;
}

// Calculate the index of the mean each data point is closest to (euclidean distance)
static void calc_clusters( index_t* clusters, const claV2d_t* pts, index_t n_pts,
        const claV2d_t* means, index_t k ) {
    for (index_t p=0; p<n_pts; ++p)
        clusters[p] = closest_mean( pts[p], means, k );
}

// Calculate means based on data points and their cluster assignments
static void calculate_means( claV2d_t* means, const claV2d_t* pts, index_t n_pts,
        const index_t* clusters, index_t k, const claV2d_t* old_means ) {
    index_t* count = calloc(k, sizeof*count);
    memset( means, 0, sizeof*means * k );

    for (index_t p=0; p<n_pts; ++p) {
        count[clusters[p]]++;
        means[clusters[p]] = claV2d_add( means[clusters[p]], pts[p] );
    }
    for (index_t i=0; i<k; ++i)
        means[i] = (count[i] == 0) ? old_means[i] : claV2d_scale(means[i], 1./(double)count[i]);

    free( count );
}

// find maximum difference between means1[...] and means2[...]
static double means_maxdiff( const claV2d_t* means1, const claV2d_t* means2, index_t k ) {
    if (k <= 0) return 0.0;
    double diff = claV2d_norm2(claV2d_sub(means1[0],means2[0]));
    for (index_t i=1; i<k; ++i) {
        double t_diff = claV2d_norm2(claV2d_sub(means1[i],means2[i]));
        if (t_diff < diff) diff = t_diff;
    }
    return sqrt(diff);
}

// implementation of the algorithm, no fuzz
dkm_result_t kmeans_lloyd( const claV2d_t* pts, index_t n_pts, index_t k,
        double limit, index_t maxiter ) {
    claV2d_t* means[] = { calloc(k, sizeof(claV2d_t)),
                          calloc(k, sizeof(claV2d_t)),
                          calloc(k, sizeof(claV2d_t)) };
    index_t* clusters = calloc(n_pts, sizeof*clusters);
    index_t iter = 0;

    // initialize means random
    random_plusplus( means[0], pts, n_pts, k, random_seed );
    do {
        // calculate clusters
        calc_clusters( clusters, pts, n_pts, means[0], k );
        // rotate ptrs s.t. means[2] is oldold, means[1] is old, and means[0] is fresh
        XORSWAP_PTR( means[2], means[0] ); XORSWAP_PTR( means[1], means[2] );
        calculate_means( means[0], pts, n_pts, clusters, k, means[1] );
        iter++;
    } while ( ( (means_maxdiff(means[0],means[1],k) > limit)||
                (means_maxdiff(means[1],means[2],k) > limit) ) &&
              ( (iter < maxiter)||(maxiter < 0) ) );
    free( means[1] );
    free( means[2] );
    return (dkm_result_t){ means[0], clusters, k };
}

void dkm_result_free( dkm_result_t x ) {
    free( x.points );
    free( x.clusters );
}
