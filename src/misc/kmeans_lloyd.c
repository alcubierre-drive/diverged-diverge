#include <assert.h>
#include <time.h>

#define LINALG_NO_LAPACKE
#define LINALG_EXPORT static inline
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

typedef struct { VECTOR_DECL( double, v ) } vecD_t;

/* Calculate the smallest distance between each of the data points and any of the input means. */
vecD_t closest_distance( const vecVec2d_t means, const vecVec2d_t data) {
    vecD_t distances;
    VECTOR_DECL_INIT_CAP( double, distances.v, data.v_sz );
    for (index_t i=0; i<data.v_sz; ++i) {
        double closest = claV2d_norm2(claV2d_sub(data.v[i],means.v[0]));
        for (index_t j=0; j<means.v_sz; ++j) {
            double distance = claV2d_norm2(claV2d_sub(data.v[i],means.v[j]));
            if (distance < closest)
                closest = distance;
        }
        VECTOR_PUSH_BACK( distances.v, closest );
    }
    return distances;
}

// Using a very simple PRBS generator, parameters selected according to
// https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
static inline uint64_t LCG_gen( uint64_t* state ) {
    #define LCG_A 6364136223846793005
    #define LCG_C 1442695040888963407
    #define LCG_M UINT64_MAX
    return *state = (LCG_A * (*state) + LCG_C) % LCG_M;
}

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

/* This is an alternate initialization method based on the
 * [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) initialization
 * algorithm. */
vecVec2d_t random_plusplus(const vecVec2d_t data, uint32_t k, uint64_t seed) {
    assert(k > 0);
    assert(data.v_sz > 0);
    vecVec2d_t means;
    VECTOR_DECL_INIT( claV2d_t, means.v );

    // Select first mean at random from the set
    VECTOR_PUSH_BACK( means.v, data.v[LCG_gen(&seed)%data.v_sz] );

    for (uint32_t count=1; count<k; ++count) {
        // Calculate the distance to the closest mean for each data point
        vecD_t distances = closest_distance( means, data );
        // Pick a random point weighted by the distance from existing means,
        // calc cumsum first
        for (index_t i=1; i<distances.v_sz; ++i)
            distances.v[i] += distances.v[i-1];

        index_t gen = find_bin_irreg( distances.v, distances.v_sz,
                (double)LCG_gen(&seed)/(double)LCG_M*distances.v[distances.v_sz-1] );
        if (gen <= 0) gen = 0;
        if (gen >= distances.v_sz) gen = distances.v_sz-1;
        VECTOR_PUSH_BACK( means.v, data.v[gen] );
        VECTOR_FREE( distances.v );
    }
    return means;
}

/* Calculate the index of the mean a particular data point is closest to (euclidean distance) */
uint32_t closest_mean(claV2d_t point, const vecVec2d_t means) {
    assert(means.v_sz > 0);
    double smallest_distance = claV2d_norm2(claV2d_sub(point,means.v[0]));
    index_t index = 0;
    double distance;
    for (index_t i=1; i<means.v_sz; ++i) {
        distance = claV2d_norm2(claV2d_sub(point,means.v[i]));
        if (distance < smallest_distance) {
            smallest_distance = distance;
            index = i;
        }
    }
    return index;
}

/* Calculate the index of the mean each data point is closest to (euclidean distance). */
vecU32_t calculate_clusters(const vecVec2d_t data, const vecVec2d_t means) {
    vecU32_t clusters;
    VECTOR_DECL_INIT_CAP( uint32_t, clusters.v, data.v_sz );
    for (index_t i=0; i<data.v_sz; ++i)
        VECTOR_PUSH_BACK( clusters.v, closest_mean(data.v[i], means) );
    return clusters;
}

/* Calculate means based on data points and their cluster assignments. */
vecVec2d_t calculate_means(const vecVec2d_t data, const vecU32_t clusters,
        const vecVec2d_t old_means, uint32_t k) {
    vecVec2d_t means = {0}; VECTOR_RESIZE(means.v, k);
    vecU32_t count = {0}; VECTOR_RESIZE(count.v, k);
    memset( means.v, 0, sizeof*means.v * means.v_sz );
    memset( count.v, 0, sizeof*count.v * count.v_sz );

    for (index_t i=0; i<MIN(clusters.v_sz,data.v_sz); ++i) {
        count.v[clusters.v[i]]++;
        means.v[clusters.v[i]] = claV2d_add( means.v[clusters.v[i]], data.v[i] );
    }
    for (index_t i=0; i<k; ++i) {
        if (count.v[i] == 0)
            means.v[i] = old_means.v[i];
        else
            means.v[i] = claV2d_scale(means.v[i], 1./(double)count.v[i]);
    }
    VECTOR_FREE( count.v );
    return means;
}

vecD_t deltas( const vecVec2d_t old_means, const vecVec2d_t means ) {
    vecD_t distances = {0};
    VECTOR_RESIZE( distances.v, means.v_sz );
    assert(old_means.v_sz == means.v_sz);
    for (index_t i=0; i<means.v_sz; ++i)
        distances.v[i] = claV2d_norm(claV2d_sub(means.v[i],old_means.v[i]));
    return distances;
}

int deltas_below_limit( const vecD_t deltas, double min_delta ) {
    for (index_t i=0; i<deltas.v_sz; ++i) if (deltas.v[i] > min_delta) return 0;
    return 1;
}

int deltas_below_limit_diff( const vecVec2d_t old_means, const vecVec2d_t means, double min_delta ) {
    vecD_t del = deltas( old_means, means );
    int res = deltas_below_limit( del, min_delta );
    VECTOR_FREE( del.v );
    return res;
}

dkm_result_t kmeans_lloyd( const vecVec2d_t data, uint32_t k, double limit, index_t maxiter ) {
    uint64_t seed = random_seed;
    vecVec2d_t means = random_plusplus( data, k, seed );

    vecVec2d_t old_means = {0}, old_old_means = {0};
    vecU32_t clusters = {0};
    index_t count = 0;
    do {
        VECTOR_FREE( clusters.v );
        clusters = calculate_clusters( data, means );
        VECTOR_FREE( old_old_means.v );
        old_old_means = old_means;
        old_means = means;
        means = calculate_means( data, clusters, old_means, k );
        count++;
    } while ((!deltas_below_limit_diff( old_means, means, limit ) ||
              !deltas_below_limit_diff( old_old_means, old_means, limit )) &&
             (count < maxiter || maxiter < 0));
    VECTOR_FREE( old_means.v );
    VECTOR_FREE( old_old_means.v );

    return (dkm_result_t){ means, clusters };
}

void dkm_result_free( dkm_result_t r ) {
    VECTOR_FREE( r.points.v );
    VECTOR_FREE( r.clusters.v );
}
