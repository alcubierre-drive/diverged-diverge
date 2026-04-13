#include <assert.h>
#include <time.h>

#define LINALG_NO_LAPACKE
#include "linalg.h"
LINALG_IMPLEMENT

#define SKIP_CLAV2D_T
#include "kmeans_lloyd.h"

#include "mt19937.h"
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

static inline double dist2(claV2d_t a, claV2d_t b) { return claV2d_norm2(claV2d_sub(a, b)); }
static index_t closest_mean(claV2d_t pt, const claV2d_t* means, index_t k) {
    index_t best = 0;
    double best_d = dist2(pt, means[0]);

    for (index_t i=1; i<k; ++i) {
        double d = dist2(pt, means[i]);
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    return best;
}

// assign all points
static void assign_clusters(index_t* clusters, const claV2d_t* pts, index_t n,
        const claV2d_t* means, index_t k) {
    for (index_t i=0; i<n; ++i)
        clusters[i] = closest_mean(pts[i], means, k);
}

// recompute means
static void recompute_means(claV2d_t* means, const claV2d_t* pts, index_t n,
        const index_t* clusters, index_t k, const claV2d_t* old_means) {
    claV2d_t* sum = calloc(k, sizeof(*sum));
    index_t* count = calloc(k, sizeof(*count));

    for (index_t i=0; i<n; ++i) {
        index_t c = clusters[i];
        sum[c] = claV2d_add(sum[c], pts[i]);
        count[c]++;
    }

    for (index_t i=0; i<k; ++i) {
        if (count[i] > 0) {
            means[i] = claV2d_scale(sum[i], 1.0/(double)count[i]);
        } else {
            // keep previous mean if cluster empty
            means[i] = old_means[i];
        }
    }

    free(sum);
    free(count);
}

// max movement between means
static double maxdiff(const claV2d_t* a, const claV2d_t* b, index_t k) {
    double maxd = 0.0;
    for (index_t i = 0; i < k; ++i) {
        double d = dist2(a[i], b[i]);
        if (d > maxd) maxd = d;
    }
    return sqrt(maxd);
}

static void kmeans_pp(claV2d_t* means, const claV2d_t* pts, index_t n, index_t k, uint64_t seed) {
    mt19937_64_t rng = mt64_init(seed);

    double* dist = malloc(n * sizeof(*dist));

    // first center random
    means[0] = pts[mt64_next(&rng) % n];

    for (index_t c=1; c<k; ++c) {

        // compute distance to nearest existing center
        double total = 0.0;

        for (index_t i=0; i<n; ++i) {
            double best = dist2(pts[i], means[0]);
            for (index_t j=1; j<c; ++j) {
                double d = dist2(pts[i], means[j]);
                if (d < best) best = d;
            }
            dist[i] = best;
            total += best;
        }

        // sample from weighted distribution
        double r = mt64_next_dbl(&rng) * total;

        double cumsum = 0.0;
        index_t chosen = n - 1;

        for (index_t i = 0; i < n; ++i) {
            cumsum += dist[i];
            if (cumsum >= r) {
                chosen = i;
                break;
            }
        }

        means[c] = pts[chosen];
    }

    free(dist);
}

dkm_result_t kmeans_lloyd(const claV2d_t* pts, index_t n, index_t k, double tol,
        index_t maxiter) {

    claV2d_t* means = calloc(k, sizeof(*means));
    claV2d_t* old   = calloc(k, sizeof(*old));
    index_t* clusters = malloc(n * sizeof(*clusters));

    // init
    kmeans_pp(means, pts, n, k, random_seed);

    index_t iter = 0;

    while (1) {
        memcpy(old, means, k * sizeof(*means));

        assign_clusters(clusters, pts, n, means, k);
        recompute_means(means, pts, n, clusters, k, old);

        double diff = maxdiff(means, old, k);

        iter++;
        if (diff <= tol) break;
        if (maxiter >= 0 && iter >= maxiter) break;
    }

    free(old);

    return (dkm_result_t){
        .points = means,
        .clusters = clusters,
        .k = k
    };
}

void dkm_result_free(dkm_result_t r) {
    free(r.points);
    free(r.clusters);
}
