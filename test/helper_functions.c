#include "helper_functions.h"
#include "../src/misc/mpi_log.h"
#include "../src/misc/mpi_functions.h"
#include "../src/misc/mergesort.h"
#include "../src/misc/mt19937.h"

#ifndef DIVERGE_SKIP_TESTS

bool check_hermicity(complex128_t* data, index_t linsize,index_t batches) {
    double tmp = 0.;
    for (index_t k=0; k<batches; ++k)
    for (index_t i=0; i<linsize; ++i)
    for (index_t j=0; j<linsize; ++j)
        tmp += cabs(data[i+linsize*(j+linsize*k)] - conj(data[j+linsize*(i+linsize*k)]));
    return fabs(tmp) < 1e-10;
}

complex128_t trace_cd(complex128_t* data, index_t linsize, index_t batches) {
    complex128_t tmp = 0.;
    for (index_t k=0; k<batches; ++k)
    for (index_t i=0; i<linsize; ++i)
        tmp += data[i+linsize*(i+linsize*k)];
    diverge_mpi_allreduce_complex_sum_inplace(&tmp, 1);
    return tmp;
}
complex128_t trace_batch_lin(complex128_t* data, index_t linsize, index_t batches) {
    complex128_t tmp = 0.;
    for (index_t k=0; k<batches; ++k)
    for (index_t i=0; i<linsize; ++i)
        tmp += data[k+batches*(i+linsize*i)];
    diverge_mpi_allreduce_complex_sum_inplace(&tmp, 1);
    return tmp;
}

double trace_d(double* data, index_t linsize, index_t batches) {
    double tmp = 0.;
    for (index_t k=0; k<batches; ++k)
    for (index_t i=0; i<linsize; ++i)
        tmp += data[i+linsize*(i+linsize*k)];
    diverge_mpi_allreduce_double_sum_inplace(&tmp, 1);
    return tmp;
}

#define Tabs_cd( num ) cabs( num )
#define Tabs_d( num ) fabs( num )
MERGESORT_IMPL( mergesort_index_t, index_t )
static inline index_t unique_index_t( index_t* arr, index_t size ) {
    if (size == 0) return 0;
    index_t j = 0;
    for (index_t i=1; i<size; ++i) if (arr[i] != arr[j]) arr[++j] = arr[i];
    return j+1;
}

#define CHECK_EQUAL_T( T, t ) \
bool check_equal_##t( T* data1, T* data2, index_t size, double errbnd ) { \
    VECTOR_INIT(index_t, wrong_indices); \
    VECTOR_RESERVE(wrong_indices, size/10); \
    for (index_t i=0; i<size; ++i) { \
        double diff = Tabs_##t( data1[i] - data2[i] ); \
        if (diff > errbnd) VECTOR_PUSH_BACK(wrong_indices, i); \
    } \
    int sz = wrong_indices_sz; \
    int comm_sz = diverge_mpi_comm_size(); \
    int* counts = calloc(2*comm_sz, sizeof*counts); \
    int* displs = counts + comm_sz; \
    diverge_mpi_allgather_int( &sz, counts, 1 ); \
    for (int i=1; i<comm_sz; ++i) displs[i] = counts[i-1] + displs[i-1]; \
    \
    if (diverge_mpi_comm_rank() == 0) \
        VECTOR_RESIZE(wrong_indices, counts[comm_sz-1]+displs[comm_sz-1]); \
    diverge_mpi_gatherv_index( wrong_indices, sz, wrong_indices, counts, displs, 0 ); \
    if (diverge_mpi_comm_rank() == 0) { \
        mergesort_index_t( wrong_indices, wrong_indices_sz, diverge_omp_num_threads(), 0 ); \
        wrong_indices_sz = unique_index_t( wrong_indices, wrong_indices_sz ); \
        for (index_t i=0; i<wrong_indices_sz; ++i) { \
            int source_rank = 0; \
            while (displs[source_rank++] < wrong_indices[i]) {if (source_rank==comm_sz) break;} \
            mpi_log_printf( "rank %02d: error at index %d (%.4e)\n", \
                    source_rank, wrong_indices[i], \
                    Tabs_##t(data1[wrong_indices[i]] - data2[wrong_indices[i]]) ); \
        } \
    } \
    int equal = (displs[comm_sz-1] + counts[comm_sz-1]) == 0; \
    VECTOR_FREE(wrong_indices); \
    free( counts ); \
    return equal; \
}
CHECK_EQUAL_T( complex128_t, cd )
CHECK_EQUAL_T( double, d )

MERGESORT_IMPL( mergesort_double, double )
bool check_abs_sorted_equal_d(double* data1, double* data2, index_t size) {
    mergesort_double( data1, size, diverge_omp_num_threads(), 0 );
    mergesort_double( data2, size, diverge_omp_num_threads(), 0 );
    return check_equal_d( data1, data2, size, 1e-10 );
}

#undef MERGESORT_COMP
#define MERGESORT_COMP(x,y) (cabs(x)-cabs(y))
MERGESORT_IMPL( mergesort_complex128_t, complex128_t )
bool check_abs_sorted_equal_cd(complex128_t* data1, complex128_t* data2, index_t size) {
    mergesort_complex128_t( data1, size, diverge_omp_num_threads(), 0 );
    mergesort_complex128_t( data2, size, diverge_omp_num_threads(), 0 );
    return check_equal_cd( data1, data2, size, 1e-10 );
}

complex128_t ff( const double* k, const double* r ) {
    return cexp(-I*(k[0]*r[0]+k[1]*r[1]+k[2]*r[2]));
}
void fill_Rvec_from_vec(const diverge_model_t* model, index_t* R, double* Rvec) {
    for (index_t i=0; i<model->n_tu_ff; ++i)
    for (index_t dim=0; dim<3; ++dim)
        Rvec[3*i+dim] = R[3*i+0] * model->lattice[0][dim] + R[3*i+1] * model->lattice[1][dim]
                                                          + R[3*i+2] * model->lattice[2][dim];
}
void fill_Rvec(const diverge_model_t* model, double* Rvec) {
    const tu_formfactor_t* tu_ff = model->tu_ff;
    for (index_t i=0; i<model->n_tu_ff; ++i)
    for (index_t dim=0; dim<3; ++dim)
        Rvec[3*i+dim] = tu_ff[i].R[0] * model->lattice[0][dim] + tu_ff[i].R[1] * model->lattice[1][dim]
                                                               + tu_ff[i].R[2] * model->lattice[2][dim];
}

static mt19937_64_t mt_rng = {0};
static int mt_rng_has_init = 0;
static uint64_t mt_rng_seed = 1234;

static inline void mt_rng_init( void ) {
    if (!mt_rng_has_init) {
        mt_rng_has_init = 1;
        mt_rng = mt64_init(mt_rng_seed);
    }
}

void random_seed( uint64_t seed ) {
    mt_rng_has_init = 0;
    mt_rng_seed = seed;
}

double random_dbl( void ) {
    #pragma omp critical
    {
        mt_rng_init();
    }
    double r;
    #pragma omp critical
    {
        r = mt64_next_dbl(&mt_rng);
    }
    return 2.*r-1.;
}

complex128_t random_cmplx( void ) {
    return random_dbl() + I*random_dbl();
}

#endif // DIVERGE_SKIP_TESTS
