#include "dos.h"
#include "../diverge_model.h"
#include "../diverge_model_internals.h"
#include "../diverge_internals_struct.h"
#include "mpi_functions.h"
#include "shared_mem.h"
#include <omp.h>
#include "mergesort.h"
#include "dos.cu.h"

static inline index_t find_bin_irreg( const double* vals, index_t n_vals, double test );
static inline double cabs2( complex128_t U );
static inline index_t regularize( index_t idx, index_t sz );
static double eta_bin_factor = 15.0;
#ifdef USE_CUDA
static int use_cuda_dos_algo = 1;
#else
static int use_cuda_dos_algo = 0;
#endif

static inline void ldos_kernel_cpu( const double* E, index_t nE, const complex128_t* U, index_t nb,
        const double* omega, index_t n_omega, double eta, double eta_bin, double* ldos ) {
    // argsort
    index_t* E_sorter = shared_calloc( nE, sizeof*E_sorter );
    if (shared_exclusive_enter( E_sorter ) == 1) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<nE; ++i) E_sorter[i] = i;
        _dos_sort_doubles_indirect_( E_sorter, nE, E );
    }
    shared_exclusive_wait( E_sorter );

    // actual sorting via argsort
    double* E_sorted = shared_calloc( nE, sizeof*E_sorted );
    complex128_t* U_sorted = shared_calloc( nE*nb, sizeof*U_sorted );
    int enter_E = shared_exclusive_enter( E_sorted ) == 1,
        enter_U = shared_exclusive_enter( U_sorted ) == 1;
    if (enter_E && enter_U) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<nE; ++i) {
            E_sorted[i] = E[E_sorter[i]];
            for (index_t o=0; o<nb; ++o)
                U_sorted[IDX2(i,o,nb)] = U ? U[IDX2(E_sorter[i],o,nb)] : 1.0;
        }
    }
    shared_exclusive_wait( E_sorted );
    shared_exclusive_wait( U_sorted );
    shared_free( E_sorter );

    // distribute frequencies
    index_t nthr = diverge_omp_num_threads();
    index_t* w_count = diverge_distribute( n_omega, nthr );
    index_t* w_displ = w_count + nthr;

    #pragma omp parallel num_threads(nthr)
    {
        index_t t = omp_get_thread_num();
        int first_omega = 1;
        index_t E_idx = -1;
        for (index_t w=0; w<w_count[t]; ++w) {
            if (first_omega) {
                // find first energy that corresponds to omega-F*eta
                E_idx = find_bin_irreg( E_sorted, nE, omega[w_displ[t]+w]-eta_bin );
                E_idx = regularize( E_idx, nE );
                first_omega = 0;
            } else {
                // find next energy that corresponds to omega-F*eta
                while (E_idx < nE && E_sorted[E_idx] <= (omega[w_displ[t]+w]-eta_bin))
                    E_idx++;
                E_idx = regularize( E_idx, nE );
            }
            for (index_t i=0; (E_idx+i) < nE &&
                              E_sorted[E_idx+i] <= (omega[w_displ[t]+w]+eta_bin); ++i) {
                for (index_t o=0; o<nb; ++o)
                    ldos[IDX2(w_displ[t]+w, o, nb)] += cabs2(U_sorted[IDX2(E_idx+i, o, nb)]) *
                        cimag(1./(omega[w_displ[t]+w] - I*eta - E_sorted[E_idx+i]));
            }
        }
    }

    free( w_count );
    shared_free( E_sorted );
    shared_free( U_sorted );
}

double* diverge_model_dos( const diverge_model_t* m, const double* E, index_t nb,
        double* omega_, index_t n_omega_, double eta ) {
    if (!m->internals->has_common_internals) {
        mpi_err_printf( "need common internals for DOS\n" );
        return NULL;
    }
    if (!E) {
        nb = (nb <= 0) ? m->n_orb * m->n_spin : nb;
        E = diverge_model_internals_get_E(m);
    }

    int reallocate_omega = n_omega_ < 0;
    n_omega_ = labs(n_omega_);

    int optimize = eta < 0 && n_omega_ > 1;
    eta = fabs(eta);

    // realloc logic
    double* dos_ = NULL;
    double* rdos_ = NULL;
    if (reallocate_omega) {
        rdos_ = omega_ = realloc(omega_, sizeof*dos_ * n_omega_ * 2);
        dos_ = rdos_ + n_omega_;
        memset( dos_, 0, sizeof*dos_ * n_omega_ );
    } else {
        dos_ = rdos_ = calloc(n_omega_, sizeof*dos_);
    }
    _dos_sort_doubles_( omega_, n_omega_ );

    // MPI distribution
    index_t* omega_counts = diverge_mpi_distribute(n_omega_);
    index_t* omega_displs = omega_counts + diverge_mpi_comm_size();

    const double* omega = omega_ + omega_displs[diverge_mpi_comm_rank()];
    index_t n_omega = omega_counts[diverge_mpi_comm_rank()];
    double* dos = dos_ + omega_displs[diverge_mpi_comm_rank()];

    index_t nk_tot = kdimtot(m->nk,m->nkf);
    if (use_cuda_dos_algo) {
        ldos_kernel_gpu( E, nk_tot*nb, NULL, 1, omega, n_omega, eta, eta_bin_factor*eta, dos );
    } else {
        if (optimize) {
            ldos_kernel_cpu( E, nk_tot*nb, NULL, 1, omega, n_omega, eta, eta_bin_factor*eta, dos );
        } else {
            #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
            for (index_t w=0; w<n_omega; ++w)
            for (index_t e=0; e<nk_tot*nb; ++e) {
                #pragma omp atomic
                dos[w] += cimag(1./(omega[w] - I*eta - E[e]));
            }
        }
    }

    int* counts = calloc(diverge_mpi_comm_size()*2, sizeof(*counts));
    for (int i=0; i<diverge_mpi_comm_size()*2; ++i) counts[i] = omega_counts[i];
    diverge_mpi_allgatherv_double( dos_, counts, counts+diverge_mpi_comm_size() );
    free( counts );
    free( omega_counts );
    return rdos_;
}

double* diverge_model_ldos( const diverge_model_t* m, const double* E,
        const complex128_t* U, index_t nb, double* omega_, index_t n_omega_,
        double eta ) {
    if (!m->internals->has_common_internals) {
        mpi_err_printf( "need common internals for DOS\n" );
        return NULL;
    }
    if (!E || !U) {
        nb = (nb <= 0) ? m->n_orb * m->n_spin : nb;
        if (!E) E = diverge_model_internals_get_E(m);
        if (!U) U = diverge_model_internals_get_U(m);
    }

    int reallocate_omega = n_omega_ < 0;
    n_omega_ = labs(n_omega_);

    int optimize = eta < 0 && n_omega_ > 1;
    eta = fabs(eta);

    // realloc logic
    double* ldos_ = NULL;
    double* rldos_ = NULL;
    if (reallocate_omega) {
        rldos_ = omega_ = realloc(omega_, sizeof*ldos_ * n_omega_ * (nb+1));
        ldos_ = rldos_ + n_omega_;
        memset( ldos_, 0, sizeof*ldos_ * n_omega_ * nb );
    } else {
        ldos_ = rldos_ = calloc(n_omega_*nb, sizeof*ldos_);
    }
    _dos_sort_doubles_( omega_, n_omega_ );

    // MPI distribution
    index_t* omega_counts = diverge_mpi_distribute(n_omega_);
    index_t* omega_displs = omega_counts + diverge_mpi_comm_size();

    const double* omega = omega_ + omega_displs[diverge_mpi_comm_rank()];
    index_t n_omega = omega_counts[diverge_mpi_comm_rank()];
    double* ldos = ldos_ + nb*omega_displs[diverge_mpi_comm_rank()];

    index_t nk_tot = kdimtot(m->nk,m->nkf);
    if (use_cuda_dos_algo) {
        ldos_kernel_gpu( E, nk_tot*nb, U, nb, omega, n_omega, eta, eta_bin_factor*eta, ldos );
    } else {
        if (optimize) {
            ldos_kernel_cpu( E, nk_tot*nb, U, nb, omega, n_omega, eta, eta_bin_factor*eta, ldos );
        } else {
            #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
            for (index_t w=0; w<n_omega; ++w)
            for (index_t e=0; e<nk_tot*nb; ++e)
            for (index_t b=0; b<nb; ++b) {
                #pragma omp atomic
                ldos[IDX2(w,b,nb)] += cabs2(U[IDX2(e,b,nb)]) * cimag(1./(omega[w] - I*eta - E[e]));
            }
        }
    }

    // MPI communication
    int* counts = calloc(diverge_mpi_comm_size()*2, sizeof(*counts));
    for (int i=0; i<diverge_mpi_comm_size()*2; ++i) counts[i] = nb*omega_counts[i];
    diverge_mpi_allgatherv_double( ldos_, counts, counts+diverge_mpi_comm_size() );
    free( counts );
    free( omega_counts );
    return rldos_;
}

void diverge_dos_set_eta_factor( double eta_factor ) {
    eta_bin_factor = eta_factor;
}

void diverge_dos_use_gpu( int use_gpu ) {
    use_cuda_dos_algo = use_gpu;
}

double* diverge_linspace( double min, double max, index_t num ) {
    int endpoint = num > 0;
    num = labs(num);

    double* res = calloc(num, sizeof*res);
    if (num == 1) {
        res[0] = min;
        return res;
    }
    if (num == 0) {
        return NULL;
    }

    double len = endpoint ? num - 1 : num;
    for (index_t i=0; i<num; ++i) {
        res[i] = min + (double)i/len * (max - min);
    }
    return res;
}

double* diverge_filling_to_energy( const diverge_model_t* m, const double* E,
        index_t nb, const double* nu, index_t n_nu ) {
    if (!m->internals->has_common_internals) {
        mpi_err_printf( "need common internals for nu to energy\n" );
        return NULL;
    }
    if (!E) {
        nb = (nb <= 0) ? m->n_orb * m->n_spin : nb;
        if (!E) E = diverge_model_internals_get_E(m);
    }

    index_t n_E = kdimtot(m->nk, m->nkf) * nb;
    double* E_sort = shared_calloc( n_E, sizeof*E_sort );
    if (shared_exclusive_enter(E_sort) == 1) {
        memcpy( E_sort, E, n_E*sizeof*E_sort );
        _dos_sort_doubles_( E_sort, n_E );
    }
    shared_exclusive_wait( E_sort );

    double* omega = calloc( n_nu, sizeof*omega );
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t n=0; n<n_nu; ++n)
        omega[n] = E_sort[ regularize( lroundl( nu[n] * (double)n_E ), n_E ) ];

    shared_free( E_sort );

    return omega;
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

static inline double cabs2( complex128_t U ) {
    const double* pU = (const double*)&U;
    return POW2(pU[0]) + POW2(pU[1]);
}

static inline index_t regularize( index_t idx, index_t sz ) {
    return idx >= sz ? (sz - 1) : (idx <= 0) ? 0 : idx;
}

#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) ( ((x)-(y)) < 0 ? -1 : \
                                 ((x)-(y)) > 0 ? 1 : 0 )
MERGESORT_IMPL( doublesort, double )

#undef MERGESORT_COMP_R
#define MERGESORT_COMP_R( x, y, A ) ( \
        (((const double*)(A))[(x)] - ((const double*)(A))[(y)]) < 0 ? -1 : \
        (((const double*)(A))[(x)] - ((const double*)(A))[(y)]) > 0 ? 1 : 0 \
)
MERGESORT_IMPL_R( doublesort, index_t )

void _dos_sort_doubles_( double* start, index_t size ) {
    doublesort( start, size, diverge_omp_num_threads(), 0 );
}
void _dos_sort_doubles_indirect_( index_t* start, index_t size, const double* vals ) {
    doublesort_r( start, size, (void*)vals, diverge_omp_num_threads(), 0 );
}

