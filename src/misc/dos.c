#include "dos.h"
#include "../diverge_model.h"
#include "../diverge_model_internals.h"
#include "../diverge_internals_struct.h"
#include "mpi_functions.h"

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

static inline index_t find_bin( double test, double min, double max, index_t num ) {
    if (test <= min) return 0;
    if (test >= max) return num-1;
    index_t result = lroundl( (test - min) / (max - min) * num );
    if (result < 0) return 0;
    if (result >= num) return num-1;
    return result;
}

static double eta_bin_factor = 10.0;
void diverge_optimized_dos_set_binfact( double binfact ) {
    eta_bin_factor = binfact;
}

double* diverge_model_dos( const diverge_model_t* m, const double* E, index_t nb,
        const double* omega, index_t n_omega, double eta ) {
    if (!m->internals->has_common_internals) {
        mpi_err_printf( "need common internals for DOS\n" );
        return NULL;
    }
    if (!E) {
        nb = (nb <= 0) ? m->n_orb * m->n_spin : nb;
        E = diverge_model_internals_get_E(m);
    }
    int optimize = eta < 0 && n_omega > 1;
    eta = fabs(eta);
    double* dos = calloc(n_omega, sizeof*dos);

    index_t nk_tot = kdimtot(m->nk,m->nkf);
    if (optimize) {
        double d_omega = omega[1] - omega[0];
        index_t nbins = lroundl(eta / d_omega * eta_bin_factor);
        if (nbins < 1) nbins = 1;
        if (nbins > n_omega) nbins = n_omega;

        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t e=0; e<nk_tot*nb; ++e) {
            double energy = E[e];
            index_t ifreq = find_bin(energy, omega[0], omega[n_omega-1], n_omega);
            for (index_t w=ifreq-nbins; w<=ifreq+nbins; ++w) {
                if (w < 0) w = 0;
                if (w >= n_omega) break;
                #pragma omp atomic
                dos[w] += cimag(1./(omega[w] - I*eta - energy));
            }
        }
    } else {
        #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
        for (index_t e=0; e<nk_tot*nb; ++e)
        for (index_t w=0; w<n_omega; ++w) {
            #pragma omp atomic
            dos[w] += cimag(1./(omega[w] - I*eta - E[e]));
        }
    }

    return dos;
}

static inline double cabs2( complex128_t U ) {
    const double* pU = (const double*)&U;
    return POW2(pU[0]) + POW2(pU[1]);
}

double* diverge_model_ldos( const diverge_model_t* m, const double* E,
        const complex128_t* U, index_t nb, const double* omega, index_t n_omega,
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
    int optimize = eta < 0 && n_omega > 1;
    eta = fabs(eta);
    double* ldos = calloc(n_omega*nb, sizeof*ldos);

    index_t nk_tot = kdimtot(m->nk,m->nkf);
    if (optimize) {
        double d_omega = omega[1] - omega[0];
        index_t nbins = lroundl(eta / d_omega * eta_bin_factor);
        if (nbins < 1) nbins = 1;
        if (nbins > n_omega) nbins = n_omega;

        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t e=0; e<nk_tot*nb; ++e) {
            double energy = E[e];
            index_t ifreq = find_bin(energy, omega[0], omega[n_omega-1], n_omega);
            for (index_t w=ifreq-nbins; w<=ifreq+nbins; ++w) {
                if (w < 0) w = 0;
                if (w >= n_omega) break;
                for (index_t b=0; b<nb; ++b) {
                    #pragma omp atomic
                    ldos[IDX2(w,b,nb)] += cabs2(U[IDX2(e,b,nb)]) * cimag(1./(omega[w] - I*eta - energy));
                }
            }
        }
    } else {
        #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
        for (index_t e=0; e<nk_tot*nb; ++e)
        for (index_t w=0; w<n_omega; ++w)
        for (index_t b=0; b<nb; ++b) {
            #pragma omp atomic
            ldos[w] += cabs2(U[IDX2(e,b,nb)]) * cimag(1./(omega[w] - I*eta - E[e]));
        }
    }

    return ldos;
}
