/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "tc_estimate.h"

#include "../diverge_flow_step_internal.h"
#include "../diverge_internals_struct.h"
#include "../diverge_phonons.h"
#include "../misc/bitary.h"
#include "../misc/mpi_functions.h"
#include "../misc/qnan.h"
#include "diverge_interface.h"
#include "phonons.h"
#include "phonons_details.h"
#include "vertex_details.h"

static double TC_FS_WEIGHT_MIN = 1.e-10;
static double TC_OMEGA_MIN = 1e-8;
static double TC_PROP_Q0 = 0.0;

double tu_tc_estimate_change_fs_weight_min( double w ) {
    if (qnan_isnan(w)) return TC_FS_WEIGHT_MIN;
    return MEMSWAP_RET(w, TC_FS_WEIGHT_MIN);
}

double tu_tc_estimate_change_omega_min( double w ) {
    if (qnan_isnan(w)) return TC_OMEGA_MIN;
    return MEMSWAP_RET(w, TC_OMEGA_MIN);
}

double tu_tc_estimate_change_propagator_q0( double q0 ) {
    if (qnan_isnan(q0)) return TC_PROP_Q0;
    return MEMSWAP_RET(q0, TC_PROP_Q0);
}

typedef struct {
    diverge_model_t* mod;
    internals_t* intls;
    tu_vertex_t* V;
    tu_data_phonons_t* ph;

    index_t nb;
    index_t n_orb, n_spin, n_orbff, nO;
    index_t nkkf[3];
    index_t nktot, nkc;

    double N0;
    index_t n_fs;

    complex128_t* A;    // (n_orbff,) pair bond amplitude on FS

    index_t n_entries;  // my_nk + n_refine
    double* wq;         // (n_entries, nO) lambda weights, N0 * P(q) * |c|^2
    double* omega;      // (n_entries, nO) mode frequencies, >=0
    double* D;          // (n_entries, nO) phonon Green function at TC_PROP_Q0
    int8_t* mflag;      // (n_entries, nO) TC_MODE_*
    index_t n_modes_skipped, n_modes_unstable, n_refine_used;
} tc_work_t;

enum { TC_MODE_OK = 0, TC_MODE_SKIP = 1, TC_MODE_UNSTABLE = 2 };

static void tc_work_free( tc_work_t* w ) {
    if (!w) return;
    free(w->A); free(w->wq); free(w->omega); free(w->D); free(w->mflag);
    free(w);
}

static inline void unravel3( index_t idx, const index_t* dim, index_t* out ) {
    out[0] = idx / (dim[1]*dim[2]);
    out[1] = (idx % (dim[1]*dim[2])) / dim[2];
    out[2] = idx % dim[2];
}

static inline double cabs2( const complex128_t c ) {
    const double* d = (const double*)&c;
    return POW2(d[0]) + POW2(d[1]);
}

// classify one (q,mode) entry and store it. om: raw freq, imag: unstable?, D: prop
static void tc_set_entry( tc_work_t* w, index_t i, double om, bool imag,
        double weight, double D ) {
    w->wq[i] = weight;
    w->omega[i] = fabs(om);
    w->D[i] = D;
    if (imag || D >= 0.0) {
        w->mflag[i] = TC_MODE_UNSTABLE;
        w->n_modes_unstable++;
    } else if (fabs(om) < TC_OMEGA_MIN) {
        w->mflag[i] = TC_MODE_SKIP;
        w->n_modes_skipped++;
    } else {
        w->mflag[i] = TC_MODE_OK;
    }
}

// build FS averages
static tc_work_t* tc_work_init( diverge_flow_step_t* step, double smearing ) {
    tc_work_t* w = calloc(1, sizeof*w);
    w->mod = step->mod;
    w->intls = step->intls;
    w->V = step->tu_vertex;
    tu_data_t* td = (tu_data_t*)w->intls->tu_data;
    w->ph = td->phonons_reference;
    w->n_orb = w->mod->n_orb;
    w->n_spin = w->mod->n_spin;
    w->nb = w->n_orb * w->n_spin;
    w->n_orbff = w->V->n_orbff;
    w->nO = w->ph->nO;
    for (int d=0; d<3; ++d) w->nkkf[d] = w->mod->nk[d] * w->mod->nkf[d];
    w->nktot = kdimtot(w->mod->nk, w->mod->nkf);
    w->nkc = kdim(w->mod->nk);

    const index_t nb = w->nb, nktot = w->nktot, n_orbff = w->n_orbff;
    const double* E = w->intls->E;
    const complex128_t* U = w->intls->U;

    double* Wk = calloc( nktot, sizeof*Wk );
    w->A = calloc( n_orbff, sizeof*w->A );

    const double gnorm = 1.0 / (smearing * sqrt(M_PI));
    const double wcut = gnorm * TC_FS_WEIGHT_MIN;
    double wsum = 0.0;

    for (index_t kf=0; kf<nktot; ++kf) {
        double sum = 0.0;
        for (index_t b=0; b<nb; ++b) {
            const double x = E[IDX2(kf,b,nb)] / smearing;
            const double g = gnorm * exp(-POW2(x));
            if (g < wcut) continue;
            sum += g;
        }
        Wk[kf] = sum;
        wsum += sum;
    }
    if (wsum <= 0.0) mpi_err_printf( "empty Fermi surface/DOS (%.5e)\n", wsum );

    const double spin_deg = (w->n_spin == 1) ? 1.0 : 2.0;
    w->N0 = wsum / ((double)nktot * spin_deg);

    const index_t sp_up = 0;
    const index_t sp_dn = (w->n_spin == 1) ? 0 : 1;
    const index_t* mi_ofrom = td->mi_to_ofrom;
    const index_t* mi_oto = td->mi_to_oto;
    const index_t* mi_R = td->mi_to_R;

    for (index_t kf=0; kf<nktot; ++kf) {
        if (Wk[kf] <= 0.0) continue;
        index_t ik[3], imk[3];
        unravel3( kf, w->nkkf, ik );
        for (int d=0; d<3; ++d) imk[d] = (w->nkkf[d] - ik[d]) % w->nkkf[d];
        const index_t mkf = IDX3(imk[0],imk[1],imk[2], w->nkkf[1],w->nkkf[2]);

        for (index_t b=0; b<nb; ++b) {
            const double x = E[IDX2(kf,b,nb)] / smearing;
            const double g = gnorm * exp( -x*x );
            if (g < wcut) continue;
            w->n_fs += 1;

            for (index_t f=0; f<n_orbff; ++f) {
                const index_t o1 = mi_ofrom[f] + w->n_orb * sp_up;
                const index_t o2 = mi_oto[f] + w->n_orb * sp_dn;
                const double ph_arg = (double)(ik[0]*mi_R[3*f+0]) / (double)w->nkkf[0] +
                                      (double)(ik[1]*mi_R[3*f+1]) / (double)w->nkkf[1] +
                                      (double)(ik[2]*mi_R[3*f+2]) / (double)w->nkkf[2];
                w->A[f] += g * conj( U[IDX3(kf,b,o1,nb,nb)] )
                             * conj( U[IDX3(mkf,b,o2,nb,nb)] )
                             * cexp( 2.*M_PI*I*ph_arg );
            }
        }
    }
    for (index_t f=0; f<n_orbff; ++f) w->A[f] /= wsum;
    for (index_t kf=0; kf<nktot; ++kf) Wk[kf] /= wsum;

    // FS-FS autocorrelation on coarse mesh
    double* Pq = calloc( w->nkc, sizeof*Pq );
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t q=0; q<w->nkc; ++q) {
        index_t qf = kidxc2f(q, w->mod->nk, w->mod->nkf);
        double sum = 0.0;
        for (index_t k=0; k<nktot; ++k) {
            if (Wk[k] <= 0.0) continue;
            const index_t kpq = k1p2(qf, k, w->nkkf);
            sum += Wk[k] * Wk[kpq];
        }
        Pq[q] = sum;
    }
    free(Wk);

    // to IBZ
    const tu_symmstruct_t* symm = w->V->symm;
    const index_t nkibz = (td->nkibz > 0) ? td->nkibz : w->nkc;
    double* Pibz = calloc( nkibz, sizeof*Pibz );
    if (symm && symm->kmaps_to) {
        for (index_t q=0; q<w->nkc; ++q) Pibz[ symm->kmaps_to[q] ] += Pq[q];
    } else {
        for (index_t q=0; q<w->nkc && q<nkibz; ++q) Pibz[q] = Pq[q];
    }
    free(Pq);

    // weights
    const index_t my_nk = w->ph->my_nk, my_nk_off = w->ph->my_nk_off, nO = w->nO;
    const index_t n_spin = w->n_spin;
    const complex128_t* gq = w->ph->g_ob_qO;
    const double* Eph = w->ph->pmod->internals->E;
    const cbytes_t* Eimag = w->ph->pmod->internals->model_phonons_imag_freq;
    const bool ignore_imag = w->ph->ignore_imag_freq;

    // D
    const complex128_t* Dq = tu_data_phonons_gfill( w->ph, TC_PROP_Q0, ignore_imag, false );

    index_t kk_gamma = -1;
    if (w->ph->enable_refinement && w->ph->n_refine > 0)
        for (index_t kk=0; kk<my_nk; ++kk)
            if (w->ph->idx_ibz_in_fullmesh[kk+my_nk_off] == 0) { kk_gamma = kk; break; }
    const index_t n_ref = (kk_gamma >= 0) ? w->ph->n_refine : 0;
    w->n_refine_used = n_ref;
    w->n_entries = my_nk + n_ref;

    w->wq = calloc( w->n_entries*nO, sizeof*w->wq );
    w->omega = calloc( w->n_entries*nO, sizeof*w->omega );
    w->D = calloc( w->n_entries*nO, sizeof*w->D );
    w->mflag = calloc( w->n_entries*nO, sizeof*w->mflag );

    for (index_t kk=0; kk<my_nk; ++kk) {
        const index_t iq = kk + my_nk_off;
        const index_t qfull = w->ph->idx_ibz_in_fullmesh[iq];
        const index_t qfine = kidxc2f(qfull, w->ph->nk, w->ph->nkf);
        const double Pw = (iq < nkibz) ? Pibz[iq] : 0.0;

        for (index_t b=0; b<nO; ++b) {
            if (kk == kk_gamma) { // carried by the refinement entries below
                w->mflag[IDX2(kk,b,nO)] = TC_MODE_SKIP;
                continue;
            }
            complex128_t c = 0.0;
            for (index_t f=0; f<n_orbff; ++f)
                c += conj(w->A[f]) * gq[IDX5(kk,0,0,f,b, n_spin,n_spin,n_orbff,nO)];
            tc_set_entry( w, IDX2(kk,b,nO), Eph[IDX2(qfine,b,nO)],
                    bitary_get(Eimag, IDX2(qfine,b,nO)),
                    w->N0 * Pw * cabs2(c), creal(Dq[IDX2(kk,b,nO)]) );
        }
    }

    if (n_ref > 0) {
        const complex128_t* gref = w->ph->g_ob_qO_refined;
        const complex128_t* Dref = tu_data_phonons_gfill( w->ph, TC_PROP_Q0, ignore_imag, true );
        const index_t iq = kk_gamma + my_nk_off;
        const double Pw = (iq < nkibz) ? Pibz[iq] : 0.0;

        for (index_t r=0; r<n_ref; ++r) {
            const index_t qfine = w->ph->refine[r].kf_idx;
            const double rw = w->ph->refine[r].weight;
            for (index_t b=0; b<nO; ++b) {
                complex128_t c = 0.0;
                for (index_t f=0; f<n_orbff; ++f)
                    c += conj(w->A[f]) * gref[IDX5(r,0,0,f,b, n_spin,n_spin,n_orbff,nO)];
                tc_set_entry( w, IDX2(my_nk+r,b,nO), Eph[IDX2(qfine,b,nO)],
                        bitary_get(Eimag, IDX2(qfine,b,nO)),
                        w->N0 * Pw * rw * cabs2(c), creal(Dref[IDX2(r,b,nO)]) );
            }
        }
    }
    free(Pibz);

    index_t cnt[2] = { w->n_modes_skipped, w->n_modes_unstable };
    diverge_mpi_allreduce_index_sum_inplace( cnt, 2 );
    w->n_modes_skipped = cnt[0];
    w->n_modes_unstable = cnt[1];
    return w;
}

static double tc_pair_vertex_fs( diverge_flow_step_t* step, const tc_work_t* w ) {
    diverge_flow_step_vertex_t vv = diverge_flow_step_vertex( step, 'p' );

    const index_t n_orbff = w->n_orbff, n_spin = w->n_spin;

    double v = 0.0;
    if (diverge_mpi_comm_rank() == 0) { // we have q=0
        complex128_t sum = 0.0;
        for (index_t f=0; f<n_orbff; ++f)
        for (index_t g=0; g<n_orbff; ++g) {
            complex128_t vfg;
            if (n_spin == 1) {
                vfg = vv.ary[IDX6(0,0,g,0,0,f, n_spin,n_orbff,n_spin,n_spin,n_orbff)];
            } else {
                vfg = vv.ary[IDX6(1,0,g,1,0,f, n_spin,n_orbff,n_spin,n_spin,n_orbff)]
                    - vv.ary[IDX6(0,1,g,1,0,f, n_spin,n_orbff,n_spin,n_spin,n_orbff)];
            }
            sum += conj(w->A[f]) * vfg * w->A[g];
        }
        v = creal(sum);
    }
    diverge_mpi_allreduce_double_sum_inplace( &v, 1 );
    return v;
}

tu_tc_result_t tu_tc_estimate( diverge_flow_step_t* step, double Lambda_star, double smearing ) {
    tu_tc_result_t out = {0};
    tc_work_t* w = tc_work_init( step, smearing );

    out.N0 = w->N0;
    out.n_fs = w->n_fs;
    out.n_modes_skipped = w->n_modes_skipped;
    out.n_modes_unstable = w->n_modes_unstable;
    out.n_refine_used = w->n_refine_used;

    // weight per mode is -wq*D
    // sums: lambda, lambda_avg, sum w ln(omega), sum w omega**2, lambda_unstable
    double sums[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    const index_t nloc = w->n_entries * w->nO;
    for (index_t i=0; i<nloc; ++i) {
        const double ww = -w->wq[i] * w->D[i];
        sums[0] += ww;
        switch (w->mflag[i]) {
            case TC_MODE_OK:
                sums[1] += ww;
                sums[2] += ww * log(w->omega[i]);
                sums[3] += ww * POW2(w->omega[i]);
                break;
            case TC_MODE_UNSTABLE:
                sums[4] += ww;
                break;
            default: break;
        }
    }
    diverge_mpi_allreduce_double_sum_inplace( sums, 5 );

    out.lambda = sums[0];
    out.lambda_avg = sums[1];
    out.lambda_unstable = sums[4];
    if (out.lambda_avg > 0.0) {
        out.omega_log = exp( sums[2] / out.lambda_avg );
        out.omega_rms = sqrt( sums[3] / out.lambda_avg );
    }
    if (out.lambda_unstable < 0.0) {
        mpi_wrn_printf( "unstable phonons contribute %.3e to lambda=%.3e\n",
                out.lambda_unstable, out.lambda );
        out.errors++;
    }

    out.vpair_fs = tc_pair_vertex_fs( step, w );
    out.mu_star_at_Lstar = w->N0 * out.vpair_fs;

    // rest of scales: Morel-Anderson Lambda* -> omega_log.
    out.mu_star = out.mu_star_at_Lstar;
    if (out.omega_log > 0.0 && Lambda_star > 0.0) {
        const double lg = log( Lambda_star / out.omega_log );
        const double den = 1.0 + out.mu_star_at_Lstar * lg;
        if (lg > 0.0 && den > 0.0) {
            out.mu_star = out.mu_star_at_Lstar / den;
        } else if (lg > 0.0) {
            mpi_wrn_printf( "Morel-Anderson denom <=0 (%.3e)\n", den );
            out.errors++;
        } else {
            mpi_wrn_printf( "Lambda* (%.3e) below omega_log (%.3e)\n", Lambda_star, out.omega_log );
            out.errors++;
        }
    }

    // McMillan / Allen-Dynes, no strong-coupling corrections
    const double den = out.lambda - out.mu_star * (1.0 + 0.62*out.lambda);
    if (out.lambda > 0.0 && den > 0.0) {
        out.Tc = out.omega_log / 1.2 * exp( -1.04 * (1.0 + out.lambda) / den );
    } else {
        out.errors++;
        out.Tc = 0.0;
    }

    tc_work_free(w);
    return out;
}

