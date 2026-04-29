#define LINALG_NO_LAPACKE
#include "../misc/linalg.h"
LINALG_IMPLEMENT
#include "phonons.h"
#include "../diverge_internals_struct.h"
#include "../misc/mergesort.h"
#include "../misc/gemm.h"
#include "../misc/bitary.h"
#include "../misc/qnan.h"
#include "../misc/dv_vector.h"
#include <assert.h>
#include <fftw3.h>

// references:
//
// Hubbard-SSH model: https://doi.org/10.1103/PhysRevB.106.245136
// CBO in Kagome/Phonon: https://doi.org/10.1103/PhysRevB.109.075130
// Hubbard-Holstein: https://doi.org/10.1103/PhysRevB.92.195102

// make the FFT convention human-readable
#define DIVERGE_PH_FFT_SIGN -1

struct tu_data_phonons_t {
    diverge_model_t* pmod;

    index_t nk[3];
    index_t nkf[3];
    index_t nO;

    ph_vertex_t* vtx;
    index_t n_vtx;

    complex128_t* g_ob_qO; // (my_nk, n_spin,n_spin,n_orbff, nO)

    int ignore_imag_freq;
    complex128_t *gbuf, *gbuf_zero; // (my_nk, nO)
    double gbuf_last_Lambda, gbuf_zero_last_Lambda;

    complex128_t *Dch, *Dch_zero; // (my_nk, POW2(POW2(n_spin)*n_orbff))
    double Dch_last_Lambda, Dch_zero_last_Lambda;

    index_t n_spin, n_orbff;
    index_t my_nk;
    index_t my_nk_off;
    const index_t* idx_ibz_in_fullmesh;

    int enable_refinement;
    refinement_t* refine; // dv_vec_free(), not normal free!
    index_t n_refine;

    complex128_t* g_ob_qO_refined; // (n_refine, n_spin,n_spin,n_orbff, nO)
    complex128_t *gbuf_ref, *gbuf_ref_zero; // (n_refined, nO)
    double gbuf_ref_last_Lambda, gbuf_ref_zero_last_Lambda;

    diverge_phonon_vertex_generator_t V;
    diverge_flow_step_t* V_step;
    void* V_ctx;
};

static refinement_t* tu_data_phonons_generate_refinement( tu_data_phonons_t* t );

static void tu_data_phonons_init_vertex( tu_data_phonons_t* t, tu_data_t* d,
        const tu_formfactor_t* tu_ff, index_t n_tu_ff, index_t n_spin, const bool SU2 );
static complex128_t* tu_data_phonons_gfill( tu_data_phonons_t* t,
        const double Lambda, const bool ignore_imag_freq, bool refined );

tu_data_phonons_t* tu_data_phonons_init( diverge_model_t* ph, diverge_model_t* el,
        bool ignore_imag_freq, bool enable_refinement ) {
    tu_data_phonons_t* t = calloc(1, sizeof*t);
    t->V = NULL;
    t->pmod = ph;
    memcpy( t->nk, t->pmod->nk, sizeof t->nk );
    memcpy( t->nkf, t->pmod->nkf, sizeof t->nkf );
    t->nO = t->pmod->n_orb;
    if (!t->pmod->internals->has_common_internals)
        mpi_err_printf( "phonon model must have common internals!\n" );
    if (memcmp( t->nk, el->nk, sizeof(t->nk) ) != 0)
        mpi_err_printf( "kpoints from ph (%li)/el (%li) don't match\n", kdim(t->nk), kdim(el->nk) );
    assert( t->pmod->n_spin == 1 );
    assert( t->pmod->internals->model_phonons_imag_freq != NULL );
    t->vtx = (ph_vertex_t*)t->pmod->vert;
    t->n_vtx = t->pmod->n_vert;
    t->ignore_imag_freq = ignore_imag_freq;
    t->gbuf_last_Lambda = t->gbuf_zero_last_Lambda = t->gbuf_ref_last_Lambda =
        t->gbuf_ref_zero_last_Lambda = t->Dch_last_Lambda = t->Dch_zero_last_Lambda = -1.0;

    t->enable_refinement = enable_refinement;
    if (enable_refinement) {
        t->refine = tu_data_phonons_generate_refinement(t);
        t->n_refine = dv_vec_sz(t->refine);
        mpi_log_printf( "automatic Gamma refinement found %li points per coarse point\n", t->n_refine );
    }

    tu_data_phonons_init_vertex( t, el->internals->tu_data, el->tu_ff, el->n_tu_ff, el->n_spin, el->SU2 );
    return t;
}

void tu_data_phonons_free( tu_data_phonons_t* t ) {
    dv_vec_free(t->refine);
    free(t->gbuf);
    free(t->gbuf_zero);
    free(t->gbuf_ref);
    free(t->gbuf_ref_zero);
    free(t->Dch);
    free(t->Dch_zero);
    free(t->g_ob_qO);
    free(t->g_ob_qO_refined);
    free(t);
}

static double PHONON_FREQ_REGULATOR = 1.e-7;

double tu_data_phonons_change_regulator( double reg ) {
    if (qnan_isnan(reg)) {
        return PHONON_FREQ_REGULATOR;
    } else {
        double old = PHONON_FREQ_REGULATOR;
        PHONON_FREQ_REGULATOR = reg;
        return old;
    }
}

void tu_data_phonons_set_vertex_generator( tu_data_phonons_t* t,
        diverge_phonon_vertex_generator_t V, diverge_flow_step_t* step, void* ctx ) {
    t->V = V;
    t->V_step = step;
    t->V_ctx = ctx;
}

static inline complex128_t gf_gp_plus_gm( const double E, const bool E_imag,
        const bool ignore_imag_freq, const double Lambda ) {
    double E2 = E_imag ? -POW2(E) : POW2(E);
    return (ignore_imag_freq&&E_imag) ? 0.0 : creal(-1./(POW2(Lambda) - I*PHONON_FREQ_REGULATOR + E2));
}
static complex128_t* tu_data_phonons_gfill( tu_data_phonons_t* t,
        const double Lambda, const bool ignore_imag_freq, bool refined ) {
    if (!t->enable_refinement) refined = false;
    const int iszero = Lambda == 0.0;
    complex128_t* buf = refined ? (iszero ? t->gbuf_ref_zero : t->gbuf_ref) :
                                  (iszero ? t->gbuf_zero : t->gbuf);
    double* last = refined ? (iszero ? &t->gbuf_ref_zero_last_Lambda : &t->gbuf_ref_last_Lambda) :
                             (iszero ? &t->gbuf_zero_last_Lambda : &t->gbuf_last_Lambda);
    if (*last == Lambda) return buf;
    *last = Lambda;

    const double* E = t->pmod->internals->E;
    const cbyte_t* E_imag = t->pmod->internals->model_phonons_imag_freq;

    if (refined) {
        #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
        for (index_t r=0; r<t->n_refine; ++r)
        for (index_t b=0; b<t->nO; ++b) {
            const index_t q = t->refine[r].kf_idx;
            buf[IDX2(r,b,t->nO)] = gf_gp_plus_gm( E[IDX2(q,b,t->nO)],
                    bitary_get(E_imag,IDX2(q,b,t->nO)), ignore_imag_freq, Lambda );
        }
    } else {
        #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
        for (index_t kk=0; kk<t->my_nk; ++kk)
        for (index_t b=0; b<t->nO; ++b) {
            const index_t q = kidxc2f( t->idx_ibz_in_fullmesh[t->my_nk_off+kk], t->nk, t->nkf );
            buf[IDX2(kk,b,t->nO)] = gf_gp_plus_gm( E[IDX2(q,b,t->nO)],
                    bitary_get(E_imag,IDX2(q,b,t->nO)), ignore_imag_freq, Lambda );
        }
    }
    return buf;
}

#undef MERGESORT_COMP
#define MERGESORT_COMP( i, j ) memcmp( &i, &j, sizeof(ph_vertex_t) )
MERGESORT_IMPL( ph_vertex_sort, ph_vertex_t )

static inline int formfac_equal( const ph_vertex_t* a, const ph_vertex_t* b ) {
    return !memcmp(a, b, offsetof(ph_vertex_t, RA));
}

static inline index_t find_tu_ff( const tu_formfactor_t* tu_ff, index_t n_tu_ff,
        ph_vertex_t ff, const index_t* nk ) {
    index_t idx = -1;
    for (int d=0; d<3; ++d) while (ff.R12[d] < 0) ff.R12[d] += nk[d]; // make positive
    for (index_t i=0; i<n_tu_ff; ++i) {
        if (tu_ff[i].ofrom == ff.o2 && tu_ff[i].oto == ff.o1 && tu_ff[i].R[0] == ff.R12[0] &&
                tu_ff[i].R[1] == ff.R12[1] && tu_ff[i].R[2] == ff.R12[2]) {
            idx = i;
            break;
        }
    }
    return idx;
}

static void tu_data_phonons_init_vertex( tu_data_phonons_t* t, tu_data_t* d,
        const tu_formfactor_t* tu_ff, index_t n_tu_ff, index_t n_spin, const bool SU2 ) {
    // unique formfactors
    ph_vertex_sort( t->vtx, t->n_vtx, diverge_omp_num_threads(), 0 );
    ph_vertex_t* formfacs = NULL;
    index_t* formfacs_displ = NULL;
    ph_vertex_t formfac = { .o1=-1, .s1=-1, .o2=-1, .s2=-1, .OA=-1 };
    for (index_t i=0; i<t->n_vtx; ++i)
    if (!formfac_equal(&formfac, t->vtx+i)) {
        dv_vec_push( formfacs, (formfac=t->vtx[i]) );
        dv_vec_push( formfacs_displ, i );
    }
    dv_vec_shrink( formfacs );
    dv_vec_shrink( formfacs_displ );
    index_t* formfacs_count = calloc(dv_vec_sz(formfacs), sizeof*formfacs_count);
    if (dv_vec_sz(formfacs) > 0) {
        for (index_t i=0; i<dv_vec_sz(formfacs)-1; ++i)
            formfacs_count[i] = formfacs_displ[i+1] - formfacs_displ[i];
        formfacs_count[dv_vec_sz(formfacs)-1] = t->n_vtx - formfacs_displ[dv_vec_sz(formfacs)-1];
    }

    // find how many modes need FFT treatment. the rest requires hand-made FT.
    index_t n_FFT = 0;
    for (; n_FFT<dv_vec_sz(formfacs); n_FFT++) if (formfacs[n_FFT].mode) break;

    mpi_log_printf( "PH: found %li unique FFT formfacs, and %li non-FFT ones\n", n_FFT, dv_vec_sz(formfacs)-n_FFT );

    // perform the FFT
    const index_t nk = kdim(t->nk);
    fftw_iodim64 dims[3] = {{.n = t->nk[0], .is = 1*t->nk[2]*t->nk[1], .os = 1*t->nk[2]*t->nk[1],},
                            {.n = t->nk[1], .is = 1*t->nk[2],          .os = 1*t->nk[2],         },
                            {.n = t->nk[2], .is = 1,                   .os = 1,                  },};
    fftw_iodim64 howmany = { .n=n_FFT, .is=nk, .os=nk };

    // allocate space for not just the FFT but also the FT. Then perform the
    // FFT on that part of the array that needs an FFT.
    fftw_complex* fftw_buf = fftw_malloc(sizeof*fftw_buf * nk*dv_vec_sz(formfacs));
    complex128_t* fftw_cbuf = (complex128_t*)fftw_buf;
    fftw_plan p = fftw_plan_guru64_dft(3, dims, 1, &howmany, fftw_buf, fftw_buf, DIVERGE_PH_FFT_SIGN, FFTW_ESTIMATE);
    ZERO_OUT(fftw_buf, nk*dv_vec_sz(formfacs));

    double tick, t_fft, t_ft, t_ref;
    tick = diverge_mpi_wtime();
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t ff=0; ff<n_FFT; ++ff) {
        for (index_t cc=0; cc<formfacs_count[ff]; ++cc) {
            index_t f = formfacs_displ[ff] + cc;
            const int32_t* R = t->vtx[f].RA;
            index_t z_id = (R[2]<0)? t->nk[2] + R[2] : R[2],
                    y_id = (R[1]<0)? t->nk[1] + R[1] : R[1],
                    x_id = (R[0]<0)? t->nk[0] + R[0] : R[0];
            fftw_cbuf[IDX4(ff,x_id,y_id,z_id,t->nk[0],t->nk[1],t->nk[2])] += t->vtx[f].g;
        }
    }
    fftw_execute(p);
    t_fft = diverge_mpi_wtime() - tick;
    fftw_destroy_plan(p);

    // now turn to the FT part, i.e., the nontrivial basis position phonons
    // (which cannot be part of the regular ones somehow)
    const claM3d_t latt = claM3d_map(t->pmod->lattice[0]);
    tick = diverge_mpi_wtime();
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t kk=0; kk<nk; ++kk) {
        claV3d_t qvec = claV3d_map(t->pmod->internals->kmesh + 3*kk);
        for (index_t ff=n_FFT; ff<dv_vec_sz(formfacs); ++ff)
        for (index_t cc=0; cc<formfacs_count[ff]; ++cc) {
            index_t f = formfacs_displ[ff] + cc;
            claV3d_t RA_pos = {{t->vtx[f].RA_pos[0], t->vtx[f].RA_pos[1], t->vtx[f].RA_pos[2]}};
            double RA_pos_dot_qvec = claV3d_dot( claM3d_vecmat( RA_pos, latt ), qvec );
            fftw_cbuf[IDX2(ff,kk, nk)] += t->vtx[f].g * cexp(I*(double)DIVERGE_PH_FFT_SIGN*RA_pos_dot_qvec);
        }
    }
    t_ft = diverge_mpi_wtime() - tick;

    t->my_nk = d->my_nk;
    t->my_nk_off = d->my_nk_off;
    t->idx_ibz_in_fullmesh = d->symm->idx_ibz_in_fullmesh;
    t->n_spin = n_spin;
    t->n_orbff = d->n_orbff;

    index_t* ph_vertex_to_mi_map = calloc(dv_vec_sz(formfacs), sizeof*ph_vertex_to_mi_map);
    int n_not_found = 0;
    #pragma omp parallel for num_threads(diverge_omp_num_threads()) reduction(+:n_not_found)
    for (index_t f=0; f<dv_vec_sz(formfacs); ++f) {
        // find tu_ff index
        index_t mapped_tu = find_tu_ff( tu_ff, n_tu_ff, formfacs[f], t->nk );
        index_t mapped_mi = -1;
        if (mapped_tu == -1) {
            n_not_found++;
        } else {
            for (index_t i=0; i<t->n_orbff; ++i) if (d->mi_to_tuffidx[i] == mapped_tu) {
                mapped_mi = i; break;
            }
            if (mapped_mi == -1) n_not_found++;
        }
        ph_vertex_to_mi_map[f] = mapped_mi;
    }
    diverge_mpi_allreduce_int_max_inplace( &n_not_found, 1 );
    if (n_not_found > 0)
        mpi_err_printf( "%li elph entries not found in tu_ff\n", n_not_found );

    const index_t linsz = POW2(t->n_spin)*t->n_orbff;
    const complex128_t* U = t->pmod->internals->U;

    tick = diverge_mpi_wtime();
    if (t->enable_refinement) {
        // do the refinement at the Gamma point
        t->g_ob_qO_refined = calloc(t->n_refine * linsz * t->nO, sizeof*t->g_ob_qO_refined);
        if (!t->g_ob_qO_refined)
            mpi_err_printf( "cant allocate t->g_ob_qO_refined(%li,%li,%li) array\n", t->n_refine, linsz, t->nO )
        #pragma omp parallel num_threads(diverge_omp_num_threads())
        { complex128_t* ffbuf = calloc(linsz * t->nO, sizeof*ffbuf);
          complex128_t* Ubuf = calloc(POW2(t->nO), sizeof*Ubuf);
        #pragma omp for
        for (index_t r=0; r<t->n_refine; ++r) {
            index_t qf = t->refine[r].kf_idx;
            transpose_outofplace( Ubuf, U+POW2(t->nO)*qf, t->nO );
            ZERO_OUT( ffbuf, linsz*t->nO );
            claV3d_t qvec = claV3d_map(t->pmod->internals->kfmesh + 3*qf);
            for (index_t ff=0; ff<dv_vec_sz(formfacs); ++ff)
            for (index_t cc=0; cc<formfacs_count[ff]; ++cc) {
                const index_t f = formfacs_displ[ff] + cc;
                const index_t mi = ph_vertex_to_mi_map[ff];
                if (mi != -1) {
                    claV3d_t RA_pos = t->vtx[f].mode ?
                        (claV3d_t){{t->vtx[f].RA_pos[0], t->vtx[f].RA_pos[1], t->vtx[f].RA_pos[2]}} :
                        (claV3d_t){{t->vtx[f].RA[0], t->vtx[f].RA[1], t->vtx[f].RA[2]}};
                    double RA_pos_dot_qvec = claV3d_dot( claM3d_vecmat( RA_pos, latt ), qvec );

                    const index_t s1s2_ff[] = {formfacs[f].s1, formfacs[f].s2};
                    const index_t s1s2_m1[] = {0,0, 1,1};
                    const index_t* si = (s1s2_ff[0] == -1) ? s1s2_m1 : s1s2_ff;
                    const index_t nsi = (s1s2_ff[0] == -1 && !SU2) ? 2 : 1;
                    for (index_t s=0; s<nsi; ++s) {
                        const index_t s1 = si[IDX2(s,0,2)],
                                      s2 = si[IDX2(s,1,2)];
                        ffbuf[IDX4(s2,s1,mi,formfacs[f].OA, t->n_spin,t->n_orbff,t->nO)] +=
                            t->vtx[f].g * cexp(I*(double)DIVERGE_PH_FFT_SIGN*RA_pos_dot_qvec);
                    }
                }
            }
            single_gemm_serial( ffbuf, Ubuf, t->g_ob_qO_refined+r*linsz*t->nO, linsz,
                    t->nO, t->nO, 1.0, 0.0 );
        } free( Ubuf );
          free( ffbuf ); }
    }
    t_ref = diverge_mpi_wtime() - tick;
    mpi_tim_printf( "PH: FFT @ %.2fs, FT @ %.2f, REF @ %.2fs\n", t_fft, t_ft, t_ref );

    free( formfacs_count );
    dv_vec_free( formfacs_displ );

    // write out the objects to g_ob_qO, but in phonon band space
    t->g_ob_qO = calloc(t->my_nk * linsz * t->nO, sizeof*t->g_ob_qO);
    if (!t->g_ob_qO)
        mpi_err_printf( "cant allocate t->g_ob_qO(%li,%li,%li) array\n", t->my_nk, linsz, t->nO );
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    { complex128_t* ffbuf = calloc(linsz * t->nO, sizeof*ffbuf);
      complex128_t* Ubuf = calloc(POW2(t->nO), sizeof*Ubuf);
    #pragma omp for
    for (index_t kk=0; kk<t->my_nk; ++kk) {
        const index_t q = t->idx_ibz_in_fullmesh[kk + t->my_nk_off];
        transpose_outofplace( Ubuf, U+POW2(t->nO)*kidxc2f(q,t->nk,t->nkf), t->nO );
        ZERO_OUT( ffbuf, linsz * t->nO );
        for (index_t f=0; f<dv_vec_sz(formfacs); ++f) {
            const index_t mi = ph_vertex_to_mi_map[f];
            if (mi != -1) {
                const index_t s1s2_ff[] = {formfacs[f].s1, formfacs[f].s2};
                const index_t s1s2_m1[] = {0,0, 1,1};
                const index_t* si = (s1s2_ff[0] == -1) ? s1s2_m1 : s1s2_ff;
                const index_t nsi = (s1s2_ff[0] == -1 && !SU2) ? 2 : 1;
                for (index_t s=0; s<nsi; ++s) {
                    const index_t s1 = si[IDX2(s,0,2)],
                                  s2 = si[IDX2(s,1,2)];
                    ffbuf[IDX4(s2,s1,mi,formfacs[f].OA, t->n_spin,t->n_orbff,t->nO)] += fftw_cbuf[IDX2(f,q,nk)];
                }
            }
        }
        single_gemm_serial( ffbuf, Ubuf, t->g_ob_qO+kk*linsz*t->nO, linsz, t->nO, t->nO, 1.0, 0.0 );
    } free( ffbuf );
      free( Ubuf ); }

    // and do some allocation
    t->Dch = calloc(t->my_nk * POW2(linsz), sizeof*t->Dch);
    t->Dch_zero = calloc(t->my_nk * POW2(linsz), sizeof*t->Dch_zero);
    t->gbuf_zero = calloc(t->my_nk * t->nO, sizeof*t->gbuf_zero);
    t->gbuf = calloc(t->my_nk * t->nO, sizeof*t->gbuf);
    if (t->enable_refinement) {
        t->gbuf_ref_zero = calloc(t->n_refine * t->nO, sizeof*t->gbuf_ref_zero);
        t->gbuf_ref = calloc(t->n_refine * t->nO, sizeof*t->gbuf_ref);
    }

    free( ph_vertex_to_mi_map );
    fftw_free(fftw_buf);
    dv_vec_free( formfacs );
}

void tu_data_phonons_gen_vert( tu_data_phonons_t* t, double Lambda,
        complex128_t* out, double fac ) {
    const index_t linsz = POW2(t->n_spin)*t->n_orbff,
                  nO = t->nO, my_nk = t->my_nk;

    int calc = 1;
    const int iszero = Lambda == 0.0;
    complex128_t* buf = iszero ? t->Dch_zero : t->Dch;
    double* last = iszero ? &t->Dch_zero_last_Lambda : &t->Dch_last_Lambda;
    if ((calc = (*last != Lambda))) {
        *last = Lambda;
        ZERO_OUT(buf, my_nk*POW2(linsz));
    }

    if (calc) {
        index_t kk_gamma = -1;
        if (t->V == NULL) {
            const complex128_t *g_ph_buf = t->g_ob_qO,
                               *prop_buf = tu_data_phonons_gfill( t, Lambda, t->ignore_imag_freq, false );
            #pragma omp parallel for num_threads(diverge_omp_num_threads()) collapse(3)
            for (index_t kk=0; kk<my_nk; ++kk)
            for (index_t l=0; l<linsz; ++l)
            for (index_t r=0; r<linsz; ++r) {
                if (t->enable_refinement && (t->idx_ibz_in_fullmesh[t->my_nk_off+kk] == 0)) {
                    // we found the gamma point and *dont* add the vertex to it!
                    // instead we save its index (only happens for one thread at a
                    // time)
                    #pragma omp atomic write
                    kk_gamma = kk;
                    continue;
                }
                complex128_t tmp = 0.0;
                for (index_t b=0; b<nO; ++b) {
                    complex128_t gl = conj(g_ph_buf[IDX3(kk,l,b, linsz,nO)]);
                    complex128_t gr = g_ph_buf[IDX3(kk,r,b, linsz,nO)];
                    complex128_t prop = prop_buf[IDX2(kk,b, nO)];
                    tmp += gl * prop * gr;
                }
                buf[IDX3(kk,l,r, linsz,linsz)] += tmp;
            }
        } else {
            (*t->V)( t->V_step, Lambda, buf, t->V_ctx );
        }

        // now we take that saved index for the gamma point and perform the
        // refinement summation (in case it's enabled)
        if (t->enable_refinement && (kk_gamma != -1)) {
            const complex128_t *g_ph_ref_buf = t->g_ob_qO_refined,
                               *prop_ref_buf = tu_data_phonons_gfill( t, Lambda, t->ignore_imag_freq, true );
            #pragma omp parallel for num_threads(diverge_omp_num_threads()) collapse(2)
            for (index_t l=0; l<linsz; ++l)
            for (index_t r=0; r<linsz; ++r) {
                complex128_t tmp = 0.0;
                for (index_t f=0; f<t->n_refine; ++f)
                for (index_t b=0; b<nO; ++b) {
                    complex128_t gl = conj(g_ph_ref_buf[IDX3(f,l,b, linsz,nO)]);
                    complex128_t gr = g_ph_ref_buf[IDX3(f,r,b, linsz,nO)];
                    complex128_t prop = prop_ref_buf[IDX2(f,b, nO)];
                    tmp += gl * prop * gr * t->refine[f].weight;
                }
                buf[IDX3(kk_gamma,l,r, linsz,linsz)] += tmp;
            }
        }
    }

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<my_nk*POW2(linsz); ++i) out[i] += fac * buf[i];
}

diverge_model_t* tu_data_phonons_get_model( tu_data_phonons_t* t ) { return t->pmod; }
complex128_t* tu_data_phonons_get_g( tu_data_phonons_t* t ) { return t->g_ob_qO; }

static inline double periodic_norm( claV3d_t k, claM3d_t G, const index_t shift[3] ) {
    double norm = claV3d_norm(k);
    for (index_t sx=-shift[0]; sx<=shift[0]; ++sx)
    for (index_t sy=-shift[1]; sy<=shift[1]; ++sy)
    for (index_t sz=-shift[2]; sz<=shift[2]; ++sz) {
        double snorm = claV3d_norm(claV3d_add(k,
                    claM3d_vecmat( (claV3d_t){{sx,sy,sz}}, G )));
        norm = MIN(norm, snorm);
    }
    return norm;
}

static double DIVERGE_PH_REFINEMENT_GEN_EPS_MESH = 1.e-4;
double tu_data_phonons_change_refinement_eps( double eps ) {
    if (qnan_isnan(eps)) {
        return DIVERGE_PH_REFINEMENT_GEN_EPS_MESH;
    } else {
        double old = DIVERGE_PH_REFINEMENT_GEN_EPS_MESH;
        DIVERGE_PH_REFINEMENT_GEN_EPS_MESH = eps;
        return old;
    }
}

typedef struct { double norm_to_star_idx; index_t star_idx; } star_norm_t;
#undef SERISORT_COMP
#define SERISORT_COMP(x, y) ((x.norm_to_star_idx)-(y.norm_to_star_idx))
SERISORT_IMPL( serisort_star_norm, star_norm_t )

#undef MERGESORT_COMP
#define MERGESORT_COMP(x, y) ((x).kf_idx-(y).kf_idx)
MERGESORT_IMPL( mergesort_refinement, refinement_t )

static refinement_t* tu_data_phonons_generate_refinement( tu_data_phonons_t* t ) {
    diverge_model_t* m = t->pmod;
    const index_t nkkf = kdimtot(m->nk, m->nkf);
    const claM3d_t G = claM3d_map(m->internals->rlattice.v[0]);

    double Gnorm = claV3d_norm(claV3d_map(G.m[0]));
    for (int d=1; d<m->internals->dim; ++d) {
        double Gnorm_d = claV3d_norm(claV3d_map(G.m[d]));
        Gnorm = MIN(Gnorm,Gnorm_d);
    }

    // set up the star around Gamma
    index_t dk[3] = {2, 2, 2};
    index_t n_star = 1;
    for (int d=0; d<3; ++d) {
        if (labs(dk[d]) > m->nk[d]/2) {
            if (dk[d] > 0)
                dk[d] = m->nk[d]/2;
            else
                dk[d] = -m->nk[d]/2;
        }
        if (d >= m->internals->dim)
            dk[d] = 0;
        n_star *= 2*dk[d]+1;
    }
    index_t star_zero = -1;
    claV3d_t* star = NULL;
    dv_vec_init_cap(star, n_star);
    for (index_t dkx=-dk[0]; dkx<=dk[0]; ++dkx)
    for (index_t dky=-dk[1]; dky<=dk[1]; ++dky)
    for (index_t dkz=-dk[2]; dkz<=dk[2]; ++dkz) {
        claV3d_t dk_vec = {{dkx, dky, dkz}};
        for (int d=0; d<3; ++d) dk_vec.v[d] /= (double)m->nk[d];
        if (dkx==0 && dky==0 && dkz==0) star_zero = dv_vec_sz(star);
        dv_vec_push( star, claM3d_vecmat(dk_vec, G) );
    }

    refinement_t* result = NULL;
    double refinement_total_weight = 0.0;
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
    star_norm_t* star_norms = calloc( n_star, sizeof*star_norms );
    star_norm_t* tmp = calloc( n_star, sizeof*tmp );
    refinement_t* t_refinement = NULL;
    dv_vec_init(t_refinement);

    #pragma omp for
    for (index_t k=0; k<nkkf; ++k) {
        claV3d_t kfvec = claV3d_map(m->internals->kfmesh + 3*k);
        for (index_t s=0; s<n_star; ++s)
            star_norms[s] = (star_norm_t){
                .norm_to_star_idx=periodic_norm(claV3d_sub(kfvec, star[s]), G, dk),
                .star_idx=s,
            };
        serisort_star_norm_s( star_norms, n_star, tmp );
        index_t n_closest = 0;
        int close_to_zero = 0;
        for (index_t s=0; s<n_star; ++s) {
            n_closest = s+1;
            if (fabs(star_norms[s].norm_to_star_idx-star_norms[0].norm_to_star_idx) >
                    (DIVERGE_PH_REFINEMENT_GEN_EPS_MESH * Gnorm))
                break;
            if (star_norms[s].star_idx == star_zero)
                close_to_zero = 1;
        }
        if (!close_to_zero) continue;

        dv_vec_push( t_refinement, (refinement_t){.kf_idx=k, .weight=1./(double)n_closest} );
    }
    dv_vec_shrink( t_refinement );
    free( tmp );
    free( star_norms );

    #pragma omp critical
    {
        for (index_t r=0; r<dv_vec_sz(t_refinement); ++r) {
            dv_vec_push( result, t_refinement[r] );
            refinement_total_weight += t_refinement[r].weight;
        }
    }

    dv_vec_free( t_refinement );
    } // pragma omp parallel

    dv_vec_free( star );

    for (index_t r=0; r<dv_vec_sz(result); ++r)
        result[r].weight /= refinement_total_weight;
    dv_vec_shrink(result);

    // consistent behaviour across MPI ranks
    mergesort_refinement(result, dv_vec_sz(result), diverge_omp_num_threads(), 0);
    return result;
}

refinement_t* tu_data_phonons_get_refine( tu_data_phonons_t* t, index_t* n_refine ) {
    if (n_refine) *n_refine = t->n_refine;
    return t->refine;
}
complex128_t* tu_data_phonons_get_g_ref( tu_data_phonons_t* t ) { return t->g_ob_qO_refined; }
