#define LINALG_NO_LAPACKE
#include "../misc/linalg.h"
LINALG_IMPLEMENT
#include "phonons.h"
#include "../diverge_internals_struct.h"
#include "../diverge_patching_helpers.h" // atomic_add
#include "../misc/mergesort.h"
#include "../misc/gemm.h"
#include "../misc/bitary.h"
#include <assert.h>
#include <fftw3.h>

// references:
//
// Hubbard-SSH model: https://doi.org/10.1103/PhysRevB.106.245136
// CBO in Kagome/Phonon: https://doi.org/10.1103/PhysRevB.109.075130
// Hubbard-Holstein: https://doi.org/10.1103/PhysRevB.92.195102

// TODO this is some infrastructure for testing the factors
static int fft_sign = -1;
static int conj_U = 0;
static int conj_l = 1, conj_r = 0;
static double phase_pref = 0.0;
static void select_config( void ) {
    char* cvalue = getenv("PHONON_CFG");
    if (cvalue) {
        int value = atoi(cvalue);
        if (value != -1) {
            int selectors[] = {value & (1<<0), value & (1<<1), value & (1<<2), value & (1<<3),
                               value & (1<<4), value & (1<<5), value & (1<<6), value & (1<<7)};
            fft_sign = selectors[0] ? 1 : -1;
            conj_U = selectors[1] ? 1 : 0;
            conj_l = selectors[2] ? 1 : 0;
            conj_r = selectors[3] ? 1 : 0;
            int s456 = (selectors[4] | selectors[5] | selectors[6]) >> 4;
            switch (s456) {
                case 0: phase_pref =  0.0; break;
                case 1: phase_pref = -0.5; break;
                case 2: phase_pref = -1.0; break;
                case 3: phase_pref = -2.0; break;
                case 4: phase_pref =  0.5; break;
                case 5: phase_pref =  1.0; break;
                case 6: phase_pref =  2.0; break;
                default: phase_pref = 0.0; break;
            }
            mpi_scc_printf( "config #%2i: fft_sign=%i, conj_U=%i, conj_l=%i, conj_r=%i, phase_pref=%.1f\n",
                    value, fft_sign, conj_U, conj_l, conj_r, phase_pref );
        }
    }
}

static void tu_data_phonons_init_vertex( tu_data_phonons_t* t, tu_data_t* d,
        const tu_formfactor_t* tu_ff, index_t n_tu_ff, index_t n_spin );
static complex128_t* tu_data_phonons_gfill( tu_data_phonons_t* t, double Lambda );

tu_data_phonons_t* tu_data_phonons_init( diverge_model_t* ph, diverge_model_t* el,
        bool buffered ) {
    select_config();
    tu_data_phonons_t* t = calloc(1, sizeof*t);
    t->pmod = ph;
    memcpy( t->nk, t->pmod->nk, sizeof t->nk );
    memcpy( t->nkf, t->pmod->nkf, sizeof t->nkf );
    t->nO = t->pmod->n_orb;
    if (!t->pmod->internals->has_common_internals)
        mpi_err_printf( "phonon model must have common internals!\n" );
    if (memcmp( t->nk, el->nk, sizeof(t->nk) ) != 0)
        mpi_err_printf( "kpoints from ph (%li)/el (%li) don't match\n", kdim(t->nk), kdim(el->nk) );
    assert( t->pmod->n_spin == 1 );
    assert( t->pmod->internals->greens_shared == 0 );
    assert( t->pmod->gproj == NULL );
    t->vtx = (ph_vertex_t*)t->pmod->vert;
    t->n_vtx = t->pmod->n_vert;
    t->buffered = buffered;
    t->gbuf_last_Lambda = t->gbuf_zero_last_Lambda =
        t->Dch_last_Lambda = t->Dch_zero_last_Lambda = -1.0;
    tu_data_phonons_init_vertex( t, el->internals->tu_data, el->tu_ff, el->n_tu_ff, el->n_spin );
    return t;
}

void tu_data_phonons_free( tu_data_phonons_t* t ) {
    free(t->gbuf);
    free(t->gbuf_zero);
    free(t->Dch);
    free(t->Dch_zero);
    free(t->g_ob_qO);
    free(t);
}

static complex128_t* tu_data_phonons_gfill( tu_data_phonons_t* t, double Lambda ) {
    const int iszero = Lambda == 0.0;
    complex128_t* buf = iszero ? t->gbuf_zero : t->gbuf;
    double* last = iszero ? &t->gbuf_zero_last_Lambda : &t->gbuf_last_Lambda;
    if (*last == Lambda) return buf;
    *last = Lambda;

    gf_complex_t* greens = t->pmod->internals->greens;
    const complex128_t cLambda = I*(iszero?DIVERGE_EPS_MESH:Lambda), one = 1.0;
    const index_t nO = t->nO, my_nk = t->my_nk, my_nk_off = t->my_nk_off,
                  nkf = kdimtot(t->pmod->nk,t->pmod->nkf);
    const double* E = t->pmod->internals->E;
    const cbyte_t* E_imag = t->pmod->internals->model_phonons_imag_freq;
    assert( E_imag != NULL );

    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for (int sign=0; sign<2; ++sign)
    for (index_t k=0; k<nkf; ++k)
    for (index_t b=0; b<nO; ++b) {
        greens[IDX3(sign, k, b, nkf, nO)] = 1./ ( (sign?cLambda:conj(cLambda))
                - (bitary_get(E_imag,IDX2(k,b,nO))?I:one) * E[IDX2(k,b,nO)] );
    }

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t kk=0; kk<my_nk; ++kk) {
        const index_t q = kidxc2f( t->idx_ibz_in_fullmesh[my_nk_off + kk], t->nk, t->nkf );
        for (index_t b=0; b<nO; ++b)
            buf[IDX2(kk,b,nO)] = greens[IDX3(0,q,b,nkf,nO)] + greens[IDX3(1,q,b,nkf,nO)];
    }
    return buf;
}

#undef MERGESORT_COMP
#define MERGESORT_COMP( i, j ) memcmp( &i, &j, offsetof(ph_vertex_t, RA) )
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
        const tu_formfactor_t* tu_ff, index_t n_tu_ff, index_t n_spin ) {
    // unique formfactors
    for (index_t i=0; i<t->n_vtx; ++i) t->vtx[i]._chan = 'p';
    ph_vertex_sort( t->vtx, t->n_vtx, diverge_omp_num_threads(), 0 );
    VECTOR_INIT( ph_vertex_t, formfacs );
    VECTOR_INIT( index_t, formfacs_displ );
    ph_vertex_t formfac = { .o1=-1, .s1=-1, .o2=-1, .s2=-1, .OA=-1 };
    for (index_t i=0; i<t->n_vtx; ++i)
    if (!formfac_equal(&formfac, t->vtx+i)) {
        VECTOR_PUSH_BACK( formfacs, (formfac=t->vtx[i]) );
        VECTOR_PUSH_BACK( formfacs_displ, i );
    }
    VECTOR_SHRINK_TO_FIT( formfacs );
    VECTOR_SHRINK_TO_FIT( formfacs_displ );
    index_t* formfacs_count = calloc(formfacs_sz, sizeof*formfacs_count);
    if (formfacs_sz > 0) {
        for (index_t i=0; i<formfacs_sz-1; ++i)
            formfacs_count[i] = formfacs_displ[i+1] - formfacs_displ[i];
        formfacs_count[formfacs_sz-1] = t->n_vtx - formfacs_displ[formfacs_sz-1];
    }

    // perform the FFT
    const index_t nk = kdim(t->nk);
    fftw_iodim64 dims[3];
    dims[0].n = t->nk[0], dims[1].n = t->nk[1], dims[2].n = t->nk[2];
    dims[0].is = t->nk[2]*t->nk[1], dims[1].is = t->nk[2], dims[2].is = 1;
    dims[0].os = t->nk[2]*t->nk[1], dims[1].os = t->nk[2], dims[2].os = 1;
    fftw_iodim64 howmany = { .n=formfacs_sz, .is=nk, .os=nk };

    fftw_complex* fftw_buf = fftw_malloc(sizeof*fftw_buf * nk*formfacs_sz);
    complex128_t* fftw_cbuf = (complex128_t*)fftw_buf;
    // TODO sign of the FFT
    fftw_plan p = fftw_plan_guru64_dft(3, dims, 1, &howmany, fftw_buf, fftw_buf, fft_sign, FFTW_ESTIMATE);
    ZERO_OUT(fftw_buf, nk*formfacs_sz);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t ff=0; ff<formfacs_sz; ++ff) {
        for (index_t cc=0; cc<formfacs_count[ff]; ++cc) {
            index_t f = formfacs_displ[ff] + cc;
            // TODO maybe we have to put another R vector in the FFT? or an
            // additional one?
            const index_t* R = t->vtx[f].RA;
            index_t z_id = (R[2]<0)? t->nk[2] + R[2] : R[2],
                    y_id = (R[1]<0)? t->nk[1] + R[1] : R[1],
                    x_id = (R[0]<0)? t->nk[0] + R[0] : R[0];
            fftw_cbuf[IDX4(ff,x_id,y_id,z_id,t->nk[0],t->nk[1],t->nk[2])] += t->vtx[f].g;
        }
    }
    fftw_execute(p);
    fftw_destroy_plan(p);
    free( formfacs_count );
    VECTOR_FREE( formfacs_displ );

    t->my_nk = d->my_nk;
    t->my_nk_off = d->my_nk_off;
    t->idx_ibz_in_fullmesh = d->symm->idx_ibz_in_fullmesh;
    t->n_spin = n_spin;
    t->n_orbff = d->n_orbff;

    index_t* ph_vertex_to_mi_map = calloc(formfacs_sz, sizeof*ph_vertex_to_mi_map);
    int n_not_found = 0;
    #pragma omp parallel for num_threads(diverge_omp_num_threads()) reduction(+:n_not_found)
    for (index_t f=0; f<formfacs_sz; ++f) {
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
        mpi_err_printf( "couldnt find <=%i tu_ff for elph, try larger max_dist\n", n_not_found );

    // write out the objects to g_ob_qO, but in phonon band space!
    const index_t linsz = POW2(t->n_spin)*t->n_orbff;
    const complex128_t* U = t->pmod->internals->U;
    t->g_ob_qO = calloc(t->my_nk * linsz * t->nO, sizeof*t->g_ob_qO);
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        complex128_t* ffbuf = calloc(linsz * t->nO, sizeof*ffbuf);
        complex128_t* Ubuf = calloc(POW2(t->nO), sizeof*Ubuf);
        #pragma omp for
        for (index_t kk=0; kk<t->my_nk; ++kk) {
            index_t q = t->idx_ibz_in_fullmesh[kk + t->my_nk_off];
            // TODO conjugate
            (conj_U ? transpose_conj_outofplace : transpose_outofplace)( Ubuf,
                    U+POW2(t->nO)*kidxc2f(q,t->nk,t->nkf), t->nO );
            ZERO_OUT( ffbuf, linsz * t->nO );
            for (index_t f=0; f<formfacs_sz; ++f) {
                // TODO value of the phase
                const claV3d_t R12 = {{formfacs[f].R12[0],formfacs[f].R12[1],formfacs[f].R12[2]}};
                const double qR12 = claV3d_dot(claV3d_map(t->pmod->internals->kmesh + 3*q),
                                               claM3d_vecmat(R12, claM3d_map(t->pmod->lattice[0])));
                ffbuf[IDX4(formfacs[f].s2,formfacs[f].s1,ph_vertex_to_mi_map[f],formfacs[f].OA,
                        t->n_spin,t->n_orbff,t->nO)] += fftw_cbuf[IDX2(f,q,nk)] * cexp(phase_pref*I*qR12);
            }
            single_gemm_serial( ffbuf, Ubuf, t->g_ob_qO+kk*linsz*t->nO, linsz, t->nO, t->nO, 1.0, 0.0 );
        }
        free( ffbuf );
        free( Ubuf );
    }

    // and do some allocation
    if (t->buffered) {
        t->Dch = calloc(t->my_nk * POW2(linsz), sizeof*t->Dch);
        t->Dch_zero = calloc(t->my_nk * POW2(linsz), sizeof*t->Dch_zero);
    }
    t->gbuf_zero = calloc(t->my_nk * t->nO, sizeof*t->gbuf_zero);
    t->gbuf = calloc(t->my_nk * t->nO, sizeof*t->gbuf);

    free( ph_vertex_to_mi_map );
    fftw_free(fftw_buf);
    VECTOR_FREE( formfacs );
}

void tu_data_phonons_gen_vert( tu_data_phonons_t* t, double Lambda,
        complex128_t* out, double fac ) {
    const index_t linsz = POW2(t->n_spin)*t->n_orbff,
                  nO = t->nO, my_nk = t->my_nk;

    int calc = 1;
    complex128_t* buf = out;
    if (t->buffered) {
        const int iszero = Lambda == 0.0;
        buf = iszero ? t->Dch_zero : t->Dch;
        double* last = iszero ? &t->Dch_zero_last_Lambda : &t->Dch_last_Lambda;
        if ((calc = (*last != Lambda))) {
            *last = Lambda;
            ZERO_OUT(buf, my_nk*POW2(linsz));
        }
    }

    if (calc) {
        const complex128_t *g_ph_buf = t->g_ob_qO,
                           *prop_buf = tu_data_phonons_gfill( t, Lambda );
        #pragma omp parallel for num_threads(diverge_omp_num_threads()) collapse(3)
        for (index_t kk=0; kk<my_nk; ++kk)
        for (index_t l=0; l<linsz; ++l)
        for (index_t r=0; r<linsz; ++r) {
            complex128_t tmp = 0.0;
            for (index_t b=0; b<nO; ++b) {
                complex128_t gl = g_ph_buf[IDX3(kk,l,b, linsz,nO)];
                if (conj_l) gl = conj(gl);
                complex128_t gr = g_ph_buf[IDX3(kk,r,b, linsz,nO)];
                if (conj_r) gr = conj(gr);
                // TODO it works without dispersion, conjugates
                tmp += gl * prop_buf[IDX2(kk,b, nO)] * gr;
            }
            buf[IDX3(kk,l,r, linsz,linsz)] += (t->buffered ? 1.0 : fac) * tmp;
        }
    }

    if (t->buffered) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<my_nk*POW2(linsz); ++i) out[i] += fac * buf[i];
    }
}

diverge_model_t* tu_data_phonons_get_model( tu_data_phonons_t* t ) { return t->pmod; }

