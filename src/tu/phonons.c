#define LINALG_NO_LAPACKE
#include "../misc/linalg.h"
LINALG_IMPLEMENT
#include "phonons.h"
#include "../diverge_internals_struct.h"
#include "../diverge_patching_helpers.h" // atomic_add
#include "../misc/mergesort.h"
#include "../misc/gemm.h"
#include <assert.h>
#include <fftw3.h>

struct tu_data_phonons_t {
    diverge_model_t* pmod;

    index_t nk[3];
    complex128_t* gbuf; // (ll, my_nk, nO, nO), with ll=(0,Lambda)
    double Lambda;
    index_t nO;

    ph_vertex_t* vtx;
    index_t n_vtx;

    complex128_t* g_ob_qO; // (my_nk, n_spin,n_spin,n_orbff, nO)
    complex128_t* tmp_g; // (my_nk, n_spin,n_spin,n_orbff, nO)

    index_t n_spin, n_orbff;
    index_t my_nk;
    index_t my_nk_off;
    const index_t* idx_ibz_in_fullmesh;
};

static void tu_data_phonons_init_vertex( tu_data_phonons_t* t, tu_data_t* d,
        const tu_formfactor_t* tu_ff, index_t n_tu_ff, index_t n_spin );
static void tu_data_phonons_gfill( tu_data_phonons_t* t, double Lambda );

tu_data_phonons_t* tu_data_phonons_init( diverge_model_t* pmod, diverge_model_t* elmod ) {
    tu_data_phonons_t* t = calloc(1, sizeof*t);
    t->pmod = pmod;
    memcpy( t->nk, t->pmod->nk, sizeof t->nk );
    t->nO = t->pmod->n_orb;
    if (!t->pmod->internals->has_common_internals)
        mpi_err_printf( "phonon model must have common internals!\n" );
    assert( t->pmod->n_spin == 1 );
    assert( t->pmod->internals->greens_shared == 0 );
    assert( t->pmod->gproj == NULL );
    t->vtx = (ph_vertex_t*)t->pmod->vert;
    t->n_vtx = t->pmod->n_vert;
    tu_data_phonons_init_vertex( t, elmod->internals->tu_data, elmod->tu_ff, elmod->n_tu_ff, elmod->n_spin );
    return t;
}

void tu_data_phonons_free( tu_data_phonons_t* t ) {
    free(t->gbuf);
    free(t->g_ob_qO);
    free(t->tmp_g);
    free(t);
}

static void tu_data_phonons_gfill( tu_data_phonons_t* t, double Lambda ) {
    if (Lambda == t->Lambda) return;
    t->Lambda = Lambda;
    const index_t nO = t->nO, my_nk = t->my_nk, my_nk_off = t->my_nk_off;
    index_t ngreens = kdim(t->nk)*POW2(nO);
    gf_complex_t *gbuf_p = t->pmod->internals->greens,
                 *gbuf_m = t->pmod->internals->greens + ngreens;

    (*t->pmod->gfill)( t->pmod, I*Lambda, gbuf_p );
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t kk=0; kk<my_nk; ++kk) {
        index_t q = t->idx_ibz_in_fullmesh[my_nk_off + kk];
        for (index_t O1=0; O1<nO; ++O1)
        for (index_t O2=0; O2<nO; ++O2)
            t->gbuf[IDX4(0,kk,O1,O2,my_nk,nO,nO)] = gbuf_p[IDX3(q,O1,O2,nO,nO)] + gbuf_m[IDX3(q,O1,O2,nO,nO)];
    }

    (*t->pmod->gfill)( t->pmod, I*DIVERGE_EPS_MESH, gbuf_p );
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t kk=0; kk<my_nk; ++kk) {
        index_t q = t->idx_ibz_in_fullmesh[my_nk_off + kk];
        for (index_t O1=0; O1<nO; ++O1)
        for (index_t O2=0; O2<nO; ++O2)
            t->gbuf[IDX4(1,kk,O1,O2,my_nk,nO,nO)] = gbuf_p[IDX3(q,O1,O2,nO,nO)] + gbuf_m[IDX3(q,O1,O2,nO,nO)];
    }
}

#undef MERGESORT_COMP
#define MERGESORT_COMP( i, j ) memcmp( &i, &j, offsetof(ph_vertex_t, R1) )
MERGESORT_IMPL( ph_vertex_sort, ph_vertex_t )

static inline int formfac_equal( const ph_vertex_t* a, const ph_vertex_t* b ) {
    return !memcmp(a, b, offsetof(ph_vertex_t, R1));
}

static inline index_t find_tu_ff( const tu_formfactor_t* tu_ff, index_t n_tu_ff,
        ph_vertex_t ff, const index_t* nk ) {
    index_t idx = -1;
    for (int d=0; d<3; ++d) while (ff.R12[d] < 0) ff.R12[d] += nk[d]; // make positive
    for (index_t i=0; i<n_tu_ff; ++i) {
        if (tu_ff[i].oto == ff.o2 && tu_ff[i].ofrom == ff.o1 && tu_ff[i].R[0] == ff.R12[0] &&
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
    ph_vertex_t formfac = { .o1=-1, .s1=-1, .o2=-1, .s2=-1, .O1=-1 };
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
    // XXX do we need to do an IFFT?
    fftw_plan p = fftw_plan_guru64_dft(3, dims, 1, &howmany, fftw_buf, fftw_buf, -1, FFTW_ESTIMATE);
    ZERO_OUT(fftw_buf, nk*formfacs_sz);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t ff=0; ff<formfacs_sz; ++ff) {
        for (index_t cc=0; cc<formfacs_count[ff]; ++cc) {
            index_t f = formfacs_displ[ff] + cc;
            const index_t* R = t->vtx[f].R1;
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

    // write out the objects to g_ob_qO;
    t->g_ob_qO = calloc(t->my_nk * POW2(t->n_spin) * t->n_orbff * t->nO, sizeof*t->g_ob_qO);
    t->tmp_g = calloc(t->my_nk * POW2(t->n_spin) * t->n_orbff * t->nO, sizeof*t->tmp_g);
    t->gbuf = calloc(2 * t->my_nk * POW2(t->nO), sizeof*t->gbuf);
    int n_not_found = 0;
    #pragma omp parallel for num_threads(diverge_omp_num_threads()) reduction(+:n_not_found)
    for (index_t f=0; f<formfacs_sz; ++f) {
        // find tu_ff index
        index_t mapped_tu = find_tu_ff( tu_ff, n_tu_ff, formfacs[f], t->nk );
        if (mapped_tu == -1) { n_not_found++; continue; }

        // find mi index
        index_t mapped_mi = -1;
        for (index_t i=0; i<t->n_orbff; ++i) if (d->mi_to_tuffidx[i] == mapped_tu) {
            mapped_mi = i; break;
        }
        if (mapped_mi == -1) { n_not_found++; continue; }
        // map to g

        const double* qmesh = t->pmod->internals->kmesh;
        const claM3d_t lat = claM3d_map(t->pmod->lattice[0]);
        for (index_t kk=0; kk<t->my_nk; ++kk) {
            index_t q = t->idx_ibz_in_fullmesh[kk + t->my_nk_off];
            ph_vertex_t ff = formfacs[f];
            claV3d_t qvec = claV3d_map(qmesh+3*q),
                     R12 = claM3d_vecmat((claV3d_t){{ff.R12[0],ff.R12[1],ff.R12[2]}}, lat),
                     R1 = claM3d_vecmat((claV3d_t){{ff.R1[0],ff.R1[1],ff.R1[2]}}, lat);
            // XXX is R12 sufficient? Or do we need a different phase factor
            // once we have some kind of dispersive phonon?
            (void)R1;
            const complex128_t to_add = fftw_cbuf[IDX2(f,q,nk)] * cexp(0.5*I*claV3d_dot(qvec,R12));
            atomic_add( &t->g_ob_qO[IDX5(kk,formfacs[f].s2,formfacs[f].s1,mapped_mi,formfacs[f].O1,
                               t->n_spin,t->n_spin,t->n_orbff,t->nO)], &to_add );
        }
    }
    diverge_mpi_allreduce_int_max_inplace( &n_not_found, 1 );
    if (n_not_found > 0)
        mpi_err_printf( "couldnt find <=%i tu_ff for elph, try larger max_dist\n", n_not_found );

    fftw_free(fftw_buf);
    VECTOR_FREE( formfacs );
}

static void phonon_gemm( const complex128_t* g, const complex128_t* D,
        complex128_t* V, complex128_t* tmp, index_t linsz, index_t nO ) {
    single_gemm( g, D, tmp, nO, linsz, nO, 1.0, 0.0 );
    single_gemm_adjoint( tmp, g, V, linsz, linsz, nO, 1.0, 0.0 );
}

void tu_data_phonons_vertices( tu_data_phonons_t* t, double Lambda,
        complex128_t* V_Lambda, complex128_t* V_zero ) {
    tu_data_phonons_gfill( t, Lambda );
    complex128_t *g_Lambda = t->gbuf,
                 *g_Zero = t->gbuf + t->my_nk * POW2(t->nO);
    index_t linsz = POW2(t->n_spin)*t->n_orbff;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t kk=0; kk<t->my_nk; ++kk) {
        complex128_t *g = t->g_ob_qO + kk*linsz*t->nO,
                     *D_Lambda = g_Lambda + kk*POW2(t->nO),
                     *tmp = t->tmp_g + kk*linsz*t->nO,
                     *D_Zero = g_Zero + kk*POW2(t->nO);
        if (V_Lambda) phonon_gemm( g, D_Lambda, V_Lambda + kk*POW2(linsz), tmp, linsz, t->nO );
        if (V_zero)   phonon_gemm( g, D_Zero, V_zero + kk*POW2(linsz), tmp, linsz, t->nO );
    }
}
