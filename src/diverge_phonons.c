/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "misc/linalg.h"
LINALG_IMPLEMENT

#include "diverge_phonons.h"
#include "diverge_common.h"
#include "diverge_model.h"
#include "misc/mpi_functions.h"
#include "tu/phonons.h"
#include "tu/phonons_details.h"

#include "misc/shared_mem.h"
#include "misc/bitary.h"
#include "misc/mergesort.h"
#include "misc/qnan.h"

#include "diverge_model.h"
#include "diverge_model_internals.h"
#include "diverge_internals_struct.h"
#include <assert.h>
#include <stdatomic.h>

rs_hopping_t* diverge_phonons_asr( rs_hopping_t* ph_hop, index_t* n_ph_hop ) {
    index_t n_hop = *n_ph_hop;
    index_t n_orb = 0;
    for (index_t i=0; i<n_hop; ++i) {
        index_t om = MAX(ph_hop[i].o1, ph_hop[i].o2);
        n_orb = MAX(n_orb, om);
    }
    n_orb = (n_orb+1);
    mpi_vrb_printf( "found %li=(%li,3) orbitals\n", n_orb, n_orb/3 );
    return diverge_phonons_asr_n_orb( ph_hop, n_ph_hop, n_orb );
}

static _Atomic int _print_massdiv = 0;

rs_hopping_t* diverge_phonons_asr_n_orb( rs_hopping_t* ph_hop, index_t* n_ph_hop,
        index_t n_orb ) {
    mpi_log_printf( "ASR (hoppings) with phonon orb index -> (site, xyz)\n" );
    index_t n_hop = *n_ph_hop;
    n_orb /= 3;

    complex128_t* self_forces = calloc(n_orb*9, sizeof*self_forces);
    for (index_t i=0; i<n_hop; ++i) {
        int ho1 = ph_hop[i].o1/3, hr1 = ph_hop[i].o1%3, hr2 = ph_hop[i].o2%3;
        self_forces[IDX3(ho1,hr1,hr2, 3,3)] += ph_hop[i].t;
    }

    ph_hop = realloc(ph_hop, (*n_ph_hop = (n_hop + (n_orb*9))) * sizeof*ph_hop);
    double forcemax = 0.0;
    for (index_t o=0; o<n_orb; ++o)
    for (index_t r1=0; r1<3; ++r1)
    for (index_t r2=0; r2<3; ++r2) {
        ph_hop[n_hop + IDX3(o,r1,r2, 3,3)] = (rs_hopping_t){
            .o1=IDX2(o,r1,3), .o2=IDX2(o,r2,3),
            .t=-self_forces[IDX3(o,r1,r2, 3,3)],
        };
        forcemax = MAX(forcemax, cabs(self_forces[IDX3(o,r1,r2, 3,3)]));
    }
    int print_massdiv = atomic_load(&_print_massdiv);
    mpi_log_printf( "ASR (hoppings%s) maximal restoring force value: %.5e\n", print_massdiv?"*masses":"", forcemax );

    free(self_forces);
    return ph_hop;
}

void diverge_phonons_apply_asr( diverge_model_t* m ) {
    m->hop = diverge_phonons_asr_n_orb( m->hop, &m->n_hop, m->n_orb );
}

void* diverge_phonons_apply_sgnsqrt( diverge_model_t* m ) {
    assert( m->n_spin == 1 );
    double* E = diverge_model_internals_get_E(m);
    const index_t nkb = kdimtot(m->nk,m->nkf)*m->n_orb;
    if (!m->internals->model_phonons_imag_freq)
        m->internals->model_phonons_imag_freq = bitary_alloc(nkb);
    if (shared_exclusive_enter(E)) for (index_t k=0; k<nkb; ++k) {
        int isimag = (E[k] < -DIVERGE_EPS_MESH);
        if (isimag)
            bitary_set(m->internals->model_phonons_imag_freq, k);
        E[k] = diverge_sgnsqrt(E[k]);
    }
    shared_exclusive_wait(E);

    index_t sz = shared_malloc_size();
    diverge_mpi_allgather_index_inplace(&sz, 1);
    if (sz > 1) {
        diverge_mpi_allreduce_64_bor_inplace(
                m->internals->model_phonons_imag_freq,
                bitary_nwords(nkb)
        );
    }

    size_t nimag = bitary_count_parallel(m->internals->model_phonons_imag_freq,
            nkb, diverge_omp_num_threads());
    if (nimag)
        mpi_wrn_printf("found %lu imaginary phonon frequencies\n", nimag);
    return m->internals->model_phonons_imag_freq;
}

void diverge_phonons_sgnsqrt( double* E, index_t nk, index_t nb, index_t nb_ignore ) {
    const index_t nb_cut = nb-nb_ignore;
    index_t n_imag = 0;
    #pragma omp parallel for num_threads(diverge_omp_num_threads()) collapse(2) reduction(+:n_imag)
    for (index_t k=0; k<nk; ++k)
    for (index_t b=0; b<nb_cut; ++b) {
        const index_t kb = IDX2(k,b,nb);
        int isimag = E[kb] < -DIVERGE_EPS_MESH;
        E[kb] = diverge_sgnsqrt(E[kb]);
        if (isimag) {
            E[kb] = fnan_gen(E[kb]);
            n_imag++;
        }
    }
    if (n_imag)
        mpi_wrn_printf("found %li imaginary phonon frequencies\n", n_imag);
}

void* diverge_phonons_get_ifrq( diverge_model_t* m ) {
    return m->internals->model_phonons_imag_freq;
}

complex128_t* diverge_phonons_get_g( diverge_model_t* m, int refined ) {
    tu_data_phonons_t* p = m->internals->tu_data_phonons;
    if (!p) return NULL;
    if (refined > 0) return p->g_ob_qO_refined;
    if (refined < 0) return p->g_ob_qO_fullmesh;
    return p->g_ob_qO;
}

void diverge_phonons_get_g_dim( diverge_model_t* m, index_t g_dim[3] ) {
    tu_data_phonons_t* p = m->internals->tu_data_phonons;
    if (!p) return;
    if (!g_dim) return;
    tu_data_phonons_get_g_dim(p, 0, g_dim);
}

double diverge_phonons_change_regulator( double eta ) {
    return tu_data_phonons_change_regulator( eta );
}

ph_vertex_t* diverge_phonons_asr_vertex( ph_vertex_t* vtx, index_t* n_vtx ) {
    return diverge_phonons_asr_vertex_n_orb( vtx, n_vtx, -1 );
}

#undef MERGESORT_COMP
static inline int ph_vertex_asr_cmp( ph_vertex_t A, ph_vertex_t B ) {
    A.OA %= 3;
    B.OA %= 3;
    return memcmp(&A, &B, offsetof(ph_vertex_t, RA));
}
#define MERGESORT_COMP ph_vertex_asr_cmp
MERGESORT_IMPL( ph_vertex_asr_sort, ph_vertex_t )

ph_vertex_t* diverge_phonons_asr_vertex_n_orb( ph_vertex_t* vtx, index_t* n_vtx, index_t n_orb ) {
    mpi_log_printf( "ASR (vertices) with phonon orb index -> (site, xyz)\n" );
    // sort vertices according to the comparison function ph_vertex_asr_cmp
    ph_vertex_asr_sort(vtx, *n_vtx, diverge_omp_num_threads(), 0);

    int find_orbs = (n_orb <= -1);
    if (find_orbs) {
        n_orb = 0;
        #pragma omp parallel for num_threads(diverge_omp_num_threads()) reduction(max:n_orb)
        for (index_t v=0; v<*n_vtx; ++v) n_orb = MAX(n_orb, vtx[v].OA);
        n_orb = (n_orb + 1)/3;
        mpi_vrb_printf( "found (%li,3) orbitals\n", n_orb );
    } else {
        n_orb /= 3;
    }

    VECTOR_INIT(ph_vertex_t, asr_elem);
    ph_vertex_t* prev = NULL;
    for (index_t i=0; i<*n_vtx; ++i) {
        if (!prev || ph_vertex_asr_cmp(*prev, vtx[i])) {
            VECTOR_PUSH_BACK(asr_elem, vtx[i]);
            prev = &vtx[i];
            asr_elem[asr_elem_sz-1].OA %= 3;
            asr_elem[asr_elem_sz-1].RA[0] = asr_elem[asr_elem_sz-1].RA[1] = asr_elem[asr_elem_sz-1].RA[2] = 0;
        } else {
            asr_elem[asr_elem_sz-1].g += vtx[i].g;
        }
    }
    VECTOR_SHRINK_TO_FIT(asr_elem);

    VECTOR_INIT(ph_vertex_t, asr_append);
    double maxval = 0.0;
    for (index_t a=0; a<asr_elem_sz; ++a)
    for (index_t o=0; o<n_orb; ++o) {
        ph_vertex_t pv = asr_elem[a];
        pv.OA = IDX2(o,pv.OA,3);
        pv.g *= -1./(double)n_orb;
        maxval = MAX(maxval, cabs(pv.g));
        VECTOR_PUSH_BACK(asr_append, pv);
    }
    VECTOR_FREE(asr_elem);
    VECTOR_SHRINK_TO_FIT(asr_append);

    int print_massdiv = atomic_load(&_print_massdiv);
    mpi_log_printf( "ASR (vertices%s) maximal restoring force value: %.5e\n", print_massdiv?"*masses":"", maxval );

    vtx = realloc(vtx, sizeof*vtx * (*n_vtx + asr_append_sz));
    memcpy(vtx + (*n_vtx), asr_append, sizeof*vtx * asr_append_sz);
    *n_vtx += asr_append_sz;
    VECTOR_FREE(asr_append);

    return vtx;
}

double diverge_phonons_enable_refinement( diverge_model_t* el, int enable_refinement, double eps_mesh ) {
    el->internals->tu_phonons_refined = enable_refinement;
    return tu_data_phonons_change_refinement_eps(eps_mesh);
}

refinement_t* diverge_phonons_get_refine( diverge_model_t* m, index_t* n_refine ) {
    tu_data_phonons_t* p = m->internals->tu_data_phonons;
    if (!p) return NULL;
    if (n_refine) *n_refine = p->n_refine;
    return p->refine;
}

void diverge_phonons_set_vertex_generator( diverge_model_t* m,
        diverge_phonon_vertex_generator_t V, diverge_flow_step_t* step, void* ctx ) {
    tu_data_phonons_t* p = m->internals->tu_data_phonons;
    if (!p) return;
    tu_data_phonons_set_vertex_generator( p, V, step, ctx );
}

typedef struct {
    int32_t v1:11;
    int32_t v2:11;
    int32_t v3:10;
} V3i32_t;
typedef struct {
    V3i32_t R12;
    V3i32_t RA;
} Rvecs_t;
static inline index_t Rvecs_cmp( Rvecs_t A, Rvecs_t B ) {
    assert(sizeof(Rvecs_t) == sizeof(index_t));
    index_t iA = 0; memcpy(&iA, &A, sizeof(iA));
    index_t iB = 0; memcpy(&iB, &B, sizeof(iB));
    return iA - iB;
}
#define vtree_key_t Rvecs_t
#define vtree_val_t index_t
#define vtree_padding_bytes 4
#define vtree_key_cmp(A,B) Rvecs_cmp(A,B)
#include "misc/vtree.h"

static inline Rvecs_t ph_vertex_to_Rvecs( ph_vertex_t pv ) {
    return (Rvecs_t){ .R12={.v1=pv.R12[0], .v2=pv.R12[1], .v3=pv.R12[2]},
                      .RA ={.v1= pv.RA[0], .v2= pv.RA[1], .v3= pv.RA[2]} };
}

static inline claV3i_t claV3d_to_i( claV3d_t R ) {
    return (claV3i_t){{lroundl(R.v[0]),lroundl(R.v[1]),lroundl(R.v[2])}};
}

typedef struct {
    complex128_t* g_array;
    index_t n_os;
    index_t n_sp;
    index_t n_OA;
    VECTOR_DECL(ph_vertex_t, pv_sym)
} Rtree_iter_t;
static inline void Rtree_g_array_push( vtree_node_t* node, void* ctx_ ) {
    Rtree_iter_t* ctx = ctx_;
    index_t iR = node->val;
    Rvecs_t Rv = node->key;
    ph_vertex_t pv = {.R12={Rv.R12.v1,Rv.R12.v2,Rv.R12.v3},
                      .RA={Rv.RA.v1,Rv.RA.v2,Rv.RA.v3}};
    complex128_t* g_array = ctx->g_array;
    for (index_t o1=0; o1<ctx->n_os; ++o1)
    for (index_t o2=0; o2<ctx->n_os; ++o2)
    for (index_t OA=0; OA<ctx->n_OA; ++OA) {
        index_t g_idx = IDX4(iR, o1,o2,OA, ctx->n_os,ctx->n_os,ctx->n_OA);
        if (creal(g_array[g_idx]) != 0.0 || cimag(g_array[g_idx]) != 0.0) {
            pv.o1 = o1/ctx->n_sp;
            pv.s1 = o1%ctx->n_sp;
            pv.o2 = o2/ctx->n_sp;
            pv.s2 = o2%ctx->n_sp;
            pv.OA = OA;
            pv.g = g_array[g_idx];
            VECTOR_PUSH_BACK(ctx->pv_sym, pv);
        }
    }
}

void diverge_phonons_symmetrize_epc( diverge_model_t* el, diverge_model_t* ph ) {
    const index_t n_os = el->n_orb * el->n_spin,
                  n_sp = el->n_spin;
    const index_t n_OA = ph->n_orb;
    const index_t n_sym = el->n_sym;
    if (ph->n_sym != el->n_sym) {
        mpi_err_printf( "must have same # of symmetries (el,ph)=(%li,%li)\n", el->n_sym, ph->n_sym );
        return;
    }

    ph_vertex_t* pv = (ph_vertex_t*)ph->vert;
    index_t n_pv = ph->n_vert;

    xvtree_t* Rtree = xvtree_init( MAX(n_pv/POW2(n_os)/n_OA, 1024) );
    index_t n_Rv = 0;
    for (index_t v=0; v<n_pv; ++v) {
        Rvecs_t Rv = ph_vertex_to_Rvecs(pv[v]);
        vtree_node_t* Rv_node = xvtree_find(Rtree, Rv);
        if (!Rv_node) {
            Rv_node = xvtree_insert(Rtree, Rv);
            Rv_node->val = n_Rv++;
        }
    }
    mpi_vrb_printf("found %li unique Rvector combis before %li sym\n", n_Rv, n_sym);

    index_t n_Rv_cap = n_Rv;
    complex128_t* g_array = calloc(n_Rv * POW2(n_os) * n_OA, sizeof*g_array);
    double norm = 1./(double)n_sym;

    const double *el_symm_beyond_UC = el->internals->symm_beyond_UC,
                 *ph_symm_beyond_UC = ph->internals->symm_beyond_UC;
    const complex128_t *el_symm_pref = el->internals->symm_pref,
                       *ph_symm_pref = ph->internals->symm_pref;
    const index_t *el_symm_orb_off = el->internals->symm_orb_off,
                  *el_symm_orb_len = el->internals->symm_orb_len,
                  *el_symm_orb_map = el->internals->symm_map_orb,
                  *ph_symm_orb_off = ph->internals->symm_orb_off,
                  *ph_symm_orb_len = ph->internals->symm_orb_len,
                  *ph_symm_orb_map = ph->internals->symm_map_orb;
    const claM3d_t el_inv_L = claM3d_scale(claM3d_map(el->internals->rlattice.v[0]), 1./(2.*M_PI)),
                   ph_inv_L = claM3d_scale(claM3d_map(ph->internals->rlattice.v[0]), 1./(2.*M_PI)),
                   el_L = claM3d_map(el->lattice[0]),
                   ph_L = claM3d_map(ph->lattice[0]);
    for (index_t v=0; v<n_pv; ++v)
    for (index_t s=0; s<n_sym; ++s) {
        const index_t el_o1 = IDX2(pv[v].o1,pv[v].s1, n_sp),
                      el_o2 = IDX2(pv[v].o2,pv[v].s2, n_sp);
        const index_t el_osym1 = IDX2(el_o1,s, n_sym),
                      el_osym2 = IDX2(el_o2,s, n_sym),
                      ph_osym = IDX2(pv[v].OA,s, n_sym);
        const double *d_R1 = el_symm_beyond_UC+3*IDX2(pv[v].o1,s, n_sym),
                     *d_R2 = el_symm_beyond_UC+3*IDX2(pv[v].o2,s, n_sym),
                     *d_RA = ph_symm_beyond_UC+3*IDX2(pv[v].OA,s, n_sym);
        const claV3d_t dVR1 = claM3d_matvec(el_inv_L, claV3d_map(d_R1)),
                       dVR2 = claM3d_matvec(el_inv_L, claV3d_map(d_R2)),
                       dVRA = claM3d_matvec(ph_inv_L, claV3d_map(d_RA));
        const claV3i_t dR1_i = claV3d_to_i(dVR1),
                       dR2_i = claV3d_to_i(dVR2),
                       dRA_i = claV3d_to_i(dVRA);
        const claV3d_t R12_rot = claM3d_matvec(
            claM3d_map(el->rs_symmetries[s][0]),
            claM3d_vecmat((claV3d_t){{pv[v].R12[0],pv[v].R12[1],pv[v].R12[2]}}, el_L)
        );
        const claV3d_t RA_rot  = claM3d_matvec(
            claM3d_map(ph->rs_symmetries[s][0]),
            claM3d_vecmat((claV3d_t){{pv[v].RA[0], pv[v].RA[1], pv[v].RA[2] }}, ph_L)
        );
        const claV3i_t R12_rot_i = claV3d_to_i(claM3d_matvec(el_inv_L, R12_rot)),
                       RA_rot_i  = claV3d_to_i(claM3d_matvec(ph_inv_L, RA_rot));
        const claV3i_t dR12 = claV3i_sub(dR2_i,dR1_i),
                       dRA  = claV3i_sub(dRA_i,dR1_i);
        const claV3i_t R12_new = claV3i_add(R12_rot_i, dR12),
                       RA_new  = claV3i_add(RA_rot_i,  dRA);
        const Rvecs_t Rv = { .R12={.v1=R12_new.v[0], .v2=R12_new.v[1], .v3=R12_new.v[2]},
                             .RA ={.v1= RA_new.v[0], .v2= RA_new.v[1],. v3= RA_new.v[2]} };
        vtree_node_t* Rv_node = xvtree_find(Rtree, Rv);
        if (!Rv_node) {
            Rv_node = xvtree_insert(Rtree, Rv);
            Rv_node->val = n_Rv++;
            if (n_Rv > n_Rv_cap) { // push back
                n_Rv_cap *= 2;
                g_array = realloc(g_array, sizeof*g_array * n_Rv_cap * POW2(n_os) * n_OA);
                ZERO_OUT(g_array + (n_Rv-1)*POW2(n_os)*n_OA, (n_Rv_cap-(n_Rv-1))*POW2(n_os)*n_OA);
            }
        }
        const index_t iR = Rv_node->val;
        for (index_t i=0; i<el_symm_orb_len[el_osym1]; ++i)
        for (index_t j=0; j<el_symm_orb_len[el_osym2]; ++j)
        for (index_t k=0; k<ph_symm_orb_len[ph_osym]; ++k) {
            const index_t xo1 = el_symm_orb_off[el_osym1] + i,
                          xo2 = el_symm_orb_off[el_osym2] + j,
                          xOA = ph_symm_orb_off[ph_osym] + k;
            g_array[IDX4(iR,el_symm_orb_map[xo1],el_symm_orb_map[xo2],ph_symm_orb_map[xOA],
                    n_os,n_os,n_OA)] += pv[v].g * norm * el_symm_pref[xo1] *
                                                    conj(el_symm_pref[xo2]) * ph_symm_pref[xOA];
        }
    }
    g_array = realloc(g_array, sizeof*g_array * n_Rv * POW2(n_os) * n_OA); // shrink

    mpi_vrb_printf("found %li unique Rvector combis after %li sym\n", xvtree_size(Rtree), n_sym);

    Rtree_iter_t it = {.g_array=g_array, .n_os=n_os, .n_sp=n_sp, .n_OA=n_OA};
    xvtree_iter( Rtree, &Rtree_g_array_push, &it );
    free( g_array );
    xvtree_free( Rtree );

    free(ph->vert);
    ph->vert = (rs_vertex_t*)it.pv_sym;
    ph->n_vert = it.pv_sym_sz;
    return;
}

void diverge_phonons_fetch_sym( diverge_model_t* el, diverge_model_t* ph ) {
    const index_t n_sym = labs(el->n_sym);
    if (n_sym == 0) return;
    const index_t n_phorb = ph->n_orb / 3;
    ph->orb_symmetries = calloc(n_sym * POW2(n_phorb*3), sizeof*ph->orb_symmetries);
    for (index_t s=0; s<n_sym; ++s) {
        memcpy(ph->rs_symmetries[s][0], el->rs_symmetries[s][0], sizeof(double)*9);
        for (index_t o1=0; o1<n_phorb; ++o1)
        for (index_t x1=0; x1<3; ++x1)
        for (index_t o2=0; o2<n_phorb; ++o2)
        for (index_t x2=0; x2<3; ++x2)
            ph->orb_symmetries[IDX5(s, o1,x1, o2,x2, n_phorb,3,n_phorb,3)] = ph->rs_symmetries[s][x1][x2];
    }
    ph->n_sym = n_sym;
}

rs_hopping_t* diverge_phonons_asr_n_orb_massdiv( rs_hopping_t* ph_hop,
        index_t* n_ph_hop, index_t n_orb, const double* masses ) {
    if (masses) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t h=0; h<*n_ph_hop; ++h)
            ph_hop[h].t *= sqrt(masses[ph_hop[h].o1] * masses[ph_hop[h].o2]);
        atomic_store(&_print_massdiv, 1);
    }
    ph_hop = diverge_phonons_asr_n_orb(ph_hop, n_ph_hop, n_orb);
    if (masses) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t h=0; h<*n_ph_hop; ++h)
            ph_hop[h].t /= sqrt(masses[ph_hop[h].o1] * masses[ph_hop[h].o2]);
        atomic_store(&_print_massdiv, 0);
    }
    return ph_hop;
}

ph_vertex_t* diverge_phonons_asr_vertex_n_orb_massdiv( ph_vertex_t* vtx,
        index_t* n_vtx, index_t n_orb, const double* masses ) {
    if (masses) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t v=0; v<*n_vtx; ++v) vtx[v].g *= sqrt(masses[vtx[v].OA]);
        atomic_store(&_print_massdiv, 1);
    }
    vtx = diverge_phonons_asr_vertex_n_orb(vtx, n_vtx, n_orb);
    if (masses) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t v=0; v<*n_vtx; ++v) vtx[v].g /= sqrt(masses[vtx[v].OA]);
        atomic_store(&_print_massdiv, 0);
    }
    return vtx;
}
