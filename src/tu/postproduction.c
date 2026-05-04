/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#define LINALG_NO_LAPACKE
#include "../misc/linalg.h"
LINALG_IMPLEMENT

#include "postproduction.h"
#include "vertex_details.h"
#include "../misc/version.h"
#include "../misc/batched_eigen.h"
#include "../misc/batched_svd.h"
#include "../misc/batched_gemms.h"
#include "../misc/eigen.h"
#include "../misc/gemm.h"
#include "../misc/ham2hop.h"
#include "../diverge_post.h"
#include "rpa.h"

#include <assert.h>
#include <stddef.h>

typedef struct tu_postproduction_t tu_postproduction_t;

// main functions
static void lingap_ff( tu_postproduction_t* p );
static void calc_susc( tu_postproduction_t* p );
static void save_channel_leading_ev( tu_postproduction_t* p );
static void to_file( tu_postproduction_t* p, const char* filename );

// helper functions
static bool check_hermiticity( complex128_t* data, index_t linsize, double lim/*=1.e-11*/ );
static void _save_channel_leading_ev( tu_postproduction_t* p, complex128_t* channel,
        complex128_t* vec, double* val, char* type, index_t* len, index_t* off );
static void _susc_o14k(const diverge_model_t* model, const complex128_t* inp, complex128_t* susc);
static inline void fill_Rvec_from_vec( const diverge_model_t* model, const claV3i_t* Rptr, double* Rvec );
static void construct_multiindex_maps( tu_postproduction_t* p );

static const index_t rs_hopping_dtype[] = { offsetof(rs_hopping_t, R),
    offsetof(rs_hopping_t, o1), offsetof(rs_hopping_t, o2),
    offsetof(rs_hopping_t, s1), offsetof(rs_hopping_t, s2),
    offsetof(rs_hopping_t, t), sizeof(rs_hopping_t), };

struct tu_postproduction_t {
    tu_postproduction_conf_t c;

    index_t* lenP;
    index_t* lenC;
    index_t* lenD;
    index_t* offP;
    index_t* offC;
    index_t* offD;
    char* Ptype;
    char* Ctype;
    char* Dtype;
    double* Pval;
    double* Cval;
    double* Dval;
    complex128_t* Pvec;
    complex128_t* Cvec;
    complex128_t* Dvec;

    double* Sgap_sc;
    double* Sgap_mag;
    double* Sgap_charge;
    complex128_t* Ugap_sc;
    complex128_t* Ugap_mag;
    complex128_t* Ugap_charge;
    complex128_t* Vgap_sc;
    complex128_t* Vgap_mag;
    complex128_t* Vgap_charge;

    complex128_t* suscPp;
    complex128_t* suscCh;
    complex128_t* suscMa;

    complex128_t* suscPp_ff;
    complex128_t* suscCh_ff;
    complex128_t* suscMa_ff;

    /* private */
    VECTOR_DECL( index_t, mi_map_idx )
    VECTOR_DECL( complex128_t, mi_map_pref_in_in )
    VECTOR_DECL( complex128_t, mi_map_pref_in_out )

    index_t tu_n_singular_valuesP, tu_n_singular_valuesC, tu_n_singular_valuesD;
    index_t tu_suscP, tu_suscC, tu_suscD;
    index_t tu_suscP_ff, tu_suscC_ff, tu_suscD_ff;

    bool called_lingap;
    bool called_suscep_ff;
    bool called_suscep_full;

    tu_vertex_t* vertex;
    tu_vertex_t* internal_vertex;
    tu_projection_t* proj;
    tu_rpa_t* rpa;
    const diverge_model_t* model;
};

void tu_postproduction( tu_vertex_t* vertex, tu_vertex_t* internal_vertex,
        tu_projection_t* proj, tu_rpa_t* rpa, const diverge_model_t* model,
        const tu_postproduction_conf_t* c, const char* filename ) {
    tu_postproduction_t p = {
        .c = c ? *c : tu_postproduction_conf_defaults,
        .vertex = vertex,
        .internal_vertex = internal_vertex,
        .proj = proj,
        .rpa = rpa,
        .model = model,
    };

    if (!p.c.skip_channel_calc) {
        index_t linsize = vertex->n_orbff*POW2(vertex->n_spin);
        p.Pval = calloc(linsize*p.vertex->my_nk,sizeof(double));
        p.Cval = calloc(linsize*p.vertex->my_nk,sizeof(double));
        p.Dval = calloc(linsize*p.vertex->my_nk,sizeof(double));
        p.Ptype = calloc(p.vertex->my_nk,sizeof(char));
        p.Ctype = calloc(p.vertex->my_nk,sizeof(char));
        p.Dtype = calloc(p.vertex->my_nk,sizeof(char));
        p.Pvec = p.internal_vertex->Pch;
        p.Cvec = p.internal_vertex->Cch;
        p.Dvec = p.internal_vertex->Dch;

        p.lenP = calloc(p.vertex->my_nk,sizeof(index_t));
        p.lenC = calloc(p.vertex->my_nk,sizeof(index_t));
        p.lenD = calloc(p.vertex->my_nk,sizeof(index_t));
        p.offP = calloc(p.vertex->my_nk,sizeof(index_t));
        p.offC = calloc(p.vertex->my_nk,sizeof(index_t));
        p.offD = calloc(p.vertex->my_nk,sizeof(index_t));
    }

    // perform
    if (p.rpa) tu_rpa_post( p.rpa, p.c.rpa_chans, p.c.rpa_nev, p.c.rpa_ncv,
            p.c.rpa_maxiter, p.c.rpa_tol );
    if (!p.c.skip_channel_calc) save_channel_leading_ev(&p);
    if (p.c.lingaps) lingap_ff(&p);
    if (p.c.susc_ff || p.c.susc_full) calc_susc(&p);
    to_file( &p, filename );

    free(p.Pval);
    free(p.Cval);
    free(p.Dval);
    free(p.Ptype);
    free(p.Ctype);
    free(p.Dtype);
    free(p.lenP);
    free(p.lenC);
    free(p.lenD);
    free(p.offP);
    free(p.offC);
    free(p.offD);
    free(p.Sgap_sc);
    free(p.Sgap_mag);
    free(p.Sgap_charge);
    free(p.Ugap_sc);
    free(p.Ugap_mag);
    free(p.Ugap_charge);
    free(p.Vgap_sc);
    free(p.Vgap_mag);
    free(p.Vgap_charge);
    free(p.suscCh);
    free(p.suscMa);
    free(p.suscPp);
    free(p.suscCh_ff);
    free(p.suscMa_ff);
    free(p.suscPp_ff);
    VECTOR_FREE( p.mi_map_idx );
    VECTOR_FREE( p.mi_map_pref_in_in );
    VECTOR_FREE( p.mi_map_pref_in_out );
}

static void save_channel_leading_ev( tu_postproduction_t* p ) {
    // use projection helper as vertex buffer in case channel_calc_project is
    // on. Otherwise keep this NULL and fall back to the respective channel in
    // the _save_channel_leading_ev call
    complex128_t* Vbuf = p->c.channel_calc_project ? p->vertex->projection_helper : NULL;
    index_t V_sz = p->vertex->full_vert_size;
    if (Vbuf) ZERO_OUT( Vbuf, V_sz );
    if (p->vertex->P_flow) {
        if (p->c.channel_calc_project) P_projection(Vbuf, p->vertex, p->proj);
        _save_channel_leading_ev(p, Vbuf ? Vbuf : p->vertex->Pch, p->Pvec, p->Pval, p->Ptype, p->lenP, p->offP);
    }
    if (p->vertex->C_flow) {
        if (p->c.channel_calc_project) {
            C_projection(Vbuf, p->vertex, p->proj);
            if (p->c.channel_calc_project < 0) for (index_t i=0; i<V_sz; ++i) Vbuf[i] *= -1.0;
        }
        _save_channel_leading_ev(p, Vbuf ? Vbuf : p->vertex->Cch, p->Cvec, p->Cval, p->Ctype, p->lenC, p->offC);
    }
    if (p->vertex->D_flow) {
        if (p->c.channel_calc_project > 0 || (p->c.channel_calc_project < 0 && !p->vertex->SU2)) {
            D_projection(Vbuf, p->vertex, p->proj);
        } else if (p->c.channel_calc_project < 0 && p->vertex->SU2) {
            if (p->vertex->C_flow) {
                C_projection(Vbuf, p->vertex, p->proj);
                for (index_t i=0; i<V_sz; ++i) Vbuf[i] *= -1.0;
            } else {
                for (index_t i=0; i<V_sz; ++i) Vbuf[i] = 0.0;
            }
            complex128_t* helper = calloc(p->vertex->full_vert_size,sizeof(complex128_t));
            D_projection(helper, p->vertex, p->proj);
            for (index_t i=0; i<V_sz; ++i) Vbuf[i] += 2.*helper[i];
            free( helper );
        }
        _save_channel_leading_ev(p, Vbuf ? Vbuf : p->vertex->Dch, p->Dvec, p->Dval, p->Dtype, p->lenD, p->offD);
    }
}

static void _save_channel_leading_ev( tu_postproduction_t* p, complex128_t* channel,
        complex128_t* vec, double* val, char* type, index_t* len, index_t* off ) {

    index_t linsize = p->vertex->n_orbff*POW2(p->vertex->n_spin);
    double* _val = malloc(linsize*sizeof(double));
    complex128_t* _vec = p->vertex->product_helper;

    double* allval = NULL;
    if (p->c.algo == 'b' || p->c.algo == 'B') {
        allval = calloc(linsize*p->vertex->my_nk, sizeof(double));
        memcpy(vec, channel, sizeof(complex128_t)*POW2(linsize)*p->vertex->my_nk);
        if (p->c.algo == 'b')
            batched_eigen_r( NULL, -2, vec, allval, linsize, p->vertex->my_nk );
        else
            #ifdef TU_POST_USE_BATCHED_SVD_U
            batched_svd( NULL, -1, vec, vec, NULL, allval, linsize, p->vertex->my_nk );
            #else
            batched_svd( NULL, -1, vec, NULL, vec, allval, linsize, p->vertex->my_nk );
            #endif
    }

    for (index_t q=0; q<p->vertex->my_nk; ++q) {
        if (q > 0) off[q] = off[q-1] + len[q-1];

        if (allval) {
            type[q] = p->c.algo;
            memcpy(_val, allval+q*linsize, sizeof(double)*linsize);
            memcpy(_vec, vec+q*POW2(linsize), sizeof(complex128_t)*POW2(linsize));
        } else {
            memcpy(_vec, channel+q*POW2(linsize), sizeof(complex128_t)*POW2(linsize));
            bool hermitian = check_hermiticity(_vec,linsize,p->c.algo_zero);
            if (p->c.algo == 'e' || (p->c.algo == 'a' && hermitian)) {
                single_eigen( NULL, _vec, _val, linsize);
                type[q] = 'e';
            } else if (p->c.algo == 's' || (p->c.algo == 'a' && !hermitian)) {
                single_svd( NULL, _vec, p->vertex->bubble_helper, _val, linsize);
                type[q] = 's';
            }
            // above the matrices are C order, we must transpose that…
            transpose_inplace( _vec, linsize );
        }

        len[q] = (p->c.tu_n_decomp_values != 0) ?
            single_eigen_sort_part( _vec, _val, linsize, 'M', p->c.tu_n_decomp_values ) :
            single_eigen_sort_N( _vec, _val, linsize, 'M', p->c.cutoff_rel ? -p->c.cutoff : p->c.cutoff );
        memcpy( val+off[q], _val, sizeof(double)*len[q] );
        memcpy( vec+off[q]*linsize, _vec, sizeof(complex128_t)*len[q]*linsize );
    }
    free(_val);
    free(allval);
}

void lingap_ff( tu_postproduction_t* p ) {
    // only calculates q = 0 linearized gap
    index_t linsize = p->vertex->n_orbff*POW2(p->vertex->n_spin);
    p->called_lingap = true;
    double *_val = malloc(linsize*sizeof(double)),
           *_xval = malloc(linsize*sizeof(double));
    p->tu_n_singular_valuesP = (p->vertex->P_flow) ? p->c.tu_n_singular_values : 0;
    p->tu_n_singular_valuesC = (p->vertex->C_flow) ? p->c.tu_n_singular_values : 0;
    p->tu_n_singular_valuesD = (p->vertex->D_flow) ? p->c.tu_n_singular_values : 0;
    if (diverge_mpi_comm_rank()==0) {
        p->Sgap_sc = malloc(p->tu_n_singular_valuesP*sizeof(double));
        p->Sgap_mag = malloc(p->tu_n_singular_valuesC*sizeof(double));
        p->Sgap_charge = malloc(p->tu_n_singular_valuesD*sizeof(double));

        p->Ugap_sc = malloc(p->tu_n_singular_valuesP*linsize*sizeof(complex128_t));
        p->Ugap_mag = malloc(p->tu_n_singular_valuesC*linsize*sizeof(complex128_t));
        p->Ugap_charge = malloc(p->tu_n_singular_valuesD*linsize*sizeof(complex128_t));

        p->Vgap_sc = malloc(p->tu_n_singular_valuesP*linsize*sizeof(complex128_t));
        p->Vgap_mag = malloc(p->tu_n_singular_valuesC*linsize*sizeof(complex128_t));
        p->Vgap_charge = malloc(p->tu_n_singular_valuesD*linsize*sizeof(complex128_t));
    }
    if (p->vertex->P_flow) {
        P_projection(p->vertex->projection_helper, p->vertex, p->proj);
        if (diverge_mpi_comm_rank()==0) {
            batched_gemm(p->vertex->projection_helper,
                    (p->c.lingap_atscale ? p->internal_vertex : p->vertex)->pp_bubble_int,
                    p->vertex->product_helper, POW2(p->vertex->n_spin)*p->vertex->n_orbff, 1.0, 0.0, 1);
            char solver_mode = p->c.tu_lingap_solver_mode[0];
            if (solver_mode == 's') {
                single_svd( NULL, p->vertex->product_helper, p->vertex->bubble_helper, _val, linsize );
            } else {
                single_svd( NULL, p->vertex->product_helper, p->vertex->bubble_helper, _val, -linsize );
                char sort_mode = solver_mode == 'E' ? 'M' : // absolute
                                 solver_mode == 'e' ? 'A' : // alternating
                                 solver_mode == 't' ? 'P' : // from largest
                                 solver_mode == 'b' ? 'N' : // from smallest
                                                      'm';  // inverse if none of the above
                memcpy(_xval, _val, sizeof(double)*linsize);

                transpose_inplace( p->vertex->product_helper, linsize );
                single_eigen_sort( p->vertex->product_helper, _val, linsize, sort_mode );
                transpose_inplace( p->vertex->product_helper, linsize );

                transpose_inplace( p->vertex->bubble_helper, linsize );
                single_eigen_sort( p->vertex->bubble_helper, _xval, linsize, sort_mode );
            }
            for (index_t i=0; i<p->tu_n_singular_valuesP; ++i) {
                p->Sgap_sc[i] = _val[i];
                for (index_t j=0; j<linsize; ++j) {
                    p->Ugap_sc[j+linsize*i] = p->vertex->product_helper[i+linsize*j];
                    p->Vgap_sc[j+linsize*i] = conj(p->vertex->bubble_helper[j+linsize*i]);
                }
            }
        }
    }
    if (p->vertex->C_flow) {
        C_projection(p->vertex->projection_helper, p->vertex, p->proj);
        if (diverge_mpi_comm_rank()==0) {
            batched_gemm(p->vertex->projection_helper,
                    (p->c.lingap_atscale ? p->internal_vertex : p->vertex)->ph_bubble_int,
                    p->vertex->product_helper, POW2(p->vertex->n_spin)*p->vertex->n_orbff, 1.0, 0.0, 1);
            char solver_mode = p->c.tu_lingap_solver_mode[1];
            if (solver_mode == 's') {
                single_svd( NULL, p->vertex->product_helper, p->vertex->bubble_helper, _val, linsize );
            } else {
                single_svd( NULL, p->vertex->product_helper, p->vertex->bubble_helper, _val, -linsize );
                char sort_mode = solver_mode == 'E' ? 'M' : // absolute
                                 solver_mode == 'e' ? 'A' : // alternating
                                 solver_mode == 't' ? 'P' : // from largest
                                 solver_mode == 'b' ? 'N' : // from smallest
                                                      'm';  // inverse if none of the above
                memcpy(_xval, _val, sizeof(double)*linsize);

                transpose_inplace( p->vertex->product_helper, linsize );
                single_eigen_sort( p->vertex->product_helper, _val, linsize, sort_mode );
                transpose_inplace( p->vertex->product_helper, linsize );

                transpose_inplace( p->vertex->bubble_helper, linsize );
                single_eigen_sort( p->vertex->bubble_helper, _xval, linsize, sort_mode );
            }
            for (index_t i=0; i<p->tu_n_singular_valuesC; ++i) {
                p->Sgap_mag[i] = _val[i];
                for (index_t j=0; j<linsize; ++j) {
                    p->Ugap_mag[j+linsize*i] = p->vertex->product_helper[i+linsize*j];
                    p->Vgap_mag[j+linsize*i] = conj(p->vertex->bubble_helper[j+linsize*i]);
                }
            }
        }
    }

    if (p->vertex->D_flow) {
        if (p->vertex->SU2 && p->vertex->C_flow) {
            // physical D channel!
            complex128_t* helper = malloc(sizeof(complex128_t)*POW2(linsize));
            memcpy(helper,p->vertex->projection_helper,sizeof(complex128_t)*POW2(linsize));
            D_projection(p->vertex->projection_helper, p->vertex, p->proj);
            for (index_t i=0; i<POW2(linsize); ++i)
                p->vertex->projection_helper[i] = 2.*p->vertex->projection_helper[i] - helper[i];
            free(helper);
        } else {
            D_projection(p->vertex->projection_helper, p->vertex, p->proj);
        }

        if (diverge_mpi_comm_rank()==0) {
            batched_gemm(p->vertex->projection_helper,
                    (p->c.lingap_atscale ? p->internal_vertex : p->vertex)->ph_bubble_int,
                    p->vertex->product_helper, POW2(p->vertex->n_spin)*p->vertex->n_orbff, 1.0, 0.0, 1);
            char solver_mode = p->c.tu_lingap_solver_mode[2];
            if (solver_mode == 's') {
                single_svd( NULL, p->vertex->product_helper, p->vertex->bubble_helper, _val, linsize );
            } else {
                single_svd( NULL, p->vertex->product_helper, p->vertex->bubble_helper, _val, -linsize );
                char sort_mode = solver_mode == 'E' ? 'M' : // absolute
                                 solver_mode == 'e' ? 'A' : // alternating
                                 solver_mode == 't' ? 'P' : // from largest
                                 solver_mode == 'b' ? 'N' : // from smallest
                                                      'm';  // inverse if none of the above
                memcpy(_xval, _val, sizeof(double)*linsize);

                transpose_inplace( p->vertex->product_helper, linsize );
                single_eigen_sort( p->vertex->product_helper, _val, linsize, sort_mode );
                transpose_inplace( p->vertex->product_helper, linsize );

                transpose_inplace( p->vertex->bubble_helper, linsize );
                single_eigen_sort( p->vertex->bubble_helper, _xval, linsize, sort_mode );
            }
            for (index_t i=0; i<p->tu_n_singular_valuesD; ++i) {
                p->Sgap_charge[i] = _val[i];
                for (index_t j=0; j<linsize; ++j) {
                    p->Ugap_charge[j+linsize*i] = p->vertex->product_helper[i+linsize*j];
                    p->Vgap_charge[j+linsize*i] = conj(p->vertex->bubble_helper[j+linsize*i]);
                }
            }
        }
    }
    free(_val);
    free(_xval);
}

static void _susc_o14k(const diverge_model_t* model, const complex128_t* inp, complex128_t* susc) {
    tu_data_t* tu_data = model->internals->tu_data;
    const index_t n_spin = model->n_spin;
    const index_t n_orb = model->n_orb;
    const index_t n_bonds = tu_data->n_bonds;
    const index_t realsp_size = tu_data->n_orbff;
    const index_t nk = tu_data->nk;

    const tu_formfactor_t* tu_ff = model->tu_ff;
    const double* kmesh = model->internals->kmesh;
    const index_t* bond_sizes = tu_data->bond_sizes;
    const index_t* bond_offsets = tu_data->bond_offsets;
    const index_t* ob_to_orbff = tu_data->ob_to_orbff;
    const index_t nff = tu_data->existing_ff_sz;

    double* R = malloc(nff*sizeof(double)*3);
    fill_Rvec_from_vec(model, tu_data->existing_ff, R);

    complex128_t* FS_ff = calloc(nff,sizeof(complex128_t));
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t ff=0; ff<nff; ++ff)
    for (index_t k=0; k<tu_data->nk; ++k)
        FS_ff[ff] += cexp(-I*claV3d_dot(claV3d_map(R+3*ff),claV3d_map(kmesh+3*k))) / ((double)tu_data->nk);
    free(R);
    const index_t nb = n_spin*n_orb;

    #pragma omp parallel for schedule(static) collapse(7) num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk; ++k)
    for (index_t s1=0; s1<n_spin; ++s1)
    for (index_t s2=0; s2<n_spin; ++s2)
    for (index_t s3=0; s3<n_spin; ++s3)
    for (index_t s4=0; s4<n_spin; ++s4)
    for (index_t o1=0; o1<n_orb; ++o1)
    for (index_t o3=0; o3<n_orb; ++o3)
    for (index_t b1=0; b1<bond_sizes[o1]; ++b1)
    for (index_t b3=0; b3<bond_sizes[o3]; ++b3) {
        const index_t o2 = tu_ff[bond_offsets[o1]+b1].oto;
        const index_t o4 = tu_ff[bond_offsets[o3]+b3].oto;
        const index_t ff = tu_ff[bond_offsets[o1]+b1].ffidx;
        const index_t ff2 = tu_ff[bond_offsets[o3]+b3].ffidx;
        index_t so1 = o1+n_orb*s1;
        index_t so2 = o2+n_orb*s2;
        index_t so3 = o3+n_orb*s3;
        index_t so4 = o4+n_orb*s4;

        index_t sp1 = s1+n_spin*s2;
        index_t sp3 = s3+n_spin*s4;
        const complex128_t ffv = FS_ff[ff];
        const complex128_t ffv2 = FS_ff[ff2];

        susc[IDX5(so4,so3,so2,so1,k, nb,nb,nb,nk)] += conj(ffv)*ffv2*
            inp[k + nk * (ob_to_orbff[b1+n_bonds*o1]+realsp_size
                            *(sp1+n_spin*n_spin*(ob_to_orbff[b3+n_bonds*o3]+realsp_size
                                *(sp3))))];
    }
    free(FS_ff);
}

static void calc_susc( tu_postproduction_t* p ) {
    index_t size = POW4(p->vertex->n_orb*p->vertex->n_spin)*p->vertex->nk;
    index_t size2 = POW4(p->vertex->n_spin)*POW2(p->vertex->n_orbff)*p->vertex->nk;
    p->called_suscep_full = p->c.susc_full;
    p->called_suscep_ff = p->c.susc_ff;
    p->tu_suscP = 0; p->tu_suscP_ff = 0;
    p->tu_suscC = 0; p->tu_suscC_ff = 0;
    p->tu_suscD = 0; p->tu_suscD_ff = 0;

    complex128_t* suscff = malloc(p->vertex->full_vert_size*sizeof(complex128_t));
    if (p->vertex->P_flow) {
        p->tu_suscP = (p->called_suscep_full) ? POW4(p->vertex->n_orb*p->vertex->n_spin)*p->vertex->nk : 0;
        p->tu_suscP_ff = (p->called_suscep_ff) ?
            POW2(p->vertex->n_orbff*POW2(p->vertex->n_spin))*p->vertex->nk : 0;
        p->suscPp_ff = malloc(size2*sizeof(complex128_t));
        P_projection(p->vertex->projection_helper, p->vertex, p->proj);
        tu_vertex_product(suscff, p->vertex->pp_bubble_int,p->vertex->projection_helper,
                p->vertex->product_helper, POW2(p->vertex->n_spin)*p->vertex->n_orbff,
                p->vertex->my_nk, 1.0, false, NULL);
        diverge_tu_vertex_reco(p->vertex, 'P', p->suscPp_ff, suscff);
        if (p->called_suscep_full) {
            p->suscPp = calloc(size,sizeof(complex128_t));
            _susc_o14k(p->model,p->suscPp_ff,p->suscPp);
        }
    }
    if (p->vertex->C_flow) {
        p->tu_suscC = (p->called_suscep_full) ? POW4(p->vertex->n_orb*p->vertex->n_spin)*p->vertex->nk : 0;
        p->tu_suscC_ff = (p->called_suscep_ff) ?
            POW2(p->vertex->n_orbff*POW2(p->vertex->n_spin))*p->vertex->nk : 0;
        p->suscMa_ff = malloc(size2*sizeof(complex128_t));
        C_projection(p->vertex->projection_helper, p->vertex, p->proj);
        tu_vertex_product(suscff, p->vertex->ph_bubble_int,p->vertex->projection_helper,
                p->vertex->product_helper, POW2(p->vertex->n_spin)*p->vertex->n_orbff,
                p->vertex->my_nk, 1.0, false, NULL);
        diverge_tu_vertex_reco(p->vertex, 'C', p->suscMa_ff, suscff);
        if (p->called_suscep_full) {
            p->suscMa = calloc(size,sizeof(complex128_t));
            _susc_o14k(p->model,p->suscMa_ff,p->suscMa);
        }
    }
    if (p->vertex->D_flow) {
        p->tu_suscD = (p->called_suscep_full) ? POW4(p->vertex->n_orb*p->vertex->n_spin)*p->vertex->nk : 0;
        p->tu_suscD_ff = (p->called_suscep_ff) ?
            POW2(p->vertex->n_orbff*POW2(p->vertex->n_spin))*p->vertex->nk : 0;
        p->suscCh_ff = malloc(size2*sizeof(complex128_t));
        if (p->vertex->SU2 && p->vertex->C_flow) {
            // physical D susc
            complex128_t* helper = calloc(p->vertex->full_vert_size,sizeof(complex128_t));
            D_projection(helper, p->vertex, p->proj);
            for (index_t i=0; i<p->vertex->full_vert_size; ++i)
                p->vertex->projection_helper[i] = 2.*helper[i] - p->vertex->projection_helper[i];
            free(helper);
        } else {
            D_projection(p->vertex->projection_helper, p->vertex, p->proj);
        }
        tu_vertex_product(suscff, p->vertex->ph_bubble_int,p->vertex->projection_helper,
                p->vertex->product_helper, POW2(p->vertex->n_spin)*p->vertex->n_orbff,
                p->vertex->my_nk, 1.0, false, NULL);
        diverge_tu_vertex_reco(p->vertex, 'D', p->suscCh_ff, suscff);
        if (p->called_suscep_full) {
            p->suscCh = calloc(size,sizeof(complex128_t));
            _susc_o14k(p->model,p->suscCh_ff,p->suscCh);
        }
    }
    if (!p->c.susc_ff) {
        free(p->suscCh_ff); p->suscCh_ff = NULL;
        free(p->suscMa_ff); p->suscMa_ff = NULL;
        free(p->suscPp_ff); p->suscPp_ff = NULL;
    }
    free(suscff);
}

static bool check_hermiticity( complex128_t* data, index_t linsize, double lim/*=1.e-11*/ ) {
    double err_max = 0.;
    #pragma omp parallel for reduction(max:err_max) collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<linsize; ++i) for (index_t j=0; j<linsize; ++j) {
        double err = cabs(data[IDX2(i,j,linsize)] - conj(data[IDX2(j,i,linsize)]));
        err_max = MAX(err_max, err);
    }
    return err_max < lim;
}

static inline claV3d_t claV3i_d( const claV3i_t in ) { return (claV3d_t){.v={in.v[0],in.v[1],in.v[2]}}; }
static inline void fill_Rvec_from_vec( const diverge_model_t* model, const claV3i_t* Rptr, double* Rvec ) {
    claM3d_t L = claM3d_map(model->lattice[0]);
    index_t nff = ((tu_data_t*)model->internals->tu_data)->existing_ff_sz;
    for (index_t i=0; i<nff; ++i)
        claV3d_rmap( claM3d_vecmat(claV3i_d(Rptr[i]),L), Rvec+3*i );
}

static void construct_multiindex_maps( tu_postproduction_t* p ) {
    mpi_vrb_printf("Generate one-to-many for all q for storing\n");
    const diverge_model_t* model = p->model;
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);
    tu_symmstruct_t* symm = tu_data->symm;
    if (!symm->use_symmetries)
        return;

    index_t n_spin = model->n_spin;
    index_t nkibz = tu_data->nkibz;
    index_t n_orbff = tu_data->n_orbff;

    VECTOR_RESIZE( p->mi_map_pref_in_in, nkibz*symm->mi_maps_to_sz );
    VECTOR_RESIZE( p->mi_map_pref_in_out, nkibz*symm->mi_maps_to_sz );
    VECTOR_RESIZE( p->mi_map_idx, nkibz*symm->mi_maps_to_sz );

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (index_t idx=0; idx<n_orbff*POW2(n_spin)*model->n_sym; ++idx)
    for (index_t q=0; q<nkibz; ++q) {
        claV3d_t kp = claV3d_map( model->internals->kmesh+3*symm->idx_ibz_in_fullmesh[q] );
        const complex128_t pref_sym = cexp(-I*claV3d_dot(kp, claV3d_map(symm->beyond_UC+3*idx)));

        for (index_t to=0; to<symm->mi_map_len[idx]; ++to) {
            p->mi_map_idx[q+nkibz*(to+symm->mi_map_off[idx])] =
                symm->mi_maps_to[to + symm->mi_map_off[idx]]+n_orbff*POW2(n_spin)*q;
            p->mi_map_pref_in_in[q+nkibz*(to+symm->mi_map_off[idx])] =
                symm->mi_prefac_in_in[to + symm->mi_map_off[idx]] * pref_sym;

            p->mi_map_pref_in_out[q+nkibz*(to+symm->mi_map_off[idx])] =
                symm->mi_prefac_in_out[to + symm->mi_map_off[idx]] * pref_sym;
        }
    }
}

static void to_file( tu_postproduction_t* p, const char* filename ) {
    index_t header[256] = {0};
    const char* version = tag_version();
    tu_data_t* tu_data = (tu_data_t*)p->model->internals->tu_data;
    tu_symmstruct_t* symm = tu_data->symm;
    int i = 0;
    index_t my_rank = diverge_mpi_comm_rank();
    const index_t my_nk = p->vertex->my_nk;
    index_t cop = 0;
    index_t coc = 0;
    index_t cod = 0;
    index_t nk_channel = 0;
    index_t my_cop = 0;
    index_t my_coc = 0;
    index_t my_cod = 0;
    if (!p->c.skip_channel_calc) {
        my_cop = p->lenP[my_nk-1]+p->offP[my_nk-1];
        my_coc = p->lenC[my_nk-1]+p->offC[my_nk-1];
        my_cod = p->lenD[my_nk-1]+p->offD[my_nk-1];
        nk_channel = tu_data->nkibz;
        for (index_t rank=0; rank<diverge_mpi_comm_size(); ++rank) {
            cop = my_cop;
            coc = my_coc;
            cod = my_cod;
            diverge_mpi_bcast_bytes((void*)(&cop), sizeof(index_t), rank);
            diverge_mpi_bcast_bytes((void*)(&coc), sizeof(index_t), rank);
            diverge_mpi_bcast_bytes((void*)(&cod), sizeof(index_t), rank);
            if (my_rank > rank) {
                for (index_t k=0; k<my_nk; ++k) {
                    p->offP[k] += cop;
                    p->offC[k] += coc;
                    p->offD[k] += cod;
                }
            }
        }
        // total numer of elements per channel
        cop = p->lenP[my_nk-1]+p->offP[my_nk-1];
        coc = p->lenC[my_nk-1]+p->offC[my_nk-1];
        cod = p->lenD[my_nk-1]+p->offD[my_nk-1];
        diverge_mpi_bcast_bytes((void*)(&cop), sizeof(index_t), diverge_mpi_comm_size()-1);
        diverge_mpi_bcast_bytes((void*)(&coc), sizeof(index_t), diverge_mpi_comm_size()-1);
        diverge_mpi_bcast_bytes((void*)(&cod), sizeof(index_t), diverge_mpi_comm_size()-1);
    }

    index_t linsize = p->vertex->n_orbff*POW2(p->vertex->n_spin);
    const index_t n_orb = p->model->n_orb;

    index_t size_self = 0;
    rs_hopping_t* hop_self = NULL;
    if (p->vertex->use_self) {
        if (p->c.save_self > 0) {
            size_self = POW2(n_orb*p->vertex->n_spin)*p->vertex->nk*p->vertex->nkf;
        } else if (p->c.save_self < 0) {
            double latnorm2 = POW2(p->model->lattice[0][0]) +
                POW2(p->model->lattice[0][1]) + POW2(p->model->lattice[0][2]);
            hop_self = diverge_model_ham2hop_fg( p->model, p->c.self_rs_mval,
                    sqrt(latnorm2) * p->c.self_rs_dist, &size_self, p->vertex->self_en );
        }
    }

    mpi_vrb_printf_all( "rank %d: try opening file %s\n", my_rank, filename );
    void* file = diverge_mpi_open_file(filename);
    header[i] = TU_MAGIC_NUMBER_POST_PROCESSING,++i;
    header[i] = p->vertex->n_orb, ++i;
    header[i] = p->vertex->n_spin, ++i;
    header[i] = p->vertex->nk, ++i;
    header[i] = p->vertex->nk*p->vertex->nkf, ++i;
    header[i] = tu_data->nkibz, ++i;
    header[i] = p->model->SU2, ++i;
    header[i] = p->vertex->n_orbff, ++i;
    header[i] = p->vertex->n_bonds, ++i;
    header[i] = p->model->n_sym, ++i;

    header[i] = sizeof(header), ++i;
    header[i] = p->vertex->n_orbff, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = p->vertex->n_orbff, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = 3*p->vertex->n_orbff, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = p->vertex->n_orb, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = p->vertex->n_orb, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = tu_data->nkibz, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    // store p->vertex diagonalizations first
    // P channel
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(char), ++i;
    header[i] = cop, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double), ++i;
    header[i] = cop*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    // C channel
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(char), ++i;
    header[i] = coc, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double), ++i;
    header[i] = coc*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    // D channel
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(char), ++i;
    header[i] = cod, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double), ++i;
    header[i] = cod*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    // lingap used?
    header[i] = p->tu_n_singular_valuesP, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double), ++i;
    header[i] = p->tu_n_singular_valuesP*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = p->tu_n_singular_valuesP*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;

    header[i] = p->tu_n_singular_valuesC, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double), ++i;
    header[i] = p->tu_n_singular_valuesC*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = p->tu_n_singular_valuesC*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;

    header[i] = p->tu_n_singular_valuesD, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double), ++i;
    header[i] = p->tu_n_singular_valuesD*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = p->tu_n_singular_valuesD*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;

    header[i] = p->tu_suscP,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = p->tu_suscC,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = p->tu_suscD,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;

    header[i] = p->tu_suscP_ff,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = p->tu_suscC_ff,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = p->tu_suscD_ff,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = size_self,++i;
    header[i] = header[i-2] + header[i-1]*(hop_self ? sizeof(rs_hopping_t) : sizeof(complex128_t)), ++i;
    if (p->c.save_full && !tu_data->tu_selfenergy_flow) {
        header[i] = p->vertex->P_flow ? POW2(linsize)*p->vertex->nk: 0,++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = p->vertex->C_flow ? POW2(linsize)*p->vertex->nk: 0,++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = p->vertex->D_flow ? POW2(linsize)*p->vertex->nk: 0,++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = p->vertex->P_flow ? POW2(linsize)*p->vertex->nk: 0,++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = (p->vertex->C_flow || p->vertex->D_flow) ? POW2(linsize)*p->vertex->nk:0, ++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    } else {
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    }

    if (p->c.save_symmetries && symm->use_symmetries) {
        construct_multiindex_maps(p);
        header[i] = symm->mi_cnt;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
        header[i] = symm->mi_cnt;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
        header[i] = p->mi_map_idx_sz;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
        header[i] = p->mi_map_pref_in_in_sz;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = p->mi_map_pref_in_out_sz;++i;
    } else {
        if (p->c.save_symmetries) {
            mpi_wrn_printf("no symmetries were used in the calculation! \n");
        }
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = 0;++i;
    }
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t); ++i;
    header[i] = strlen(version); ++i;

    header[i] = header[i-2] + header[i-1]*sizeof(char); ++i;
    if (p->c.save_symmetries && symm->use_symmetries)
        header[i] = symm->nk;
    i++;

    header[i] = header[i-2] + header[i-1]*sizeof(index_t); ++i;
    header[i] = p->vertex->n_orbff; ++i;

    const int chi0_reco = p->c.bare_susceptibilities < 0;
    const index_t chi0_nk = chi0_reco ? tu_data->nk : tu_data->nkibz;

    header[i] = header[i-2] + header[i-1]*sizeof(index_t); ++i;
    if (p->vertex->P_flow && p->c.bare_susceptibilities)
        header[i] = POW2(linsize) * chi0_nk;
    ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t); ++i;
    if ((p->vertex->C_flow || p->vertex->D_flow) && p->c.bare_susceptibilities)
        header[i] = POW2(linsize) * chi0_nk;
    ++i;

    // RPA
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t); ++i;
    if (p->rpa) header[i] = p->rpa->nev;
    ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double); ++i;
    if (p->rpa) header[i] = (index_t)p->rpa->nev * (index_t)p->rpa->legdim;
    ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t); ++i;
    if (p->rpa) header[i] = p->rpa->n_kidx;
    ++i;

    header[i] = hop_self ? 1 : 0; ++i;
    header[i] = header[i-3] + header[i-2]*sizeof(index_t); ++i;
    header[i] = sizeof(rs_hopping_dtype)/sizeof(rs_hopping_dtype[0]); ++i;

    assert(i < 125);
    // this flag is to indicate our header is actually 256 long. Only supported after v0.8
    header[125] = ~(header[125]);
    header[126] = TU_MAGIC_NUMBER_POST_PROCESSING;
    header[127] = header[i-1]*sizeof(index_t)+header[i-2];

    i = 128;
    header[i++] = sizeof(diverge_postprocess_conf_t);
    header[i++] = offsetof(diverge_postprocess_conf_t, patch_q_matrices);
    header[i++] = offsetof(diverge_postprocess_conf_t, patch_q_matrices_use_dV);
    header[i++] = offsetof(diverge_postprocess_conf_t, patch_q_matrices_nv);
    header[i++] = offsetof(diverge_postprocess_conf_t, patch_q_matrices_max_rel);
    header[i++] = offsetof(diverge_postprocess_conf_t, patch_q_matrices_eigen_which);
    header[i++] = offsetof(diverge_postprocess_conf_t, patch_V);
    header[i++] = offsetof(diverge_postprocess_conf_t, patch_dV);
    header[i++] = offsetof(diverge_postprocess_conf_t, patch_Lp);
    header[i++] = offsetof(diverge_postprocess_conf_t, patch_Lm);
    header[i++] = offsetof(diverge_postprocess_conf_t, grid_lingap_vertex_file_P);
    header[i++] = offsetof(diverge_postprocess_conf_t, grid_lingap_vertex_file_C);
    header[i++] = offsetof(diverge_postprocess_conf_t, grid_lingap_vertex_file_D);
    header[i++] = offsetof(diverge_postprocess_conf_t, grid_n_singular_values);
    header[i++] = offsetof(diverge_postprocess_conf_t, grid_use_loop);
    header[i++] = offsetof(diverge_postprocess_conf_t, grid_vertex_file);
    header[i++] = offsetof(diverge_postprocess_conf_t, grid_vertex_chan);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_which_solver_mode);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_skip_channel_calc);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_storing_threshold);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_storing_relative);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_n_singular_values);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_lingap);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_susceptibilities_full);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_susceptibilities_ff);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_selfenergy);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_channels);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_symmetry_maps);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_n_decomp_values);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_lingap_atscale);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_lingap_solver_mode);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_susceptibilities_bare);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_channel_calc_project);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_which_solver_mode_hermiticity_limit);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_rpa_chans);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_rpa_nev);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_rpa_ncv);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_rpa_maxiter);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_rpa_tol);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_rpa_sorting);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_selfenergy_rs_dist);
    header[i++] = offsetof(diverge_postprocess_conf_t, tu_selfenergy_rs_mval);
    assert(i <= 256);

    index_t my_nk_off = p->vertex->my_nk_off;
    index_t sindex = sizeof(index_t);
    index_t schar = sizeof(char);
    index_t sdou = sizeof(double);
    index_t scdou = sizeof(complex128_t);
    index_t new_off = 0;

    new_off = diverge_mpi_write_byte_to_file_master(file, header, new_off, sizeof(header));
    new_off = diverge_mpi_write_byte_to_file_master(file, tu_data->mi_to_ofrom, new_off, p->vertex->n_orbff*sindex);
    new_off = diverge_mpi_write_byte_to_file_master(file, tu_data->mi_to_oto, new_off, p->vertex->n_orbff*sindex);
    new_off = diverge_mpi_write_byte_to_file_master(file, tu_data->mi_to_R, new_off, 3*p->vertex->n_orbff*sindex);
    new_off = diverge_mpi_write_byte_to_file_master(file, tu_data->bond_sizes, new_off, p->vertex->n_orb*sindex);
    new_off = diverge_mpi_write_byte_to_file_master(file, tu_data->bond_offsets, new_off, p->vertex->n_orb*sindex);

    new_off = diverge_mpi_write_byte_to_file_master(file, symm->idx_ibz_in_fullmesh, new_off, tu_data->nkibz*sindex);
    // P p->vertex write
    if (!p->c.skip_channel_calc) {
        new_off = diverge_mpi_write_byte_to_file(file, p->lenP, new_off + my_nk_off*sindex, my_nk*sindex);
        new_off = diverge_mpi_write_byte_to_file(file, p->offP, new_off + my_nk_off*sindex, my_nk*sindex);
        new_off = diverge_mpi_write_byte_to_file(file, p->Ptype, new_off + my_nk_off*schar, my_nk*schar);
        new_off = diverge_mpi_write_byte_to_file(file, p->Pval, new_off + p->offP[0]*sdou, my_cop*sdou);
        new_off = diverge_mpi_write_byte_to_file(file, p->Pvec, new_off + p->offP[0]*linsize*scdou, my_cop*linsize*scdou);

        // C p->vertex write
        new_off = diverge_mpi_write_byte_to_file(file, p->lenC, new_off + my_nk_off*sindex, my_nk*sindex);
        new_off = diverge_mpi_write_byte_to_file(file, p->offC, new_off + my_nk_off*sindex, my_nk*sindex);
        new_off = diverge_mpi_write_byte_to_file(file, p->Ctype, new_off + my_nk_off*schar, my_nk*schar);
        new_off = diverge_mpi_write_byte_to_file(file, p->Cval, new_off + p->offC[0]*sdou, my_coc*sdou);
        new_off = diverge_mpi_write_byte_to_file(file, p->Cvec, new_off + p->offC[0]*linsize*scdou, my_coc*linsize*scdou);

        // D p->vertex write
        new_off = diverge_mpi_write_byte_to_file(file, p->lenD, new_off + my_nk_off*sindex, my_nk*sindex);
        new_off = diverge_mpi_write_byte_to_file(file, p->offD, new_off + my_nk_off*sindex, my_nk*sindex);
        new_off = diverge_mpi_write_byte_to_file(file, p->Dtype, new_off + my_nk_off*schar, my_nk*schar);
        new_off = diverge_mpi_write_byte_to_file(file, p->Dval, new_off + p->offD[0]*sdou, my_cod*sdou);
        new_off = diverge_mpi_write_byte_to_file(file, p->Dvec, new_off + p->offD[0]*linsize*scdou, my_cod*linsize*scdou);
    }
    if (p->called_lingap) {
        new_off = diverge_mpi_write_byte_to_file_master(file, p->Sgap_sc, new_off, sdou*p->tu_n_singular_valuesP);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->Ugap_sc, new_off, scdou*p->tu_n_singular_valuesP*linsize);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->Vgap_sc, new_off, scdou*p->tu_n_singular_valuesP*linsize);

        new_off = diverge_mpi_write_byte_to_file_master(file, p->Sgap_mag, new_off, sdou*p->tu_n_singular_valuesC);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->Ugap_mag, new_off, scdou*p->tu_n_singular_valuesC*linsize);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->Vgap_mag, new_off, scdou*p->tu_n_singular_valuesC*linsize);

        new_off = diverge_mpi_write_byte_to_file_master(file, p->Sgap_charge, new_off, sdou*p->tu_n_singular_valuesD);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->Ugap_charge, new_off, scdou*p->tu_n_singular_valuesD*linsize);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->Vgap_charge, new_off, scdou*p->tu_n_singular_valuesD*linsize);
    }

    if (p->called_suscep_full) {
        new_off = diverge_mpi_write_byte_to_file_master(file, p->suscPp,
                            new_off,scdou*p->tu_suscP);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->suscMa,
                            new_off,scdou*p->tu_suscC);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->suscCh,
                            new_off,scdou*p->tu_suscD);

    }
    if (p->called_suscep_ff) {
        new_off = diverge_mpi_write_byte_to_file_master(file, p->suscPp_ff,
                            new_off,scdou*p->tu_suscP_ff);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->suscMa_ff,
                            new_off,scdou*p->tu_suscC_ff);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->suscCh_ff,
                            new_off,scdou*p->tu_suscD_ff);
    }

    if (p->vertex->use_self && p->c.save_self > 0) {
        new_off = diverge_mpi_write_byte_to_file_master(file, p->vertex->self_en,
                        new_off,scdou*size_self);
    } else if (p->vertex->use_self && p->c.save_self < 0) {
        new_off = diverge_mpi_write_byte_to_file_master(file, hop_self,
                new_off, sizeof(rs_hopping_t)*size_self);
        free(hop_self);
    }
    if (p->c.save_full && !tu_data->tu_selfenergy_flow) {
        index_t vsize = POW2(linsize)*p->vertex->nk*sizeof(complex128_t);
        complex128_t* helper = (complex128_t*)malloc(vsize);
        if(p->vertex->P_flow) {
            diverge_tu_vertex_reco(p->vertex, 'P', helper, NULL);
            new_off = diverge_mpi_write_byte_to_file_master(file, helper, new_off,vsize);
        }
        if(p->vertex->C_flow) {
            diverge_tu_vertex_reco(p->vertex, 'C', helper, NULL);
            new_off = diverge_mpi_write_byte_to_file_master(file, helper, new_off,vsize);
        }
        if(p->vertex->D_flow) {
            diverge_tu_vertex_reco(p->vertex, 'D', helper, NULL);
            new_off = diverge_mpi_write_byte_to_file_master(file, helper, new_off,vsize);
        }
        if(p->vertex->P_flow) {
            diverge_tu_vertex_reco(p->vertex, 'P', helper, p->vertex->pp_bubble_int);
            new_off = diverge_mpi_write_byte_to_file_master(file, helper, new_off,vsize);
        }
        if(p->vertex->C_flow || p->vertex->D_flow) {
            diverge_tu_vertex_reco(p->vertex, 'C', helper, p->vertex->ph_bubble_int);
            new_off = diverge_mpi_write_byte_to_file_master(file, helper, new_off,vsize);
        }
        free(helper);
    }
    if (p->c.save_symmetries && symm->use_symmetries) {
        mpi_log_printf("saving symm!\n");
        new_off = diverge_mpi_write_byte_to_file_master(file, symm->mi_map_len, new_off,sindex*symm->mi_cnt);
        new_off = diverge_mpi_write_byte_to_file_master(file, symm->mi_map_off, new_off,sindex*symm->mi_cnt);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->mi_map_idx, new_off,sindex*p->mi_map_idx_sz);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->mi_map_pref_in_in, new_off,scdou*p->mi_map_pref_in_in_sz);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->mi_map_pref_in_out, new_off,scdou*p->mi_map_pref_in_out_sz);
    }

    new_off = diverge_mpi_write_byte_to_file_master(file, version, new_off, strlen(version));

    if (p->c.save_symmetries && symm->use_symmetries) {
        new_off = diverge_mpi_write_byte_to_file_master(file, symm->kmaps_to, new_off, symm->nk * sindex);
    }

    new_off = diverge_mpi_write_byte_to_file_master(file, tu_data->mi_to_tuffidx, new_off, sindex*p->vertex->n_orbff);

    index_t scdlin = POW2(linsize) * sizeof(complex128_t);
    if (p->c.bare_susceptibilities && p->vertex->P_flow) {
        if (!chi0_reco) {
            new_off = diverge_mpi_write_byte_to_file(file, p->vertex->pp_bubble_int,
                    new_off+my_nk_off*scdlin, my_nk*scdlin);
        } else {
            complex128_t* buf = calloc(chi0_nk * POW2(linsize), sizeof*buf);
            diverge_tu_vertex_reco(p->vertex, 'P', buf, p->vertex->pp_bubble_int);
            transpose_inplace_rect_any(buf, sizeof*buf, chi0_nk, POW2(linsize));
            new_off = diverge_mpi_write_byte_to_file_master(file, buf, new_off, chi0_nk*POW2(linsize)*sizeof*buf);
            free(buf);
        }
    }
    if (p->c.bare_susceptibilities && (p->vertex->C_flow || p->vertex->D_flow)) {
        if (!chi0_reco) {
            new_off = diverge_mpi_write_byte_to_file(file, p->vertex->ph_bubble_int,
                    new_off+my_nk_off*scdlin, my_nk*scdlin);
        } else {
            complex128_t* buf = calloc(chi0_nk * POW2(linsize), sizeof*buf);
            diverge_tu_vertex_reco(p->vertex, 'D', buf, p->vertex->ph_bubble_int);
            transpose_inplace_rect_any(buf, sizeof*buf, chi0_nk, POW2(linsize));
            new_off = diverge_mpi_write_byte_to_file_master(file, buf, new_off, chi0_nk*POW2(linsize)*sizeof*buf);
            free(buf);
        }
    }
    if (p->rpa) {
        new_off = diverge_mpi_write_byte_to_file_master(file, p->rpa->vals,
                new_off, sizeof*p->rpa->vals * p->rpa->nev);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->rpa->vecs,
                new_off, sizeof*p->rpa->vecs * p->rpa->nev * p->rpa->legdim);
        new_off = diverge_mpi_write_byte_to_file_master(file, p->rpa->kidx,
                new_off, sizeof*p->rpa->kidx * p->rpa->n_kidx);
    }

    new_off = diverge_mpi_write_byte_to_file_master(file, rs_hopping_dtype,
            new_off, sizeof(rs_hopping_dtype));

    if (new_off != header[127])
        mpi_wrn_printf( "offset of file (%li) dosn't match size (%li)\n", new_off, header[127] );

    diverge_mpi_close_file(file);
}
