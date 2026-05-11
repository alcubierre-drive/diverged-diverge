/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#define LINALG_NO_LAPACKE
#include "../misc/linalg.h"
LINALG_IMPLEMENT

#include "selfenergy.h"
#include "vertex.h"
#include "phonons.h"
#include "vertex_details.h"

static void apply_circ_cutoff( complex128_t* rs_ary/*(ostride,nk[0],nk[1],nk[2],istride)*/,
        const index_t* nk, index_t istride, index_t ostride, tu_selfenergy_t* s );

struct tu_selfenergy_t {
    fftw_plan ifft_self_oop;
    fftw_plan fft_self_oop;
    fftw_plan ifft_self_ip;
    fftw_plan fft_self_ip;

    fftw_plan fft_self_oop_gfmesh;

    fftw_plan ifft_fullself_ip;
    fftw_plan fft_fullself_ip;

    fftw_plan ifft_vert_oop;
    fftw_plan fft_vert_oop;
    fftw_plan ifft_vert_ip;
    fftw_plan fft_vert_ip;

    //inputs
    diverge_model_t* model;
    const tu_data_t* tu_data;
    index_t n_orb;
    index_t n_spin;
    index_t ns;
    index_t n_orbff;
    index_t n_bonds;
    index_t nk;
    index_t nkf;
    index_t nktot;
    index_t my_nk;
    index_t my_nk_off;
    index_t nff;
    bool SU2;

    complex128_t* ffkmesh;

    const index_t* ob_to_orbff;
    const claV3i_t* existing_ff;
    index_t existing_ff_sz;

    const tu_formfactor_t* tu_ff;
    const index_t* bond_sizes;
    const index_t* bond_offsets;
    const tu_symmstruct_t* symm;
    bool P_flow;
    bool C_flow;
    bool D_flow;

    complex128_t* reco_helper;
    complex128_t* interpol_helper;

    complex128_t* ff_helper;
    complex128_t* greensfkt_helper;

    claM3d_t circ_cutoff_lat;
    double circ_cutoff;
    bool circ_cutoff_use;
};

static void generate_fftw_plans( tu_selfenergy_t* s );

// different flows to dispatch to
static void selfenergy_flow_rs( tu_selfenergy_t* s, complex128_t* self_en,
        const complex128_t* P_in, complex128_t* C_in, complex128_t* D_in,
        const complex128_t* GF, double prefac_sub, double lambda );
static void selfenergy_flow_rs_SU2( tu_selfenergy_t* s, complex128_t* self_en,
        const complex128_t* P_in, complex128_t* C_in, complex128_t* D_in,
        const complex128_t* GF, double prefac_sub, double lambda );
static void selfenergy_flow_mom(tu_selfenergy_t* s, complex128_t* self_en,
        const complex128_t* P_in, complex128_t* C_in, complex128_t* D_in,
        const complex128_t* GF, double prefac_sub, double lambda );
static void selfenergy_flow_mom_SU2(tu_selfenergy_t* s, complex128_t* self_en,
        const complex128_t* P_in, complex128_t* C_in, complex128_t* D_in,
        const complex128_t* GF, double prefac_sub, double lambda );

// calculates buf[i] = a*Reco(inp)[i] + b*buf[i]
static inline void reco( const char channel, tu_selfenergy_t* s, complex128_t* buf,
        const complex128_t* inp, double a, double b/*=1*/ );

// some helpers
// transposes (b,o,k) -> (k,o,b) if nk > 0 && nb > 0, otherwise (k,o,b) -> (b,o,k)
static inline void transpose( complex128_t* buf, complex128_t* aux, index_t nk, index_t nb );
#define FFTW_EXECUTE_DFT( plan, in, out ) fftw_execute_dft( (plan), (fftw_complex*)(in), (fftw_complex*)(out) )
static inline index_t rpb1mb3(const index_t* nk, index_t x, index_t y, index_t z,
                              const index_t* vb1, const index_t* vb3);
static inline index_t rpb1pb3(const index_t* nk, index_t x, index_t y, index_t z,
                              const index_t* vb1, const index_t* vb3);
static inline index_t rpb1(const index_t* nk, index_t x, index_t y, index_t z, const index_t* vb1);

tu_selfenergy_t* tu_selfenergy_init( tu_vertex_t* V ) {
    tu_selfenergy_t* s = calloc(1, sizeof*s);

    s->model = V->model;
    s->tu_data = V->tu_data;
    s->n_orb = V->n_orb;
    s->n_spin = V->n_spin;
    s->ns = POW2(s->n_spin);
    s->n_orbff = V->n_orbff;
    s->n_bonds = s->tu_data->n_bonds;
    s->nk = V->nk;
    s->nkf = V->nkf;
    s->nktot = s->nk * s->nkf;
    s->my_nk = V->my_nk;
    s->my_nk_off = V->my_nk_off;
    s->nff = s->tu_data->existing_ff_sz;
    s->SU2 = (s->model->SU2 > 0);
    s->ob_to_orbff = s->tu_data->ob_to_orbff;
    s->existing_ff = s->tu_data->existing_ff;
    s->existing_ff_sz = s->tu_data->existing_ff_sz;
    s->tu_ff = s->model->tu_ff;
    s->bond_sizes = s->tu_data->bond_sizes;
    s->bond_offsets = s->tu_data->bond_offsets;
    s->symm = V->symm;
    s->P_flow = V->P_flow;
    s->C_flow = V->C_flow;
    s->D_flow = V->D_flow;

    s->reco_helper = V->product_helper;
    s->interpol_helper = V->bubble_helper;
    s->ff_helper = calloc(POW2(s->n_orb*s->n_spin)*s->nktot,sizeof(complex128_t));
    if (s->nk!=1) {
        s->greensfkt_helper = calloc(POW2(s->n_spin*s->n_orb)*s->nktot,sizeof(complex128_t));
        generate_fftw_plans(s);
        s->ffkmesh = calloc(s->nk*s->nff,sizeof(complex128_t));

        #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(diverge_omp_num_threads())
        for (index_t rv=0; rv<s->nff; ++rv)
        for (index_t kx=0; kx<s->model->nk[0]; ++kx)
        for (index_t ky=0; ky<s->model->nk[1]; ++ky)
        for (index_t kz=0; kz<s->model->nk[2]; ++kz) {
            index_t k = IDX3(kx, ky, kz, s->model->nk[1], s->model->nk[2]);
            s->ffkmesh[k+s->nk*rv] = cexp(2.*M_PI*I*(
                        s->existing_ff[rv].v[0]*kx/(double)s->model->nk[0] +
                        s->existing_ff[rv].v[1]*ky/(double)s->model->nk[1] +
                        s->existing_ff[rv].v[2]*kz/(double)s->model->nk[2]));
        }

        if (s->tu_data->selfenergy_circular_cutoff) {
            int dim = s->model->internals->dim;
            if (dim > 1) {
                index_t nr[3] = {s->model->nk[0]/2-1, s->model->nk[1]/2-1, s->model->nk[2]/2-1};
                for (int d=dim; d<3; ++d) nr[d] = 0;
                for (int d=0; d<3; ++d) nr[d] = MAX(0,nr[d]);

                const claM3d_t lat = claM3d_map( s->model->lattice[0] );
                double minnorm = claV3d_norm(claV3d_scale(claV3d_map(s->model->lattice[0]), nr[0]));
                for (index_t rx=-nr[0]; rx<=nr[0]; ++rx)
                for (index_t ry=-nr[0]; ry<=nr[0]; ++ry)
                for (index_t rz=-nr[0]; rz<=nr[0]; ++rz) {
                    bool is_boundary = (labs(rx) == nr[0]);
                    if (dim > 1) is_boundary = is_boundary || (labs(ry) == nr[1]);
                    if (dim > 2) is_boundary = is_boundary || (labs(rz) == nr[2]);
                    if (is_boundary) {
                        double nrm = claV3d_norm(claM3d_vecmat((claV3d_t){{rx,ry,rz}}, lat));
                        minnorm = MIN(nrm, minnorm);
                    }
                }
                s->circ_cutoff = minnorm;
                s->circ_cutoff_use = true;
                s->circ_cutoff_lat = lat;
                mpi_log_printf( "circular RS cutoff (%.2f) for self-energy\n", s->circ_cutoff );
            }
        }
    }
    return s;
}

void tu_selfenergy_free( tu_selfenergy_t* s ) {
    free(s->ff_helper);
    if (s->nk!=1) {
        free(s->ffkmesh);
        free(s->greensfkt_helper);
        fftw_destroy_plan(s->ifft_self_oop);
        fftw_destroy_plan(s->fft_self_oop);
        fftw_destroy_plan(s->ifft_self_ip);
        fftw_destroy_plan(s->fft_self_ip);
        fftw_destroy_plan(s->fft_fullself_ip);
        fftw_destroy_plan(s->ifft_fullself_ip);
        fftw_destroy_plan(s->fft_self_oop_gfmesh);
        fftw_destroy_plan(s->ifft_vert_oop);
        fftw_destroy_plan(s->fft_vert_oop);
        fftw_destroy_plan(s->ifft_vert_ip);
        fftw_destroy_plan(s->fft_vert_ip);
    }
    free(s);
}

// only a dispatcher
void tu_selfenergy_flow( tu_selfenergy_t* s, complex128_t* self_en,
        const complex128_t* P_in, complex128_t* C_in,
        complex128_t* D_in, const complex128_t* GF,
        const double prefac_sub, const double lambda ) {
    if (s->nk == 1 && !s->SU2) {
        selfenergy_flow_rs( s, self_en, P_in, C_in, D_in, GF, prefac_sub, lambda );
    } else if (s->nk == 1 && s->SU2) {
        selfenergy_flow_rs_SU2( s, self_en, P_in, C_in, D_in, GF, prefac_sub, lambda  );
    } else if (s->nk != 1 && !s->SU2) {
        selfenergy_flow_mom( s, self_en, P_in, C_in, D_in, GF, prefac_sub, lambda );
    } else {
        selfenergy_flow_mom_SU2( s, self_en, P_in, C_in, D_in, GF, prefac_sub, lambda );
    }
}

// calculates buf[i] = a*Reco(inp)[i] + b*buf[i]
static inline void reco( const char channel, tu_selfenergy_t* s, complex128_t* buf,
        const complex128_t* inp, double a, double b/*=1*/ ) {
    if (fabs(b) < 1e-10) ZERO_OUT( buf, POW2(s->n_orbff*s->n_spin*s->n_spin)*s->nk );
    index_t rest = POW2(s->n_orbff*POW2(s->n_spin));
    if (s->symm->use_symmetries) {
        const index_t my_nk_in_bz = s->symm->my_nk_in_bz;
        const index_t my_nk_in_bz_off = s->symm->my_nk_in_bz_off;

        if (fabs(b-1.) > 1e-10) {
            #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
            for (index_t r=0; r<rest*s->nk; ++r)
                buf[r] *= b;
        }

        const index_t nsq = POW2(s->n_spin);
        const index_t legdim = nsq*s->n_orbff;
        #pragma omp parallel for schedule(static) collapse(5) num_threads(diverge_omp_num_threads())
        for (index_t s4=0; s4<s->n_spin; ++s4)
        for (index_t s3=0; s3<s->n_spin; ++s3)
        for (index_t mo3=0; mo3<s->n_orbff; ++mo3)
        for (index_t s2=0; s2<s->n_spin; ++s2)
        for (index_t s1=0; s1<s->n_spin; ++s1)
        for (index_t mo1=0; mo1<s->n_orbff; ++mo1)
        for (index_t q=0; q<my_nk_in_bz; ++q) {
            const index_t q_in_full = s->symm->from_ibz_2_bz[q + my_nk_in_bz_off];
            const index_t qFROM_IBZ = s->symm->kmaps_to[q_in_full] - s->tu_data->my_nk_off;
            const index_t idx_TO = q_in_full + s->nk*(mo1 + s->n_orbff*((s1+s->n_spin*s2)+nsq*
                    (mo3 + s->n_orbff*(s3+s->n_spin*s4))));

            if (channel == 'P') {
                index_t lidx = IDX4(s2,s1,mo1,q,s->n_spin,s->n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s4,s3,mo3,q,s->n_spin,s->n_orbff,my_nk_in_bz);

                for (index_t l=0; l<s->symm->o2m_map_len[lidx]; ++l)
                for (index_t r=0; r<s->symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM = s->symm->idx_map_symm[l+s->symm->o2m_map_off[lidx]] + legdim * (
                            s->symm->idx_map_symm[r+s->symm->o2m_map_off[ridx]]
                            + legdim * (qFROM_IBZ));

                    buf[idx_TO] += s->symm->prefactor_symmP[l+s->symm->o2m_map_off[lidx]]*
                              conj(s->symm->prefactor_symmP[r+s->symm->o2m_map_off[ridx]])* a*inp[idx_FROM];
                }
            }
            if (channel == 'C') {
                index_t lidx = IDX4(s4,s1,mo1,q,s->n_spin,s->n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s2,s3,mo3,q,s->n_spin,s->n_orbff,my_nk_in_bz);

                for (index_t l=0; l<s->symm->o2m_map_len[lidx]; ++l)
                for (index_t r=0; r<s->symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM = s->symm->idx_map_symm[l+s->symm->o2m_map_off[lidx]] + legdim * (
                            s->symm->idx_map_symm[r+s->symm->o2m_map_off[ridx]]
                            + legdim * (qFROM_IBZ));

                    buf[idx_TO] += s->symm->prefactor_symmCD[l+s->symm->o2m_map_off[lidx]]*
                              conj(s->symm->prefactor_symmCD[r+s->symm->o2m_map_off[ridx]])* a*inp[idx_FROM];
                }
            }
            if (channel == 'D') {
                index_t lidx = IDX4(s3,s1,mo1,q,s->n_spin,s->n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s2,s4,mo3,q,s->n_spin,s->n_orbff,my_nk_in_bz);

                for (index_t l=0; l<s->symm->o2m_map_len[lidx]; ++l)
                for (index_t r=0; r<s->symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM = s->symm->idx_map_symm[l+s->symm->o2m_map_off[lidx]] + legdim * (
                            s->symm->idx_map_symm[r+s->symm->o2m_map_off[ridx]]
                            + legdim * (qFROM_IBZ));

                    buf[idx_TO] += s->symm->prefactor_symmCD[l+s->symm->o2m_map_off[lidx]]*
                              conj(s->symm->prefactor_symmCD[r+s->symm->o2m_map_off[ridx]])* a*inp[idx_FROM];
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(static) collapse(7) num_threads(diverge_omp_num_threads())
        for (index_t s4=0; s4<s->n_spin; ++s4)
        for (index_t s3=0; s3<s->n_spin; ++s3)
        for (index_t mo3=0; mo3<s->n_orbff; ++mo3)
        for (index_t s2=0; s2<s->n_spin; ++s2)
        for (index_t s1=0; s1<s->n_spin; ++s1)
        for (index_t mo1=0; mo1<s->n_orbff; ++mo1)
        for (index_t q=0; q<s->my_nk; ++q) {
            const index_t idx_TO = q + s->my_nk_off+ s->nk*(mo1+s->n_orbff*(s1+s->n_spin*
                                (s2+s->n_spin*(mo3+s->n_orbff*(s3+s->n_spin*s4)))));
            if (channel == 'P') {
                buf[idx_TO] = b*buf[idx_TO]+a*inp[mo1+s->n_orbff*(s1+s->n_spin*
                                    (s2+s->n_spin*(mo3+s->n_orbff*(s3+s->n_spin*
                                                        (s4+s->n_spin*q)))))];
            }
            if (channel == 'C') {
                buf[idx_TO] = b*buf[idx_TO]+a*inp[mo1+s->n_orbff*(s1+s->n_spin*
                                    (s4+s->n_spin*(mo3+s->n_orbff*(s3+s->n_spin*
                                                        (s2+s->n_spin*q)))))];
            }
            if (channel == 'D') {
                buf[idx_TO] = b*buf[idx_TO]+a*inp[mo1+s->n_orbff*(s1+s->n_spin*
                                    (s3+s->n_spin*(mo3+s->n_orbff*(s4+s->n_spin*
                                                        (s2+s->n_spin*q)))))];
            }
        }
    }
}

static inline void transpose( complex128_t* buf, complex128_t* aux, index_t nk, index_t nb ) {
    if (nk > 0 && nb > 0) {
        for (index_t k=0; k<nk; ++k)
        for (index_t o=0; o<nb; ++o)
        for (index_t b=0; b<nb; ++b) aux[IDX3(k,o,b, nb,nb)] = buf[IDX3(b,o,k, nb,nk)];
    } else {
        nk = labs(nk), nb = labs(nb);
        for (index_t k=0; k<nk; ++k)
        for (index_t o=0; o<nb; ++o)
        for (index_t b=0; b<nb; ++b) aux[IDX3(b,o,k, nb,nk)] = buf[IDX3(k,o,b, nb,nb)];
    }
    memcpy( buf, aux, POW2(nb)*nk * sizeof*buf );
}

static inline index_t rpb1mb3(const index_t* nk, index_t x, index_t y, index_t z,
                              const index_t* vb1, const index_t* vb3) {
    index_t mx = (x+vb1[0]-vb3[0]+2*nk[0])%nk[0];
    index_t my = (y+vb1[1]-vb3[1]+2*nk[1])%nk[1];
    index_t mz = (z+vb1[2]-vb3[2]+2*nk[2])%nk[2];
    return mz + nk[2] *(my + nk[1] * mx);
}

static inline index_t rpb1pb3(const index_t* nk, index_t x, index_t y, index_t z,
                              const index_t* vb1, const index_t* vb3) {
    index_t mx = (x+vb1[0]+vb3[0])%nk[0];
    index_t my = (y+vb1[1]+vb3[1])%nk[1];
    index_t mz = (z+vb1[2]+vb3[2])%nk[2];
    return mz + nk[2] *(my + nk[1] * mx);
}

static inline index_t rpb1(const index_t* nk, index_t x, index_t y, index_t z, const index_t* vb1) {
    index_t mx = (x+vb1[0])%nk[0];
    index_t my = (y+vb1[1])%nk[1];
    index_t mz = (z+vb1[2])%nk[2];
    return mz + nk[2] *(my + nk[1] * mx);
}

static void apply_circ_cutoff( complex128_t* rs_ary, const index_t* nk,
        index_t istride, index_t ostride, tu_selfenergy_t* s ) {
    if (!s->circ_cutoff_use) return;

    const claM3d_t lat = s->circ_cutoff_lat;
    const double cut = s->circ_cutoff;

    const float hx = (nk[0]>1) ? (float)nk[0]/2. : 2;
    const float hy = (nk[1]>1) ? (float)nk[1]/2. : 2;
    const float hz = (nk[2]>1) ? (float)nk[2]/2. : 2;

    #pragma omp parallel for num_threads(diverge_omp_num_threads()) collapse(5)
    for (index_t os=0; os<ostride; ++os)
    for (index_t x=0; x<nk[0]; ++x)
    for (index_t y=0; y<nk[1]; ++y)
    for (index_t z=0; z<nk[2]; ++z)
    for (index_t is=0; is<istride; ++is) {
        const claV3d_t ivec = {{x<=hx?x:x-nk[0],y<=hy?y:y-nk[1],z<=hz?z:z-nk[2]}};
        double dist = claV3d_norm(claM3d_vecmat(ivec, lat));
        if (dist >= (cut + DIVERGE_EPS_MESH))
            rs_ary[ IDX5( os, x,y,z, is, nk[0],nk[1],nk[2], istride ) ] = 0.0;
    }
}

#define IS_ZERO(x) (fabs(x) < 1e-7)

void tu_selfenergy_fft_interpolate( tu_selfenergy_t* s, complex128_t* self_en ) {
    const index_t nk[] = { s->model->nk[0], s->model->nk[1], s->model->nk[2] };
    const index_t nkff[] = { nk[0]*s->model->nkf[0], nk[1]*s->model->nkf[1], nk[2]*s->model->nkf[2] };
    // pad with zeros
    ZERO_OUT( s->ff_helper, s->nktot*POW2(s->n_orb*s->n_spin) );
    const double pref = 1./(double)s->nk;

    float hx = (nk[0]>1) ? (float)nk[0]/2. : 2;
    float hy = (nk[1]>1) ? (float)nk[1]/2. : 2;
    float hz = (nk[2]>1) ? (float)nk[2]/2. : 2;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t r=0; r<POW2(s->n_orb*s->n_spin); ++r) {
        for (index_t x=0; x<nk[0]; ++x)
        for (index_t y=0; y<nk[1]; ++y)
        for (index_t z=0; z<nk[2]; ++z) {

            const index_t xto = (x<=hx) ? x : nkff[0]-(nk[0]-x);
            const index_t yto = (y<=hy) ? y : nkff[1]-(nk[1]-y);
            const index_t zto = (z<=hz) ? z : nkff[2]-(nk[2]-z);

            complex128_t helper = self_en[IDX4(r, x,y,z, nk[0],nk[1],nk[2])]*pref;
            // copy edges and corners
            if (IS_ZERO(x-hx) && IS_ZERO(y-hy) && IS_ZERO(z-hz)) {
                helper *= 0.125;
                const index_t xto2 = nkff[0]-hx, yto2 = nkff[1]-hy, zto2 = nkff[2]-hz;
                s->ff_helper[IDX4(r,xto ,yto ,zto2, nkff[0],nkff[1],nkff[2])] += helper;
                s->ff_helper[IDX4(r,xto ,yto2,zto , nkff[0],nkff[1],nkff[2])] += helper;
                s->ff_helper[IDX4(r,xto ,yto2,zto2, nkff[0],nkff[1],nkff[2])] += helper;
                s->ff_helper[IDX4(r,xto2,yto ,zto , nkff[0],nkff[1],nkff[2])] += helper;
                s->ff_helper[IDX4(r,xto2,yto ,zto2, nkff[0],nkff[1],nkff[2])] += helper;
                s->ff_helper[IDX4(r,xto2,yto2,zto , nkff[0],nkff[1],nkff[2])] += helper;
                s->ff_helper[IDX4(r,xto2,yto2,zto2, nkff[0],nkff[1],nkff[2])] += helper;
            } else if (IS_ZERO(x-hx) && IS_ZERO(y-hy)) {
                helper *= 0.25;
                const index_t xto2 = nkff[0]-hx, yto2 = nkff[1]-hy;
                s->ff_helper[IDX4(r,xto ,yto2,zto , nkff[0],nkff[1],nkff[2])] += helper;
                s->ff_helper[IDX4(r,xto2,yto ,zto , nkff[0],nkff[1],nkff[2])] += helper;
                s->ff_helper[IDX4(r,xto2,yto2,zto , nkff[0],nkff[1],nkff[2])] += helper;
            } else if (IS_ZERO(x-hx) && IS_ZERO(z-hz)) {
                helper *= 0.25;
                const index_t xto2 = nkff[0]-hx, zto2 = nkff[2]-hz;
                s->ff_helper[IDX4(r,xto ,yto ,zto2, nkff[0],nkff[1],nkff[2])] += helper;
                s->ff_helper[IDX4(r,xto2,yto ,zto , nkff[0],nkff[1],nkff[2])] += helper;
                s->ff_helper[IDX4(r,xto2,yto ,zto2, nkff[0],nkff[1],nkff[2])] += helper;
            } else if (IS_ZERO(y-hy) && IS_ZERO(z-hz)) {
                helper *= 0.25;
                const index_t yto2 = nkff[1]-hy, zto2 = nkff[2]-hz;
                s->ff_helper[IDX4(r,xto ,yto ,zto2, nkff[0],nkff[1],nkff[2])] += helper;
                s->ff_helper[IDX4(r,xto ,yto2,zto , nkff[0],nkff[1],nkff[2])] += helper;
                s->ff_helper[IDX4(r,xto ,yto2,zto2, nkff[0],nkff[1],nkff[2])] += helper;
            } else if (IS_ZERO(x-hx)) {
                helper *= 0.5;
                const index_t xto2 = nkff[0]-hx;
                s->ff_helper[IDX4(r,xto2,yto ,zto , nkff[0],nkff[1],nkff[2])] += helper;
            } else if (IS_ZERO(z-hz)) {
                helper *= 0.5;
                const index_t zto2 = nkff[2]-hz;
                s->ff_helper[IDX4(r,xto ,yto ,zto2, nkff[0],nkff[1],nkff[2])] += helper;
            } else if (IS_ZERO(y-hy)) {
                helper *= 0.5;
                const index_t yto2 = nkff[1]-hy;
                s->ff_helper[IDX4(r,xto ,yto2,zto , nkff[0],nkff[1],nkff[2])] += helper;
            }
            s->ff_helper[IDX4(r,xto ,yto ,zto , nkff[0],nkff[1],nkff[2])] += helper;
        }
    }
    FFTW_EXECUTE_DFT( s->fft_self_oop_gfmesh, s->ff_helper, self_en);
}

static void selfenergy_flow_rs( tu_selfenergy_t* s, complex128_t* self_en,
        const complex128_t* P_in, complex128_t* C_in, complex128_t* D_in,
        const complex128_t* GF, double pref_subst, double lambda ) {
    tu_data_phonons_t* phonons = s->tu_data->phonons_reference;
    // TODO: The non-SU2 one might be wrong (it kinda breaks crossing, but on the other hand not really...
    if (phonons && D_in && s->D_flow) tu_data_phonons_gen_vert( phonons, 0.0, D_in, 1.0 );
    if (phonons && C_in && s->C_flow) tu_data_phonons_gen_vert( phonons, lambda, C_in, -1.0 );
    // selfenergy flow equation
    const index_t blocksize_self = POW2(s->n_orb*s->n_spin);
    const double prefac = pref_subst/(2.0*M_PI);
    // set all elements to zero
    ZERO_OUT( self_en, blocksize_self );

    // D-channel -> real part as D is freq. independent =>
    // same for all frequencies
    // again catch the zero frequency independently
    if (s->D_flow) {
        #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
        for (index_t o1=0; o1<s->n_orb; ++o1)
        for (index_t s1=0; s1<s->n_spin; ++s1)
        for (index_t s3=0; s3<s->n_spin; ++s3)
        for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
            complex128_t tmp = 0.0;
            for (index_t s2=0; s2<s->n_spin; ++s2)
            for (index_t s4=0; s4<s->n_spin; ++s4)
            for (index_t o4=0; o4<s->n_orb; ++o4)
            for (index_t b4=0; b4<s->bond_sizes[o4]; ++b4) {
                tmp += (conj(GF[s->tu_ff[s->bond_offsets[o4]+b4].oto+s->n_orb*
                                (s2+s->n_spin*(o4 + s->n_orb*s4))])
                           + GF[o4+s->n_orb*(s4+s->n_spin*
                                (s->tu_ff[s->bond_offsets[o4]+b4].oto + s->n_orb*s2))]) *
                    D_in[s->ob_to_orbff[b1+s->n_bonds*o1]+s->n_orbff*(s1+s->n_spin
                            *(s3+s->n_spin*(s->ob_to_orbff[b4+s->n_bonds*o4]+s->n_orbff
                                    *(s4+s->n_spin*s2))))];
            }
            self_en[o1+s->n_orb*(s1+s->n_spin*
                    (s->tu_ff[s->bond_offsets[o1]+b1].oto+s->n_orb*(s3)))] -= prefac*tmp;
        }
    }
    if (s->P_flow) {
        #pragma omp parallel for collapse(4) num_threads(diverge_omp_num_threads())
        for (index_t s3=0; s3<s->n_spin; ++s3)
        for (index_t o3=0; o3<s->n_orb; ++o3)
        for (index_t s1=0; s1<s->n_spin; ++s1)
        for (index_t o1=0; o1<s->n_orb; ++o1) {
            complex128_t tmp = 0.0;
            for (index_t s4=0; s4<s->n_spin; ++s4)
            for (index_t s2=0; s2<s->n_spin; ++s2)
            for (index_t b3=0; b3<s->bond_sizes[o3]; ++b3)
            for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
                // +
                const index_t idxGF_P_p = s->tu_ff[s->bond_offsets[o3]+b3].oto + s->n_orb
                            *(s4+s->n_spin*(s->tu_ff[s->bond_offsets[o1]+b1].oto + s->n_orb*s2));
                const index_t idx_P_p = s->ob_to_orbff[b1+s->n_bonds*o1]+s->n_orbff
                            *(s1+s->n_spin*(s2+s->n_spin*(s->ob_to_orbff[b3+s->n_bonds*o3]+
                                            s->n_orbff*(s3+s->n_spin*(s4)))));
                tmp += GF[idxGF_P_p]*P_in[idx_P_p];
                // -
                const index_t idxGF_P_m = s->tu_ff[s->bond_offsets[o1]+b1].oto+s->n_orb
                            *(s2+s->n_spin*(s->tu_ff[s->bond_offsets[o3]+b3].oto + s->n_orb*s4));
                const index_t idx_P_m = s->ob_to_orbff[b3+s->n_bonds*o3]+s->n_orbff*(s3+s->n_spin
                            *(s4+s->n_spin*(s->ob_to_orbff[b1+s->n_bonds*o1]+s->n_orbff
                                *(s1+s->n_spin*(s2)))));

                tmp += conj(GF[idxGF_P_m]*P_in[idx_P_m]);
            }
            self_en[o1+s->n_orb*(s1+s->n_spin*(o3+s->n_orb*(s3)))] -= prefac*tmp;
        }
    }
    if (s->C_flow && C_in) {
        #pragma omp parallel for collapse(4) num_threads(diverge_omp_num_threads())
        for (index_t s3=0; s3<s->n_spin; ++s3)
        for (index_t o3=0; o3<s->n_orb; ++o3)
        for (index_t s1=0; s1<s->n_spin; ++s1)
        for (index_t o1=0; o1<s->n_orb; ++o1) {
            complex128_t tmp = 0.0;
            for (index_t s4=0; s4<s->n_spin; ++s4)
            for (index_t s2=0; s2<s->n_spin; ++s2)
            for (index_t b3=0; b3<s->bond_sizes[o3]; ++b3)
            for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
                // +
                const index_t idxGF_C_p = s->tu_ff[s->bond_offsets[o1]+b1].oto+s->n_orb
                            *(s4+s->n_spin*(s->tu_ff[s->bond_offsets[o3]+b3].oto + s->n_orb*s2));
                const index_t idx_C_p = s->ob_to_orbff[b1+s->n_bonds*o1]+s->n_orbff
                            *(s1+s->n_spin*(s4+s->n_spin*(s->ob_to_orbff[b3+s->n_bonds*o3]+s->n_orbff
                                *(s3+s->n_spin*(s2)))));
                tmp += GF[idxGF_C_p]*conj(C_in[idx_C_p]);
                // -
                const index_t idxGF_C_m = s->tu_ff[s->bond_offsets[o3]+b3].oto+s->n_orb
                            *(s2+s->n_spin*(s->tu_ff[s->bond_offsets[o1]+b1].oto + s->n_orb*s4));
                const index_t idx_C_m = s->ob_to_orbff[b3+s->n_bonds*o3]+s->n_orbff*(s3+s->n_spin
                            *(s2+s->n_spin*(s->ob_to_orbff[b1+s->n_bonds*o1]+s->n_orbff
                                *(s1+s->n_spin*(s4)))));
                tmp += conj(GF[idxGF_C_m])*C_in[idx_C_m];
            }
            self_en[o1+s->n_orb*(s1+s->n_spin*(o3+s->n_orb*(s3)))] -=prefac*tmp;
        }
    }
    if (phonons && D_in && s->D_flow) tu_data_phonons_gen_vert( phonons, 0.0, D_in, -1.0 );
    if (phonons && C_in && s->C_flow) tu_data_phonons_gen_vert( phonons, lambda, C_in, 1.0 );
}

static void selfenergy_flow_rs_SU2( tu_selfenergy_t* s, complex128_t* self_en,
        const complex128_t* P_in, complex128_t* C_in, complex128_t* D_in,
        const complex128_t* GF, double pref_subst, double lambda ) {
    tu_data_phonons_t* phonons = s->tu_data->phonons_reference;
    // selfenergy flow equation
    const index_t blocksize_self = POW2(s->n_orb*s->n_spin);
    const double prefac = pref_subst/(2.0*M_PI);
    //set all elements to zero
    ZERO_OUT( self_en, blocksize_self );

    if (s->D_flow) {
        // Hartree D
        if (phonons) tu_data_phonons_gen_vert( phonons, 0.0, D_in, 1.0 );
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t o1=0; o1<s->n_orb; ++o1)
        for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
            const index_t o3 = s->tu_ff[s->bond_offsets[o1]+b1].oto;
            complex128_t tmp = 0.0;
            for (index_t o4=0; o4<s->n_orb; ++o4)
            for (index_t b4=0; b4<s->bond_sizes[o4]; ++b4) {
                tmp += creal(GF[o4+s->n_orb*s->tu_ff[s->bond_offsets[o4]+b4].oto]*
                            D_in[s->ob_to_orbff[b1+s->n_bonds*o1]+s->n_orbff*s->ob_to_orbff[b4+s->n_bonds*o4]]);
            }
            self_en[o1+s->n_orb*o3] = -4.0*prefac*tmp;
        }
        if (phonons) tu_data_phonons_gen_vert( phonons, 0.0, D_in, -1.0 );
        // Fock D
        if (phonons) tu_data_phonons_gen_vert( phonons, lambda, D_in, 1.0 );
        #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
        for (index_t o3=0; o3<s->n_orb; ++o3)
        for (index_t o1=0; o1<s->n_orb; ++o1) {
            complex128_t tmp = 0.0;
            for (index_t b3=0; b3<s->bond_sizes[o3]; ++b3)
            for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
                const index_t idx_GF = s->tu_ff[s->bond_offsets[o1]+b1].oto
                                    +s->n_orb*s->tu_ff[s->bond_offsets[o3]+b3].oto;
                const index_t idx_CD = s->ob_to_orbff[b3+s->n_bonds*o3]
                                        +s->n_orbff*s->ob_to_orbff[b1+s->n_bonds*o1];
                tmp -= creal(GF[idx_GF]*D_in[idx_CD]);
            }
            self_en[o1+s->n_orb*o3] -= prefac*2.0*tmp;
        }
        if (phonons) tu_data_phonons_gen_vert( phonons, lambda, D_in, -1.0 );
    }
    if (s->C_flow && C_in) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t o1=0; o1<s->n_orb; ++o1)
        for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
            const index_t o3 = s->tu_ff[s->bond_offsets[o1]+b1].oto;
            complex128_t tmp = 0.0;
            for (index_t o4=0; o4<s->n_orb; ++o4)
            for (index_t b4=0; b4<s->bond_sizes[o4]; ++b4) {
                tmp -= creal(GF[o4+s->n_orb*s->tu_ff[s->bond_offsets[o4]+b4].oto]*
                     C_in[s->ob_to_orbff[b1+s->n_bonds*o1]+s->n_orbff*s->ob_to_orbff[b4+s->n_bonds*o4]]);

            }
            self_en[o1+s->n_orb*o3] -= 2.0*(prefac*tmp);
        }
        #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
        for (index_t o3=0; o3<s->n_orb; ++o3)
        for (index_t o1=0; o1<s->n_orb; ++o1) {
            complex128_t tmp = 0.0;
            for (index_t b3=0; b3<s->bond_sizes[o3]; ++b3)
            for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {

                const index_t idx_GF = s->tu_ff[s->bond_offsets[o1]+b1].oto
                                    +s->n_orb*s->tu_ff[s->bond_offsets[o3]+b3].oto;
                const index_t idx_CD = s->ob_to_orbff[b3+s->n_bonds*o3]
                                        +s->n_orbff*s->ob_to_orbff[b1+s->n_bonds*o1];

                tmp+= creal(GF[idx_GF]*(2.0*C_in[idx_CD]));
            }
            self_en[o1+s->n_orb*o3] -= prefac*2.0*(tmp);
        }
    }
    if (s->P_flow) {
        #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
        for (index_t o3=0; o3<s->n_orb; ++o3)
        for (index_t o1=0; o1<s->n_orb; ++o1) {
            complex128_t tmp = 0.0;
            for (index_t b3=0; b3<s->bond_sizes[o3]; ++b3)
            for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {

                const index_t idx_GF = s->tu_ff[s->bond_offsets[o1]+b1].oto
                                    +s->n_orb*s->tu_ff[s->bond_offsets[o3]+b3].oto;
                const index_t idx_P = s->ob_to_orbff[b1+s->n_bonds*o1]+s->n_orbff*s->ob_to_orbff[b3+s->n_bonds*o3];

                tmp += creal(GF[idx_GF] * 2.0*P_in[idx_P]);
            }
            self_en[o1+s->n_orb*o3] -= prefac*2.0*(tmp);
        }
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t o1=0; o1<s->n_orb; ++o1)
        for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
            for (index_t o4=0; o4<s->n_orb; ++o4)
            for (index_t b4=0; b4<s->bond_sizes[o4]; ++b4) {
                const index_t o3 = s->tu_ff[s->bond_offsets[o4]+b4].oto;
                self_en[o1+s->n_orb*o3] += 2.0*prefac
                                *creal(GF[s->tu_ff[s->bond_offsets[o1]+b1].oto+s->n_orb*o4]
                                *P_in[s->ob_to_orbff[b4+s->n_bonds*o4]
                                                    +s->n_orbff*s->ob_to_orbff[b1+s->n_bonds*o1]]);
            }
        }
    }
}

static void selfenergy_flow_mom(tu_selfenergy_t* s, complex128_t* self_en,
        const complex128_t* P_in, complex128_t* C_in, complex128_t* D_in,
        const complex128_t* GF, double pref_subst, double lambda ) {
    tu_data_phonons_t* phonons = s->tu_data->phonons_reference;
    // TODO: The non-SU2 one might be wrong (it kinda breaks crossing, but on the other hand not really...
    if (phonons && D_in && s->D_flow) tu_data_phonons_gen_vert( phonons, 0.0, D_in, 1.0 );
    if (phonons && C_in && s->C_flow) tu_data_phonons_gen_vert( phonons, lambda, C_in, -1.0 );
    // selfenergy flow equation
    const index_t blocksize_self = POW2(s->n_orb*s->n_spin)*s->nktot;
    ZERO_OUT( self_en, blocksize_self );
    ZERO_OUT( s->ff_helper, blocksize_self );
    ZERO_OUT( s->interpol_helper, s->n_orbff*POW2(s->n_spin) );
    if (diverge_mpi_comm_rank() == 0) {
        if (s->D_flow) {
            #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
            for (index_t s2=0; s2<s->n_spin; ++s2)
            for (index_t s4=0; s4<s->n_spin; ++s4)
            for (index_t o4=0; o4<s->n_orb; ++o4)
            for (index_t b4=0; b4<s->bond_sizes[o4]; ++b4) {
                const index_t bo4 = s->bond_offsets[o4]+b4;
                const index_t oto4 = s->tu_ff[bo4].oto;
                const index_t ffidx4 = s->tu_ff[bo4].ffidx;
                complex128_t tmp2 = 0.0;
                for (index_t k=0; k<s->nk; ++k) {
                    tmp2 += s->ffkmesh[IDX2(ffidx4, k, s->nk)] * (
                            GF[IDX5(s2, oto4, s4, o4, k, s->n_orb, s->n_spin, s->n_orb, s->nk)]+
                       conj(GF[IDX5(s4, o4, s2, oto4, k, s->n_orb, s->n_spin, s->n_orb, s->nk)]) );
                }
                s->interpol_helper[IDX3(s2, s4, bo4, s->n_spin, s->n_orbff)] += tmp2;
            }

            #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
            for (index_t o1=0; o1<s->n_orb; ++o1)
            for (index_t s1=0; s1<s->n_spin; ++s1)
            for (index_t s3=0; s3<s->n_spin; ++s3)
            for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
                const index_t mi1 = s->ob_to_orbff[b1+s->n_bonds*o1];
                complex128_t tmp = 0.0;
                for (index_t s2=0; s2<s->n_spin; ++s2)
                for (index_t s4=0; s4<s->n_spin; ++s4)
                for (index_t o4=0; o4<s->n_orb; ++o4)
                for (index_t b4=0; b4<s->bond_sizes[o4]; ++b4) {
                    const index_t bo4 = s->bond_offsets[o4]+b4;
                    const index_t mi4 = s->ob_to_orbff[b4+s->n_bonds*o4];
                    tmp += s->interpol_helper[IDX3(s2, s4, bo4, s->n_spin, s->n_orbff)]
                        * D_in[IDX6(s2, s4, mi4, s3, s1, mi1,
                                    s->n_spin, s->n_orbff, s->n_spin, s->n_spin, s->n_orbff)];
                }
                const index_t bo1 = s->bond_offsets[o1]+b1;
                const index_t oto1 = s->tu_ff[bo1].oto;
                const index_t ffidx1 = s->tu_ff[bo1].ffidx;
                for (index_t q=0; q<s->nk; ++q)
                    s->ff_helper[IDX5(s3, oto1, s1, o1, q, s->n_orb, s->n_spin, s->n_orb, s->nk)]
                        -= conj(s->ffkmesh[IDX2(ffidx1, q, s->nk)]) * tmp;
            }

            FFTW_EXECUTE_DFT(s->ifft_self_oop,s->ff_helper,self_en);
        }
    }
    if (s->P_flow) {
        reco('P',s,s->reco_helper,P_in,1.0,0.0);

        memcpy(s->greensfkt_helper,GF,sizeof(*s->greensfkt_helper) * POW2(s->n_spin*s->n_orb)*s->nk);
        FFTW_EXECUTE_DFT(s->ifft_vert_oop,s->reco_helper,s->interpol_helper);
        FFTW_EXECUTE_DFT(s->fft_self_oop,s->greensfkt_helper,s->ff_helper);

        #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<POW2(s->n_orbff)*POW4(s->n_spin)*s->nk; ++i)
            s->reco_helper[i] = conj(s->reco_helper[i]);
        #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<POW2(s->n_spin*s->n_orb)*s->nk; ++i)
            s->greensfkt_helper[i] = conj(s->greensfkt_helper[i]);

        FFTW_EXECUTE_DFT(s->ifft_vert_ip,s->reco_helper,s->reco_helper);
        FFTW_EXECUTE_DFT(s->fft_self_ip,s->greensfkt_helper,s->greensfkt_helper);

        #pragma omp parallel for collapse(4) num_threads(diverge_omp_num_threads())
        for (index_t o3=0; o3<s->n_orb; ++o3)
        for (index_t s3=0; s3<s->n_spin; ++s3)
        for (index_t s1=0; s1<s->n_spin; ++s1)
        for (index_t o1=0; o1<s->n_orb; ++o1)
        for (index_t b3=0; b3<s->bond_sizes[o3]; ++b3)
        for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
            const index_t bo1 = s->bond_offsets[o1]+b1;
            const index_t bo3 = s->bond_offsets[o3]+b3;
            const index_t oto1 = s->tu_ff[bo1].oto;
            const index_t oto3 = s->tu_ff[bo3].oto;
            const index_t mi1 = s->ob_to_orbff[b1+s->n_bonds*o1];
            const index_t mi2 = s->ob_to_orbff[b3+s->n_bonds*o3];
            const index_t* R1 = s->tu_ff[bo1].R;
            const index_t* R3 = s->tu_ff[bo3].R;
            for (index_t s4=0; s4<s->n_spin; ++s4)
            for (index_t s2=0; s2<s->n_spin; ++s2)
            for (index_t x=0; x<s->model->nk[0]; ++x)
            for (index_t y=0; y<s->model->nk[1]; ++y)
            for (index_t z=0; z<s->model->nk[2]; ++z) {
                const index_t ridx  = IDX3(x, y, z, s->model->nk[1], s->model->nk[2]);
                const index_t rb1b3 = rpb1mb3(s->model->nk, x, y, z, R1, R3);
                const index_t idx_GFP_p = IDX5(s2, oto1, s4, oto3, rb1b3,
                                               s->n_orb, s->n_spin, s->n_orb, s->nk);
                const index_t idx_GFP_m = IDX5(s4, oto3, s2, oto1, rb1b3,
                                               s->n_orb, s->n_spin, s->n_orb, s->nk);
                const complex128_t idx_P_p = s->interpol_helper[IDX5(s3+s->n_spin*s4, mi2, s1+s->n_spin*s2, mi1, rb1b3,
                                                            s->n_orbff, POW2(s->n_spin), s->n_orbff, s->nk)];
                const complex128_t idx_P_m = s->reco_helper[IDX5(s1+s->n_spin*s2, mi1, s3+s->n_spin*s4, mi2, rb1b3,
                                                            s->n_orbff, POW2(s->n_spin), s->n_orbff, s->nk)];
                self_en[IDX5(s3, o3, s1, o1, ridx, s->n_orb, s->n_spin, s->n_orb, s->nk)]
                    -= s->ff_helper[idx_GFP_p]*idx_P_p + s->greensfkt_helper[idx_GFP_m]*idx_P_m;
            }
        }
    }

    if (s->C_flow && C_in) {
        reco('C',s,s->reco_helper,C_in,1.0,0.0);

        memcpy(s->greensfkt_helper,GF, sizeof(*s->greensfkt_helper)*POW2(s->n_spin*s->n_orb)*s->nk);
        FFTW_EXECUTE_DFT(s->ifft_vert_oop,s->reco_helper,s->interpol_helper);
        FFTW_EXECUTE_DFT(s->ifft_self_oop,s->greensfkt_helper,s->ff_helper);

        #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<POW2(s->n_orbff)*POW4(s->n_spin)*s->nk; ++i)
            s->reco_helper[i] = conj(s->reco_helper[i]);
        #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<POW2(s->n_spin*s->n_orb)*s->nk; ++i)
            s->greensfkt_helper[i] = conj(s->greensfkt_helper[i]);

        FFTW_EXECUTE_DFT(s->ifft_vert_ip,s->reco_helper,s->reco_helper);
        FFTW_EXECUTE_DFT(s->ifft_self_ip,s->greensfkt_helper,s->greensfkt_helper);

        // P and C have similar form -> compress into one.
        #pragma omp parallel for collapse(4) num_threads(diverge_omp_num_threads())
        for (index_t o3=0; o3<s->n_orb; ++o3)
        for (index_t s3=0; s3<s->n_spin; ++s3)
        for (index_t s1=0; s1<s->n_spin; ++s1)
        for (index_t o1=0; o1<s->n_orb; ++o1)
        for (index_t b3=0; b3<s->bond_sizes[o3]; ++b3)
        for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
            const index_t bo1 = s->bond_offsets[o1]+b1;
            const index_t bo3 = s->bond_offsets[o3]+b3;
            const index_t oto1 = s->tu_ff[bo1].oto;
            const index_t oto3 = s->tu_ff[bo3].oto;
            const index_t mi1 = s->ob_to_orbff[b1+s->n_bonds*o1];
            const index_t mi2 = s->ob_to_orbff[b3+s->n_bonds*o3];
            const index_t* R1 = s->tu_ff[bo1].R;
            const index_t* R3 = s->tu_ff[bo3].R;
            for (index_t s4=0; s4<s->n_spin; ++s4)
            for (index_t s2=0; s2<s->n_spin; ++s2)
            for (index_t x=0; x<s->model->nk[0]; ++x)
            for (index_t y=0; y<s->model->nk[1]; ++y)
            for (index_t z=0; z<s->model->nk[2]; ++z) {
                const index_t ridx  = IDX3(x, y, z, s->model->nk[1], s->model->nk[2]);
                const index_t rb1b3 = rpb1mb3(s->model->nk, x, y, z, R1, R3);
                const index_t idx_GFC_m = IDX5(s2, oto3, s4, oto1, rb1b3,
                                               s->n_orb, s->n_spin, s->n_orb, s->nk);
                const index_t idx_GFC_p = IDX5(s4, oto1, s2, oto3, rb1b3,
                                               s->n_orb, s->n_spin, s->n_orb, s->nk);
                const complex128_t idx_C_p = s->interpol_helper[IDX5(s3+s->n_spin*s4, mi2, s1+s->n_spin*s2, mi1, rb1b3,
                                                            s->n_orbff, POW2(s->n_spin), s->n_orbff, s->nk)];
                const complex128_t idx_C_m = s->reco_helper[IDX5(s1+s->n_spin*s2, mi1, s3+s->n_spin*s4, mi2, rb1b3,
                                                            s->n_orbff, POW2(s->n_spin), s->n_orbff, s->nk)];
                self_en[IDX5(s3, o3, s1, o1, ridx, s->n_orb, s->n_spin, s->n_orb, s->nk)]
                    -= s->greensfkt_helper[idx_GFC_p]*idx_C_p + s->ff_helper[idx_GFC_m]*idx_C_m;
            }
        }
    }
    if (phonons && D_in && s->D_flow) tu_data_phonons_gen_vert( phonons, 0.0, D_in, -1.0 );
    if (phonons && C_in && s->C_flow) tu_data_phonons_gen_vert( phonons, lambda, C_in, 1.0 );

    apply_circ_cutoff( self_en, s->model->nk, 1, POW2(s->n_orb*s->n_spin), s );

    const index_t selfsize = POW2(s->n_orb*s->n_spin)*s->nk;
    diverge_mpi_allreduce_complex_sum_inplace(self_en, selfsize);
    if (s->nk != s->nktot) {
        const double prefac = pref_subst/(2.0*M_PI*(double)s->nk*(double)s->nk);
        for (index_t i=0; i<POW2(s->n_spin*s->n_orb)*s->nk; ++i)
            self_en[i] *= prefac;
        FFTW_EXECUTE_DFT(s->fft_fullself_ip,self_en,self_en);
        transpose(self_en,s->ff_helper, s->nk, s->n_orb*s->n_spin);
        diverge_symmetrize_2pt_coarse( s->model, self_en, s->ff_helper );
        transpose(self_en,s->ff_helper, -s->nk, -s->n_orb*s->n_spin);
        FFTW_EXECUTE_DFT(s->ifft_fullself_ip,self_en,self_en);
        tu_selfenergy_fft_interpolate(s,self_en);
    } else {
        const double prefac = pref_subst/(2.0*M_PI*(double)s->nk*(double)s->nk);
        for (index_t i=0; i<POW2(s->n_spin*s->n_orb)*s->nk; ++i)
            self_en[i] *= prefac;

        FFTW_EXECUTE_DFT(s->fft_fullself_ip,self_en,self_en);
    }
    transpose(self_en,s->ff_helper, s->nktot, s->n_orb*s->n_spin);
    diverge_symmetrize_2pt_fine( s->model, self_en, s->ff_helper );
}

static void selfenergy_flow_mom_SU2(tu_selfenergy_t* s, complex128_t* self_en,
        const complex128_t* P_in, complex128_t* C_in, complex128_t* D_in,
        const complex128_t* GF, double pref_subst, double lambda ) {
    tu_data_phonons_t* phonons = s->tu_data->phonons_reference;
    // selfenergy flow equation
    const index_t blocksize_self = s->nktot*POW2(s->n_spin*s->n_orb);
    //set all elements to zero
    ZERO_OUT( self_en, blocksize_self );
    ZERO_OUT( s->ff_helper, blocksize_self );
    ZERO_OUT( s->interpol_helper, s->n_orbff*POW2(s->n_spin) );
    // Hartree D/Fock C
    if (phonons && D_in && s->D_flow) tu_data_phonons_gen_vert( phonons, 0.0, D_in, 1.0 );

    if (diverge_mpi_comm_rank() == 0) {
        if (s->D_flow || s->C_flow ) {
            #pragma omp parallel for schedule(dynamic) num_threads(diverge_omp_num_threads())
            for (index_t o4=0; o4<s->n_orb; ++o4)
            for (index_t b4=0; b4<s->bond_sizes[o4]; ++b4) {
                complex128_t tmp = 0.0;
                for (index_t k=0; k<s->nk; ++k) {
                    const complex128_t GF_prefac = (GF[k+s->nk*(o4+s->n_orb*s->tu_ff[s->bond_offsets[o4]+b4].oto)])
                                            +conj(GF[k+s->nk*(s->tu_ff[s->bond_offsets[o4]+b4].oto+s->n_orb*o4)]);
                    tmp += GF_prefac*s->ffkmesh[k+s->nk*s->tu_ff[s->bond_offsets[o4]+b4].ffidx];
                }
                s->interpol_helper[s->bond_offsets[o4]+b4] += tmp;
            }
        }
        if (s->D_flow) {
            #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(diverge_omp_num_threads())
            for (index_t q=0; q<s->nk; ++q)
            for (index_t o1=0; o1<s->n_orb; ++o1)
            for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
                complex128_t tmp = 0.0;
                for (index_t o4=0; o4<s->n_orb; ++o4)
                for (index_t b4=0; b4<s->bond_sizes[o4]; ++b4) {
                    const index_t mi1 = s->ob_to_orbff[b1+s->n_bonds*o1];
                    const index_t mi2 = s->ob_to_orbff[b4+s->n_bonds*o4];
                    tmp += s->interpol_helper[s->bond_offsets[o4]+b4] * 2.0 * D_in[mi1+s->n_orbff*(mi2)];
                }
                s->ff_helper[q+s->nk*(o1+s->n_orb*s->tu_ff[s->bond_offsets[o1]+b1].oto)] -=
                                (tmp)*conj(s->ffkmesh[q+s->nk*s->tu_ff[s->bond_offsets[o1]+b1].ffidx]);
            }
        }
        if (s->C_flow && C_in) {
            #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(diverge_omp_num_threads())
            for (index_t q=0; q<s->nk; ++q)
            for (index_t o1=0; o1<s->n_orb; ++o1)
            for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
                complex128_t tmp = 0.0;
                for (index_t o4=0; o4<s->n_orb; ++o4)
                for (index_t b4=0; b4<s->bond_sizes[o4]; ++b4) {
                    const index_t mi1 = s->ob_to_orbff[b1+s->n_bonds*o1];
                    const index_t mi2 = s->ob_to_orbff[b4+s->n_bonds*o4];
                    tmp -= s->interpol_helper[s->bond_offsets[o4]+b4] * C_in[mi1+s->n_orbff*(mi2)];
                }
                s->ff_helper[q+s->nk*(o1+s->n_orb*s->tu_ff[s->bond_offsets[o1]+b1].oto)] -=
                                (tmp)*conj(s->ffkmesh[q+s->nk*s->tu_ff[s->bond_offsets[o1]+b1].ffidx]);
            }
        }
        if (s->C_flow || s->D_flow)
            FFTW_EXECUTE_DFT(s->ifft_self_oop,s->ff_helper,self_en);
    }
    if (phonons && D_in && s->D_flow) tu_data_phonons_gen_vert( phonons, 0.0, D_in, -1.0 );

    // Fock D/Hartree C
    if (phonons && D_in && s->D_flow) tu_data_phonons_gen_vert( phonons, lambda, D_in, 1.0 );
    if (s->C_flow && C_in)
        reco('C',s,s->reco_helper,C_in,1.0,0.0);

    if (s->D_flow) {
        if (!s->C_flow || !C_in) {
            reco('D',s,s->reco_helper,D_in,-0.5,0.0);
        } else {
            reco('D',s,s->reco_helper,D_in,-0.5,1.0);
        }
    }
    if (phonons && D_in && s->D_flow) tu_data_phonons_gen_vert( phonons, lambda, D_in, -1.0 );

    if (s->C_flow || s->D_flow) {
        memcpy(s->greensfkt_helper,GF, sizeof(*s->greensfkt_helper)*POW2(s->n_spin*s->n_orb)*s->nk);
        FFTW_EXECUTE_DFT(s->ifft_vert_oop,s->reco_helper,s->interpol_helper);
        FFTW_EXECUTE_DFT(s->ifft_self_oop,s->greensfkt_helper,s->ff_helper);
        #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<POW2(s->n_orbff)*POW4(s->n_spin)*s->nk; ++i)
            s->reco_helper[i] = conj(s->reco_helper[i]);
        #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<POW2(s->n_spin*s->n_orb)*s->nk; ++i)
            s->greensfkt_helper[i] = conj(s->greensfkt_helper[i]);
        FFTW_EXECUTE_DFT(s->ifft_vert_ip,s->reco_helper,s->reco_helper);
        FFTW_EXECUTE_DFT(s->ifft_self_ip,s->greensfkt_helper,s->greensfkt_helper);

        // P and C have similar form -> compress into one
        #pragma omp parallel for collapse(5) schedule(dynamic) num_threads(diverge_omp_num_threads())
        for (index_t z=0; z<s->model->nk[2]; ++z)
        for (index_t y=0; y<s->model->nk[1]; ++y)
        for (index_t x=0; x<s->model->nk[0]; ++x)
        for (index_t o3=0; o3<s->n_orb; ++o3)
        for (index_t o1=0; o1<s->n_orb; ++o1) {
            complex128_t tmp = 0.0;
            index_t ridx = z + s->model->nk[2]*(y+s->model->nk[1]*x);

            for (index_t b3=0; b3<s->bond_sizes[o3]; ++b3)
            for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
                index_t rb1b3 = rpb1mb3(s->model->nk,x,y,z,s->tu_ff[s->bond_offsets[o1]+b1].R,
                                                                s->tu_ff[s->bond_offsets[o3]+b3].R);
                const index_t mi1 =  s->ob_to_orbff[b1+s->n_bonds*o1];
                const index_t mi2 =  s->ob_to_orbff[b3+s->n_bonds*o3];
                    // + Lambda
                const index_t  idx_GFCD_p = rb1b3+s->nk*(s->tu_ff[s->bond_offsets[o3]+b3].oto+s->n_orb*
                                            s->tu_ff[s->bond_offsets[o1]+b1].oto);
                const index_t  idx_GFCD_m = rb1b3+s->nk*(s->tu_ff[s->bond_offsets[o1]+b1].oto+s->n_orb*
                                            s->tu_ff[s->bond_offsets[o3]+b3].oto);

                const complex128_t idx_C_p = s->interpol_helper[IDX3(mi2,mi1,rb1b3,s->n_orbff,s->nk)];
                const complex128_t idx_C_m = s->reco_helper[IDX3(mi1,mi2,rb1b3,s->n_orbff,s->nk)];

                tmp+= (s->greensfkt_helper[idx_GFCD_p]*idx_C_p
                            +(s->ff_helper[idx_GFCD_m]*idx_C_m));
            }
            self_en[ridx+s->nk*(o1+s->n_orb*o3)] -= 2.0*(tmp);
        }
    }
    if (s->P_flow) {
        reco('P',s,s->reco_helper,P_in,1.0,0.0);

        memcpy(s->greensfkt_helper,GF, sizeof(*s->greensfkt_helper)*POW2(s->n_spin*s->n_orb)*s->nk);
        FFTW_EXECUTE_DFT(s->ifft_vert_oop,s->reco_helper,s->interpol_helper);
        FFTW_EXECUTE_DFT(s->fft_self_oop,s->greensfkt_helper,s->ff_helper);

        #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<POW2(s->n_orbff)*POW4(s->n_spin)*s->nk; ++i)
            s->reco_helper[i] = conj(s->reco_helper[i]);
        #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<POW2(s->n_spin*s->n_orb)*s->nk; ++i)
            s->greensfkt_helper[i] = conj(s->greensfkt_helper[i]);
        FFTW_EXECUTE_DFT(s->ifft_vert_ip,s->reco_helper,s->reco_helper);
        FFTW_EXECUTE_DFT(s->fft_self_ip,s->greensfkt_helper,s->greensfkt_helper);

        #pragma omp parallel for collapse(5) schedule(dynamic) num_threads(diverge_omp_num_threads())
        for (index_t z=0; z<s->model->nk[2]; ++z)
        for (index_t y=0; y<s->model->nk[1]; ++y)
        for (index_t x=0; x<s->model->nk[0]; ++x)
        for (index_t o3=0; o3<s->n_orb; ++o3)
        for (index_t o1=0; o1<s->n_orb; ++o1) {
            complex128_t tmp = 0.0;
            index_t ridx = z + s->model->nk[2]*(y+s->model->nk[1]*x);

            for (index_t b3=0; b3<s->bond_sizes[o3]; ++b3)
            for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
                index_t rb1b3 = rpb1mb3(s->model->nk,x,y,z,s->tu_ff[s->bond_offsets[o1]+b1].R,
                                                                s->tu_ff[s->bond_offsets[o3]+b3].R);
                // + Lambda
                const index_t mi1 =  s->ob_to_orbff[b1+s->n_bonds*o1];
                const index_t mi2 =  s->ob_to_orbff[b3+s->n_bonds*o3];
                const index_t idx_GFP_p = rb1b3+s->nk*(s->tu_ff[s->bond_offsets[o3]+b3].oto+s->n_orb*
                                            s->tu_ff[s->bond_offsets[o1]+b1].oto);
                const index_t idx_GFP_m = rb1b3+s->nk*(s->tu_ff[s->bond_offsets[o1]+b1].oto+s->n_orb*
                                            s->tu_ff[s->bond_offsets[o3]+b3].oto);

                const complex128_t idx_P_p = s->interpol_helper[IDX3(mi2,mi1,rb1b3,s->n_orbff,s->nk)];
                const complex128_t idx_P_m = s->reco_helper[IDX3(mi1,mi2,rb1b3,s->n_orbff,s->nk)];
                tmp += (s->ff_helper[idx_GFP_p]*idx_P_p
                        +(s->greensfkt_helper[idx_GFP_m]*idx_P_m));
            }
            self_en[ridx+s->nk*(o1+s->n_orb*o3)] -= (tmp)*2.0;
        }
        #pragma omp parallel for collapse(4) schedule(dynamic) num_threads(diverge_omp_num_threads())
        for (index_t z=0; z<s->model->nk[2]; ++z)
        for (index_t y=0; y<s->model->nk[1]; ++y)
        for (index_t x=0; x<s->model->nk[0]; ++x)
        for (index_t o1=0; o1<s->n_orb; ++o1)
        for (index_t o4=0; o4<s->n_orb; ++o4)
        for (index_t b4=0; b4<s->bond_sizes[o4]; ++b4) {
            complex128_t tmp = 0.0;
            index_t ridx = z + s->model->nk[2]*(y+s->model->nk[1]*x);
            for (index_t b1=0; b1<s->bond_sizes[o1]; ++b1) {
                const index_t mi1 =  s->ob_to_orbff[b1+s->n_bonds*o1];
                const index_t mi2 =  s->ob_to_orbff[b4+s->n_bonds*o4];
                index_t gfrb1b3 = rpb1pb3(s->model->nk,x,y,z,s->tu_ff[s->bond_offsets[o1]+b1].R,
                                                        s->tu_ff[s->bond_offsets[o4]+b4].R);
                index_t vrb1b3 = rpb1(s->model->nk,x,y,z,s->tu_ff[s->bond_offsets[o1]+b1].R);

                const index_t  idx_GFP_p = gfrb1b3+s->nk*
                            (o4+s->n_orb*s->tu_ff[s->bond_offsets[o1]+b1].oto);
                const index_t  idx_GFP_m = gfrb1b3+s->nk*(s->tu_ff[s->bond_offsets[o1]+b1].oto+s->n_orb*o4);

                const complex128_t idx_P_p = s->interpol_helper[IDX3(mi2,mi1,vrb1b3,s->n_orbff,s->nk)];
                const complex128_t idx_P_m = s->reco_helper[IDX3(mi1,mi2,vrb1b3,s->n_orbff,s->nk)];

                tmp += s->ff_helper[idx_GFP_p]*idx_P_p
                    +s->greensfkt_helper[idx_GFP_m]*idx_P_m;
            }
            self_en[ridx+s->nk*(o1+s->n_orb*s->tu_ff[s->bond_offsets[o4]+b4].oto)] += tmp;
        }
    }

    apply_circ_cutoff( self_en, s->model->nk, 1, POW2(s->n_orb*s->n_spin), s );

    const index_t selfsize = POW2(s->n_orb*s->n_spin)*s->nk;
    diverge_mpi_allreduce_complex_sum_inplace(self_en, selfsize);

    if (s->nk != s->nktot) {
        const double prefac = pref_subst/(2.0*M_PI*(double)s->nk*(double)s->nk);
        for (index_t i=0; i<POW2(s->n_spin*s->n_orb)*s->nk; ++i)
            self_en[i] *= prefac;
        FFTW_EXECUTE_DFT(s->fft_fullself_ip,self_en,self_en);
        transpose(self_en,s->ff_helper, s->nk, s->n_orb*s->n_spin);
        diverge_symmetrize_2pt_coarse( s->model, self_en, s->ff_helper );
        transpose(self_en,s->ff_helper, -s->nk, -s->n_orb*s->n_spin);
        FFTW_EXECUTE_DFT(s->ifft_fullself_ip,self_en,self_en);
        tu_selfenergy_fft_interpolate(s,self_en);
    } else {
        const double prefac = pref_subst/(2.0*M_PI*(double)s->nk*(double)s->nk);
        for (index_t i=0; i<POW2(s->n_spin*s->n_orb)*s->nk; ++i)
            self_en[i] *= prefac;
        FFTW_EXECUTE_DFT(s->fft_fullself_ip,self_en,self_en);
    }
    transpose(self_en,s->ff_helper, s->nktot, s->n_orb*s->n_spin);
    diverge_symmetrize_2pt_fine( s->model, self_en, s->ff_helper );
}

static void generate_fftw_plans( tu_selfenergy_t* s ) {
    int dims[] = {(int)s->model->nk[0],(int)s->model->nk[1],(int)s->model->nk[2]};
    s->ifft_self_oop = fftw_plan_many_dft(3, dims, POW2(s->n_spin*s->n_orb),
        (fftw_complex*)(s->ff_helper), dims, 1, s->nk, (fftw_complex*)(s->greensfkt_helper),
        dims, 1, s->nk, FFTW_BACKWARD, FFTW_ESTIMATE);
    s->fft_self_oop = fftw_plan_many_dft(3, dims, POW2(s->n_spin*s->n_orb),
        (fftw_complex*)(s->ff_helper), dims, 1, s->nk, (fftw_complex*)(s->greensfkt_helper),
        dims, 1, s->nk, FFTW_FORWARD, FFTW_ESTIMATE);
    s->ifft_self_ip = fftw_plan_many_dft(3, dims, POW2(s->n_spin*s->n_orb),
        (fftw_complex*)(s->ff_helper), dims, 1, s->nk, (fftw_complex*)(s->ff_helper),
        dims, 1, s->nk,FFTW_BACKWARD, FFTW_ESTIMATE);
    s->fft_self_ip = fftw_plan_many_dft(3, dims, POW2(s->n_spin*s->n_orb),
        (fftw_complex*)(s->ff_helper), dims, 1, s->nk, (fftw_complex*)(s->ff_helper),
        dims, 1, s->nk,FFTW_FORWARD, FFTW_ESTIMATE);

    int nkff[3]; for (int i=0; i<3; ++i) nkff[i] = s->model->nk[i] * s->model->nkf[i];
    s->fft_self_oop_gfmesh = fftw_plan_many_dft(3, nkff, POW2(s->n_spin*s->n_orb),
        (fftw_complex*)(s->ff_helper), nkff, 1, s->nktot, (fftw_complex*)(s->greensfkt_helper),
        nkff, 1, s->nktot, FFTW_FORWARD, FFTW_ESTIMATE);
    s->ifft_fullself_ip = fftw_plan_many_dft(3, dims, POW2(s->n_spin*s->n_orb),
        (fftw_complex*)(s->ff_helper), dims, 1, s->nk, (fftw_complex*)(s->ff_helper),
        dims, 1, s->nk, FFTW_BACKWARD, FFTW_ESTIMATE);
    s->fft_fullself_ip = fftw_plan_many_dft(3, dims, POW2(s->n_spin*s->n_orb),
        (fftw_complex*)(s->ff_helper), dims, 1, s->nk, (fftw_complex*)(s->ff_helper),
        dims, 1, s->nk, FFTW_FORWARD, FFTW_ESTIMATE);

    s->ifft_vert_oop = fftw_plan_many_dft(3, dims, POW2(s->n_orbff)*POW4(s->n_spin),
        (fftw_complex*)(s->reco_helper), dims, 1, s->nk, (fftw_complex*)(s->interpol_helper),
        dims, 1, s->nk, FFTW_BACKWARD, FFTW_ESTIMATE);
    s->fft_vert_oop = fftw_plan_many_dft(3, dims, POW2(s->n_orbff)*POW4(s->n_spin),
        (fftw_complex*)(s->reco_helper), dims, 1, s->nk, (fftw_complex*)(s->interpol_helper),
        dims, 1, s->nk,FFTW_FORWARD, FFTW_ESTIMATE);
    s->ifft_vert_ip = fftw_plan_many_dft(3, dims, POW2(s->n_orbff)*POW4(s->n_spin),
        (fftw_complex*)(s->reco_helper), dims, 1, s->nk, (fftw_complex*)(s->reco_helper),
        dims, 1, s->nk, FFTW_BACKWARD, FFTW_ESTIMATE);
    s->fft_vert_ip = fftw_plan_many_dft(3, dims, POW2(s->n_orbff)*POW4(s->n_spin),
        (fftw_complex*)(s->reco_helper), dims, 1, s->nk, (fftw_complex*)(s->reco_helper),
        dims, 1, s->nk, FFTW_FORWARD, FFTW_ESTIMATE);
}

fftw_plan tu_selfenergy_get_ifft_self_ip( tu_selfenergy_t* s ) {
    return s->ifft_self_ip;
}
