/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "propagator.hpp"
#include "../diverge_internals_struct.h"
#include <omp.h>


void tu_loop_t::S_ph_loop_static_simple(complex128_t* loop, index_t ntasks, index_t taskoff, bool rescale) const {
    if(ntasks == -1) ntasks = POW2(n_orbff), taskoff = 0;
    const double prefac = -1.0/(2.0*M_PI);
    const index_t* vnk = model->nk;
    const index_t* vnkf = model->nkf;
    int dims[] = {(int)(vnk[0]*vnkf[0]), (int)(vnk[1]*vnkf[1]),
                                         (int)(vnk[2]*vnkf[2])};
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        fftw_plan _fft_L = fftPH[omp_get_thread_num()];
        complex128_t* MY_BUFFER_GGm = buffer_ggm[omp_get_thread_num()];
        complex128_t* MY_BUFFER_GGp = buffer_ggp[omp_get_thread_num()];
        #pragma omp for collapse(5) schedule(dynamic,1)
        for(index_t s3 = 0; s3<n_spin;++s3)
        for(index_t s2 = 0; s2<n_spin;++s2)
        for(index_t s4 = 0; s4<n_spin;++s4)
        for(index_t s1 = 0; s1<n_spin;++s1)
        for(index_t m = 0; m<ntasks;++m) {
            index_t mo3 = (taskoff+m)/n_orbff;
            index_t mo1 = (taskoff+m) - mo3*n_orbff;
            const index_t o1 = mi_to_ofrom[mo1];
            const index_t ob1 = mi_to_oto[mo1];
            const index_t* off1 = mi_to_R+3*mo1;
            const index_t o3 = mi_to_ofrom[mo3];
            const index_t ob3 = mi_to_oto[mo3];
            const index_t* off3 = mi_to_R+3*mo3;
            for(index_t kffx = 0; kffx<dims[0];++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kz = (-kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = (-kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = (-kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGm[kff] = GF_fft_plus[kid + nktot*(o1+n_orb*(s1+n_spin
                                                    *(o3+n_orb*s3)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGm[kff] *= GF_fft_plus[kff+nktot*(ob3+n_orb*(s2+n_spin
                                                *(ob1+n_orb*s4)))];
            }

            for(index_t kffx = 0; kffx<dims[0];++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kz = (-kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = (-kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = (-kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGp[kff] = GF_fft_conj_plus[kid + nktot*(o1+n_orb*(s1+n_spin
                                    *(o3+n_orb*s3)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGp[kff] *= GF_fft_conj_plus[kff+nktot*(ob3+n_orb*(s2+n_spin
                                                *(ob1+n_orb*s4)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGm[kff] += MY_BUFFER_GGp[kff];
            }

            fftw_execute(_fft_L);

            for(index_t q = 0; q<my_nk;++q) {
                index_t idx_to = mo1+n_orbff*(s1+s4*n_spin
                        +POW2(n_spin)*(mo3+n_orbff*
                        (s3 + s2*n_spin+ POW2(n_spin)*q)));
                loop[idx_to] = MY_BUFFER_GGp[c2fmap[q]];
            }
        }
    }
    if(rescale) scale_loop(loop, prefac, my_nk*POW4(n_spin)*POW2(n_orbff)*n_frq);
}


void tu_loop_t::S_pp_loop_static_simple(complex128_t* loop, index_t ntasks, index_t taskoff, bool rescale) const {
    if(ntasks == -1) ntasks = POW2(n_orbff), taskoff = 0;
    const double prefac = -1.0/(2.0*M_PI);
    const index_t* vnk = model->nk;
    const index_t* vnkf = model->nkf;
    int dims[] = {(int)(vnk[0]*vnkf[0]), (int)(vnk[1]*vnkf[1]),
                                         (int)(vnk[2]*vnkf[2])};
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        fftw_plan _fft_L = fftPP[omp_get_thread_num()];
        complex128_t* MY_BUFFER_GGm = buffer_ggm[omp_get_thread_num()];
        complex128_t* MY_BUFFER_GGp = buffer_ggp[omp_get_thread_num()];
        #pragma omp for collapse(5) schedule(dynamic,1)
        for(index_t s3 = 0; s3<n_spin;++s3)
        for(index_t s2 = 0; s2<n_spin;++s2)
        for(index_t s4 = 0; s4<n_spin;++s4)
        for(index_t s1 = 0; s1<n_spin;++s1)
        for(index_t m = 0; m<ntasks;++m) {
            index_t mo3 = (taskoff+m)/n_orbff;
            index_t mo1 = (taskoff+m) - mo3*n_orbff;
            const index_t o1 = mi_to_ofrom[mo1];
            const index_t ob1 = mi_to_oto[mo1];
            const index_t* off1 = mi_to_R+3*mo1;
            const index_t o3 = mi_to_ofrom[mo3];
            const index_t ob3 = mi_to_oto[mo3];
            const index_t* off3 = mi_to_R+3*mo3;
            for(index_t kffx = 0; kffx<dims[0];++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kz = (kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = (kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = (kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGm[kff] = GF_fft_conj_plus[kid + nktot*(o1+n_orb*(s1+n_spin
                                                *(o3+n_orb*s3)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGm[kff] *= GF_fft_plus[kff+nktot*(ob1+n_orb*(s2+n_spin
                                                *(ob3+n_orb*s4)))];
            }

            for(index_t kffx = 0; kffx<dims[0];++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kz = (kffz - off1[2] + off3[2]+3*dims[2])%dims[2];
                index_t ky = (kffy - off1[1] + off3[1]+3*dims[1])%dims[1];
                index_t kx = (kffx - off1[0] + off3[0]+3*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGp[kff] = GF_fft_plus[kid + nktot*(o1+n_orb*(s1+n_spin
                                                *(o3+n_orb*s3)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGp[kff] *= GF_fft_conj_plus[kff+nktot*(ob1+n_orb*(s2+n_spin
                                                *(ob3+n_orb*s4)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGm[kff] += MY_BUFFER_GGp[kff];
            }

            fftw_execute(_fft_L);

            for(index_t q = 0; q<my_nk;++q)
            {
                index_t idx_to = mo1+n_orbff*(s1+s2*n_spin
                        +POW2(n_spin)*(mo3+n_orbff*
                        (s3 + s4*n_spin+ POW2(n_spin)*q)));
                loop[idx_to] = MY_BUFFER_GGp[c2fmap[q]];
            }
        }
    }
    if(rescale) scale_loop(loop, prefac, my_nk*POW4(n_spin)*POW2(n_orbff)*n_frq);
}


void tu_loop_t::S_ph_loop_static_red(complex128_t* loop, index_t ntasks, index_t taskoff, bool rescale) const {
    if(ntasks == -1) ntasks = (index_t)tu_data->unique_mi_l.size(), taskoff = 0;
    const double prefac = -1.0/(2.0*M_PI);
    const index_t* vnk = model->nk;
    const index_t* vnkf = model->nkf;
    int dims[] = {(int)(vnk[0]*vnkf[0]), (int)(vnk[1]*vnkf[1]),
                                         (int)(vnk[2]*vnkf[2])};
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        fftw_plan _fft_L = fftPH[omp_get_thread_num()];
        complex128_t* MY_BUFFER_GGm = buffer_ggm[omp_get_thread_num()];
        complex128_t* MY_BUFFER_GGp = buffer_ggp[omp_get_thread_num()];
        #pragma omp for collapse(5) schedule(dynamic,1)
        for(index_t s3 = 0; s3<n_spin;++s3)
        for(index_t s2 = 0; s2<n_spin;++s2)
        for(index_t s4 = 0; s4<n_spin;++s4)
        for(index_t s1 = 0; s1<n_spin;++s1)
        for(index_t m = 0; m<ntasks;++m){
            index_t mo1 = tu_data->unique_mi_l[m+taskoff];
            index_t mo3 = tu_data->unique_mi_r[m+taskoff];
            const index_t o1 = mi_to_ofrom[mo1];
            const index_t ob1 = mi_to_oto[mo1];
            const index_t* off1 = mi_to_R+3*mo1;
            const index_t o3 = mi_to_ofrom[mo3];
            const index_t ob3 = mi_to_oto[mo3];
            const index_t* off3 = mi_to_R+3*mo3;
            for(index_t kffx = 0; kffx<dims[0];++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kz = (-kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = (-kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = (-kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGm[kff] = GF_fft_plus[kid + nktot*(o1+n_orb*(s1+n_spin
                                                    *(o3+n_orb*s3)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGm[kff] *= GF_fft_plus[kff+nktot*(ob3+n_orb*(s2+n_spin
                                                *(ob1+n_orb*s4)))];
            }

            for(index_t kffx = 0; kffx<dims[0];++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kz = (-kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = (-kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = (-kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGp[kff] = GF_fft_conj_plus[kid + nktot*(o1+n_orb*(s1+n_spin
                                    *(o3+n_orb*s3)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGp[kff] *= GF_fft_conj_plus[kff+nktot*(ob3+n_orb*(s2+n_spin
                                                *(ob1+n_orb*s4)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGm[kff] += MY_BUFFER_GGp[kff];
            }

            fftw_execute(_fft_L);

            for(index_t q = 0; q<my_nk;++q)
            {
                index_t idx_to = mo1+n_orbff*(s1+s4*n_spin
                        +POW2(n_spin)*(mo3+n_orbff*
                        (s3 + s2*n_spin+ POW2(n_spin)*q)));
                loop[idx_to] = MY_BUFFER_GGp[c2fmap[q]];
            }
        }
    }
    if(rescale) {
        reconstruct_from_reduced(loop);
        scale_loop(loop, prefac, my_nk*POW4(n_spin)*POW2(n_orbff)*n_frq);
    }
}


void tu_loop_t::S_pp_loop_static_red(complex128_t* loop, index_t ntasks, index_t taskoff, bool rescale) const {
    if(ntasks == -1) ntasks = (index_t)tu_data->unique_mi_l.size(), taskoff = 0;
    const double prefac = -1.0/(2.0*M_PI);
    const index_t* vnk = model->nk;
    const index_t* vnkf = model->nkf;
    int dims[] = {(int)(vnk[0]*vnkf[0]), (int)(vnk[1]*vnkf[1]),
                                         (int)(vnk[2]*vnkf[2])};
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        fftw_plan _fft_L = fftPP[omp_get_thread_num()];
        complex128_t* MY_BUFFER_GGm = buffer_ggm[omp_get_thread_num()];
        complex128_t* MY_BUFFER_GGp = buffer_ggp[omp_get_thread_num()];
        #pragma omp for collapse(5) schedule(dynamic,1)
        for(index_t s3 = 0; s3<n_spin;++s3)
        for(index_t s2 = 0; s2<n_spin;++s2)
        for(index_t s4 = 0; s4<n_spin;++s4)
        for(index_t s1 = 0; s1<n_spin;++s1)
        for(index_t m = 0; m<ntasks;++m){
            index_t mo1 = tu_data->unique_mi_l[m+taskoff];
            index_t mo3 = tu_data->unique_mi_r[m+taskoff];
            const index_t o1 = mi_to_ofrom[mo1];
            const index_t ob1 = mi_to_oto[mo1];
            const index_t* off1 = mi_to_R+3*mo1;
            const index_t o3 = mi_to_ofrom[mo3];
            const index_t ob3 = mi_to_oto[mo3];
            const index_t* off3 = mi_to_R+3*mo3;
            for(index_t kffx = 0; kffx<dims[0];++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kz = (kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = (kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = (kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGm[kff] = GF_fft_conj_plus[kid + nktot*(o1+n_orb*(s1+n_spin
                                                *(o3+n_orb*s3)))];
            }

            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGm[kff] *= GF_fft_plus[kff+nktot*(ob1+n_orb*(s2+n_spin
                                                *(ob3+n_orb*s4)))];
            }

            for(index_t kffx = 0; kffx<dims[0];++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kz = (kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = (kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = (kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGp[kff] = GF_fft_plus[kid + nktot*(o1+n_orb*(s1+n_spin
                                                *(o3+n_orb*s3)))];
            }

            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGp[kff] *= GF_fft_conj_plus[kff+nktot*(ob1+n_orb*(s2+n_spin
                                                *(ob3+n_orb*s4)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGm[kff] += MY_BUFFER_GGp[kff];
            }

            fftw_execute(_fft_L);

            for(index_t q = 0; q<my_nk;++q)
            {
                index_t idx_to = mo1+n_orbff*(s1+s2*n_spin
                        +POW2(n_spin)*(mo3+n_orbff*
                        (s3 + s4*n_spin+ POW2(n_spin)*q)));
                loop[idx_to] = MY_BUFFER_GGp[c2fmap[q]];
            }
        }
    }
    if(rescale) {
        reconstruct_from_reduced(loop);
        scale_loop(loop, prefac, my_nk*POW4(n_spin)*POW2(n_orbff)*n_frq);
    }
}


void tu_loop_t::S_ph_loop_static_rs(complex128_t* L) const {
    const double prefac = -1.0/(2.0*M_PI);
    #pragma omp parallel for collapse(3) schedule(dynamic) num_threads(diverge_omp_num_threads())
    for(index_t s3 = 0; s3<(n_spin);++s3)
    for(index_t s4 = 0; s4<(n_spin);++s4)
    for(index_t mo3 = 0; mo3<n_orbff;++mo3)
    for(index_t s1 = 0; s1<(n_spin);++s1)
    for(index_t s2 = 0; s2<(n_spin);++s2)
    for(index_t mo1 = 0; mo1<n_orbff;++mo1) {
        const index_t o1 = mi_to_ofrom[mo1];
        const index_t ob1 = mi_to_oto[mo1];
        const index_t o3 = mi_to_ofrom[mo3];
        const index_t ob3 = mi_to_oto[mo3];
        L[mo1 + n_orbff*((s1+n_spin*s4) + POW2(n_spin)*(mo3+n_orbff*(s3+n_spin*s2)))] =
            prefac*(GF_fft_plus[o1+n_orb*(s1+n_spin*(o3+n_orb*s3))]
                *GF_fft_plus[ob3+n_orb*(s2+n_spin*(ob1+n_orb*s4))]
                +GF_fft_conj_plus[o1+n_orb*(s1+n_spin*(o3+n_orb*s3))]
                *GF_fft_conj_plus[ob3+n_orb*(s2+n_spin*(ob1+n_orb*s4))]);
    }
}


void tu_loop_t::S_pp_loop_static_rs(complex128_t* L) const {
    const double prefac = -1.0/(2.0*M_PI);
    #pragma omp parallel for collapse(3) schedule(dynamic) num_threads(diverge_omp_num_threads())
    for(index_t s3 = 0; s3<(n_spin);++s3)
    for(index_t s4 = 0; s4<(n_spin);++s4)
    for(index_t mo3 = 0; mo3<n_orbff;++mo3)
    for(index_t s1 = 0; s1<(n_spin);++s1)
    for(index_t s2 = 0; s2<(n_spin);++s2)
    for(index_t mo1 = 0; mo1<n_orbff;++mo1) {
        const index_t o1 = mi_to_ofrom[mo1];
        const index_t ob1 = mi_to_oto[mo1];
        const index_t o3 = mi_to_ofrom[mo3];
        const index_t ob3 = mi_to_oto[mo3];
        L[mo1 + n_orbff*((s1+n_spin*s2) + POW2(n_spin)*(mo3+n_orbff*(s3+n_spin*s4)))] =
            prefac*(GF_fft_conj_plus[o1+n_orb*(s1+n_spin*(o3+n_orb*s3))]
                *GF_fft_plus[ob1+n_orb*(s2+n_spin*(ob3+n_orb*s4))]
                +GF_fft_plus[o1+n_orb*(s1+n_spin*(o3+n_orb*s3))]
                *GF_fft_conj_plus[ob1+n_orb*(s2+n_spin*(ob3+n_orb*s4))]);
    }
}


void tu_loop_t::reconstruct_from_reduced(complex128_t* loop) const
{
    index_t n_unique = (index_t)tu_data->unique_mi_l.size();
    #pragma omp for collapse(4) schedule(dynamic,1)
    for(index_t q = 0; q<my_nk;++q)
    for(index_t sr = 0; sr<POW2(n_spin);++sr)
    for(index_t sl = 0; sl<POW2(n_spin);++sl)
    for(index_t m = 0; m<n_unique;++m){
        index_t gmo1 = tu_data->unique_mi_l[m];
        index_t gmo3 = tu_data->unique_mi_r[m];
        complex128_t from = loop[gmo1+n_orbff*(sl +POW2(n_spin)*(gmo3+n_orbff*
                                (sr + POW2(n_spin)*q)))];
        for(index_t l = 1; l < tu_data->lmap_len[m];++l){
            index_t mo1 = tu_data->lmap_l[l+tu_data->lmap_off[m]];
            index_t mo3 = tu_data->lmap_r[l+tu_data->lmap_off[m]];
            loop[mo1+n_orbff*(sl +POW2(n_spin)*(mo3+n_orbff*
                                (sr + POW2(n_spin)*q)))] = from;
        }
    }
}
