/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include <omp.h>

typedef struct {
    index_t s1, s2, s3, s4, m;
} task_t;

template<bool pp, bool subtract>
void tu_loop_t::S_loop_static_simple(complex128_t* loop, index_t ntasks, index_t taskoff, bool rescale) const {

    if (which_fft_simple == tu_fft_algorithm_mpi_new) {
        tu_mpi_loop_calc( mpi_loop, loop, ntasks, taskoff, rescale, pp, true, subtract );
        return;
    }

    if(ntasks == -1) ntasks = POW2(n_orbff), taskoff = 0;
    const double prefac = -1.0/(2.0*M_PI);
    const index_t* vnk = model->nk;
    const index_t* vnkf = model->nkf;
    int dims[] = {(int)(vnk[0]*vnkf[0]), (int)(vnk[1]*vnkf[1]),
                                         (int)(vnk[2]*vnkf[2])};

    // XXX algorithm choice: greedy XXX
    if (which_fft_simple == tu_fft_algorithm_greedy) {

    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
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
                index_t kz = ((pp?1:-1)*kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = ((pp?1:-1)*kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = ((pp?1:-1)*kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGm[kff] = pp ?
                    GF_fft_conj_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))] :
                    GF_fft_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGm[kff] *= pp ?
                    GF_fft_plus[kff+nktot*(ob1+n_orb*(s2+n_spin*(ob3+n_orb*s4)))] :
                    GF_fft_plus[kff+nktot*(ob3+n_orb*(s2+n_spin*(ob1+n_orb*s4)))];
            }

            for(index_t kffx = 0; kffx<dims[0];++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kz = pp ? (kffz - off1[2] + off3[2]+3*dims[2])%dims[2] :
                                  (-kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = pp ? (kffy - off1[1] + off3[1]+3*dims[1])%dims[1] :
                                  (-kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = pp ? (kffx - off1[0] + off3[0]+3*dims[0])%dims[0] :
                                  (-kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGp[kff] = pp ?
                    GF_fft_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))] :
                    GF_fft_conj_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGp[kff] *= pp ?
                    GF_fft_conj_plus[kff+nktot*(ob1+n_orb*(s2+n_spin*(ob3+n_orb*s4)))] :
                    GF_fft_conj_plus[kff+nktot*(ob3+n_orb*(s2+n_spin*(ob1+n_orb*s4)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGm[kff] += MY_BUFFER_GGp[kff];
            }

            fftw_execute_dft( pp ? fft_single_forward : fft_single_backward,
                    (fftw_complex*)MY_BUFFER_GGm, (fftw_complex*)MY_BUFFER_GGp );

            for(index_t q = 0; q<my_nk;++q) {
                index_t idx_to = pp ?
                    mo1+n_orbff*(s1+s2*n_spin+POW2(n_spin)*(mo3+n_orbff*(s3 + s4*n_spin+ POW2(n_spin)*q))) :
                    mo1+n_orbff*(s1+s4*n_spin+POW2(n_spin)*(mo3+n_orbff*(s3 + s2*n_spin+ POW2(n_spin)*q)));
                if constexpr(subtract)
                    loop[idx_to] -= MY_BUFFER_GGp[c2fmap[q]];
                else
                    loop[idx_to] = MY_BUFFER_GGp[c2fmap[q]];
            }
        }
    }

    // XXX algorithm choice: split/mpi XXX
    } else {

    // create stack
    vector<task_t> stack( POW4(n_spin) * ntasks );
    index_t idx = stack.size();
    for(index_t s3 = 0; s3<n_spin;++s3)
    for(index_t s2 = 0; s2<n_spin;++s2)
    for(index_t s4 = 0; s4<n_spin;++s4)
    for(index_t s1 = 0; s1<n_spin;++s1)
    for(index_t m = 0; m<ntasks;++m)
        stack[--idx] = {s1, s2, s3, s4, m};
    vector<task_t> tasks(chunksize_max);

    while (stack.size() > 0) {
        index_t chunksize = MIN((index_t)stack.size(), chunksize_max);
        for (index_t i=0; i<chunksize; ++i) {
            tasks[i] = stack.back();
            stack.pop_back();
        }

#define MY_BUFFER_GGM_IDX( i__, kff__ ) buffer_ggm_many[ i__ * nktot + kff__ ]
#define MY_BUFFER_GGP_IDX( i__, kff__ ) buffer_ggp_many[ i__ * nktot + kff__ ]

        index_t kx_start = 0, kx_stop = dims[0];
        if (which_fft_simple == tu_fft_algorithm_mpi) {
            fftw_mpi_plan_t* p = pp ? fft_many_forward_mpi : fft_many_backward_mpi;
            kx_start = fftw_mpi_plan_get_local_0_start(p);
            kx_stop = fftw_mpi_plan_get_local_0_start(p) + fftw_mpi_plan_get_local_n0(p);
        }

        // fill buffers
        #pragma omp parallel for num_threads(diverge_omp_num_threads()) schedule(dynamic)
        for (index_t i=0; i<chunksize; ++i) {
            const index_t& s1 = tasks[i].s1, s2 = tasks[i].s2, s3 = tasks[i].s3, s4 = tasks[i].s4, m = tasks[i].m;

            index_t mo3 = (taskoff+m)/n_orbff;
            index_t mo1 = (taskoff+m) - mo3*n_orbff;
            const index_t o1 = mi_to_ofrom[mo1];
            const index_t ob1 = mi_to_oto[mo1];
            const index_t* off1 = mi_to_R+3*mo1;
            const index_t o3 = mi_to_ofrom[mo3];
            const index_t ob3 = mi_to_oto[mo3];
            const index_t* off3 = mi_to_R+3*mo3;
            for(index_t kffx = kx_start; kffx<kx_stop;++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kz = ((pp?1:-1)*kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = ((pp?1:-1)*kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = ((pp?1:-1)*kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGM_IDX( i, kff ) = pp ?
                    GF_fft_conj_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))] :
                    GF_fft_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))];
                MY_BUFFER_GGM_IDX( i, kff ) *= pp ?
                    GF_fft_plus[kff+nktot*(ob1+n_orb*(s2+n_spin*(ob3+n_orb*s4)))] :
                    GF_fft_plus[kff+nktot*(ob3+n_orb*(s2+n_spin*(ob1+n_orb*s4)))];
            }

            for(index_t kffx = kx_start; kffx<kx_stop;++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kz = pp ? (kffz - off1[2] + off3[2]+3*dims[2])%dims[2] :
                                  (-kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = pp ? (kffy - off1[1] + off3[1]+3*dims[1])%dims[1] :
                                  (-kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = pp ? (kffx - off1[0] + off3[0]+3*dims[0])%dims[0] :
                                  (-kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGP_IDX( i, kff ) = pp ?
                    GF_fft_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))] :
                    GF_fft_conj_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))];
                MY_BUFFER_GGP_IDX( i, kff ) *= pp ?
                    GF_fft_conj_plus[kff+nktot*(ob1+n_orb*(s2+n_spin*(ob3+n_orb*s4)))] :
                    GF_fft_conj_plus[kff+nktot*(ob3+n_orb*(s2+n_spin*(ob1+n_orb*s4)))];
            }
            for(index_t kffx = kx_start; kffx<kx_stop;++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGM_IDX( i, kff ) += MY_BUFFER_GGP_IDX( i, kff );
            }
        }

        // execute DFTs
        if (which_fft_simple == tu_fft_algorithm_mpi) {
            fftw_mpi_plan_t* p = pp ? fft_many_forward_mpi : fft_many_backward_mpi;
            #ifdef TRANSPOSE_MPI_FFT_BUFFERS
            fftw_execute_dft( fft_T_to, (fftw_complex*)buffer_ggm_many, (fftw_complex*)buffer_ggp_many );
            fftw_mpi_plan_execute( p, buffer_ggp_many, buffer_ggm_many, true );
            fftw_execute_dft( fft_T_from, (fftw_complex*)buffer_ggm_many, (fftw_complex*)buffer_ggp_many );
            #else
            fftw_mpi_plan_execute( p, buffer_ggm_many, buffer_ggp_many, false );
            #endif
        } else {
            double tick = 0, tock = 0;
            if (extra_timings_enabled) tick = diverge_mpi_wtime();
            fftw_execute_dft( pp ? fft_many_forward : fft_many_backward,
                (fftw_complex*)buffer_ggm_many, (fftw_complex*)buffer_ggp_many );
            if (extra_timings_enabled) tock = diverge_mpi_wtime();
            extra_tvec->at(4) += tock - tick;
        }

        // put into loop
        #pragma omp parallel for num_threads(diverge_omp_num_threads()) schedule(dynamic)
        for (index_t i=0; i<chunksize; ++i) {
            const index_t& s1 = tasks[i].s1, s2 = tasks[i].s2, s3 = tasks[i].s3, s4 = tasks[i].s4, m = tasks[i].m;

            index_t mo3 = (taskoff+m)/n_orbff;
            index_t mo1 = (taskoff+m) - mo3*n_orbff;
            for(index_t q = 0; q<my_nk;++q) {
                index_t idx_to = pp ?
                    mo1+n_orbff*(s1+s2*n_spin+POW2(n_spin)*(mo3+n_orbff*(s3 + s4*n_spin+ POW2(n_spin)*q))) :
                    mo1+n_orbff*(s1+s4*n_spin+POW2(n_spin)*(mo3+n_orbff*(s3 + s2*n_spin+ POW2(n_spin)*q)));
                if constexpr(subtract)
                    loop[idx_to] -= MY_BUFFER_GGP_IDX( i, c2fmap[q] );
                else
                    loop[idx_to] = MY_BUFFER_GGP_IDX( i, c2fmap[q] );
            }
        }
    }

    // XXX algorithm choice: end XXX
    }

    if (rescale)
        scale_loop(loop, prefac, my_nk*POW4(n_spin)*POW2(n_orbff));
}

template<bool pp, bool subtract>
void tu_loop_t::S_loop_static_red(complex128_t* loop, index_t ntasks, index_t taskoff, bool rescale) const {

    if (which_fft_red == tu_fft_algorithm_mpi_new) {
        tu_mpi_loop_calc( mpi_loop, loop, ntasks, taskoff, rescale, pp, false, subtract );
        return;
    }

    if(ntasks == -1) ntasks = (index_t)tu_data->unique_mi_l.size(), taskoff = 0;
    const double prefac = -1.0/(2.0*M_PI);
    const index_t* vnk = model->nk;
    const index_t* vnkf = model->nkf;
    int dims[] = {(int)(vnk[0]*vnkf[0]), (int)(vnk[1]*vnkf[1]), (int)(vnk[2]*vnkf[2])};

    // XXX algorithm choice: greedy XXX
    if (which_fft_red == tu_fft_algorithm_greedy) {

    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
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
                index_t kz = ((pp ? 1:-1)*kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = ((pp ? 1:-1)*kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = ((pp ? 1:-1)*kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGm[kff] = pp ?
                    GF_fft_conj_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))] :
                    GF_fft_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGm[kff] *= pp ?
                    GF_fft_plus[kff+nktot*(ob1+n_orb*(s2+n_spin*(ob3+n_orb*s4)))] :
                    GF_fft_plus[kff+nktot*(ob3+n_orb*(s2+n_spin*(ob1+n_orb*s4)))];
            }

            for(index_t kffx = 0; kffx<dims[0];++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kz = ((pp?1:-1)*kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = ((pp?1:-1)*kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = ((pp?1:-1)*kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGp[kff] = pp ?
                    GF_fft_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))] :
                    GF_fft_conj_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGp[kff] *= pp ?
                    GF_fft_conj_plus[kff+nktot*(ob1+n_orb*(s2+n_spin*(ob3+n_orb*s4)))] :
                    GF_fft_conj_plus[kff+nktot*(ob3+n_orb*(s2+n_spin*(ob1+n_orb*s4)))];
            }
            for(index_t kff = 0; kff<nktot;++kff){
                MY_BUFFER_GGm[kff] += MY_BUFFER_GGp[kff];
            }

            fftw_execute_dft( pp ? fft_single_forward : fft_single_backward,
                    (fftw_complex*)MY_BUFFER_GGm, (fftw_complex*)MY_BUFFER_GGp );

            for(index_t q = 0; q<my_nk;++q) {
                index_t idx_to = pp ?
                    mo1+n_orbff*(s1+s2*n_spin+POW2(n_spin)*(mo3+n_orbff*(s3 + s4*n_spin+ POW2(n_spin)*q))) :
                    mo1+n_orbff*(s1+s4*n_spin+POW2(n_spin)*(mo3+n_orbff*(s3 + s2*n_spin+ POW2(n_spin)*q)));
                if constexpr(subtract)
                    loop[idx_to] -= MY_BUFFER_GGp[c2fmap[q]];
                else
                    loop[idx_to] = MY_BUFFER_GGp[c2fmap[q]];
            }
        }
    }

    // XXX algorithm choice: split/mpi XXX
    } else {

    // create stack
    vector<task_t> stack( POW4(n_spin) * ntasks );
    index_t idx = stack.size();
    for(index_t s3 = 0; s3<n_spin;++s3)
    for(index_t s2 = 0; s2<n_spin;++s2)
    for(index_t s4 = 0; s4<n_spin;++s4)
    for(index_t s1 = 0; s1<n_spin;++s1)
    for(index_t m = 0; m<ntasks;++m)
        stack[--idx] = {s1, s2, s3, s4, m};
    vector<task_t> tasks(chunksize_max);

    while (stack.size() > 0) {
        index_t chunksize = MIN((index_t)stack.size(), chunksize_max);
        for (index_t i=0; i<chunksize; ++i) {
            tasks[i] = stack.back();
            stack.pop_back();
        }

        index_t kx_start = 0, kx_stop = dims[0];
        if (which_fft_red == tu_fft_algorithm_mpi) {
            fftw_mpi_plan_t* p = pp ? fft_many_forward_mpi : fft_many_backward_mpi;
            kx_start = fftw_mpi_plan_get_local_0_start(p);
            kx_stop = fftw_mpi_plan_get_local_0_start(p) + fftw_mpi_plan_get_local_n0(p);
        }

        // fill buffers
        #pragma omp parallel for num_threads(diverge_omp_num_threads()) schedule(dynamic)
        for (index_t i=0; i<chunksize; ++i) {
            const index_t& s1 = tasks[i].s1, s2 = tasks[i].s2, s3 = tasks[i].s3, s4 = tasks[i].s4, m = tasks[i].m;
            index_t mo1 = tu_data->unique_mi_l[m+taskoff];
            index_t mo3 = tu_data->unique_mi_r[m+taskoff];
            const index_t o1 = mi_to_ofrom[mo1];
            const index_t ob1 = mi_to_oto[mo1];
            const index_t* off1 = mi_to_R+3*mo1;
            const index_t o3 = mi_to_ofrom[mo3];
            const index_t ob3 = mi_to_oto[mo3];
            const index_t* off3 = mi_to_R+3*mo3;
            for(index_t kffx = kx_start; kffx<kx_stop;++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kz = ((pp ? 1:-1)*kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = ((pp ? 1:-1)*kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = ((pp ? 1:-1)*kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGM_IDX( i, kff ) = pp ?
                    GF_fft_conj_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))] :
                    GF_fft_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))];
                MY_BUFFER_GGM_IDX( i, kff ) *= pp ?
                    GF_fft_plus[kff+nktot*(ob1+n_orb*(s2+n_spin*(ob3+n_orb*s4)))] :
                    GF_fft_plus[kff+nktot*(ob3+n_orb*(s2+n_spin*(ob1+n_orb*s4)))];
            }

            for(index_t kffx = kx_start; kffx<kx_stop;++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kz = ((pp?1:-1)*kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = ((pp?1:-1)*kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = ((pp?1:-1)*kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGP_IDX( i, kff ) = pp ?
                    GF_fft_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))] :
                    GF_fft_conj_plus[kid + nktot*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))];
                MY_BUFFER_GGP_IDX( i, kff ) *= pp ?
                    GF_fft_conj_plus[kff+nktot*(ob1+n_orb*(s2+n_spin*(ob3+n_orb*s4)))] :
                    GF_fft_conj_plus[kff+nktot*(ob3+n_orb*(s2+n_spin*(ob1+n_orb*s4)))];
            }
            for(index_t kffx = kx_start; kffx<kx_stop;++kffx)
            for(index_t kffy = 0; kffy<dims[1];++kffy)
            for(index_t kffz = 0; kffz<dims[2];++kffz){
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx);
                MY_BUFFER_GGM_IDX( i, kff ) += MY_BUFFER_GGP_IDX( i, kff );
            }
        }

        // execute DFTs
        if (which_fft_red == tu_fft_algorithm_mpi) {
            fftw_mpi_plan_t* p = pp ? fft_many_forward_mpi : fft_many_backward_mpi;
            #ifdef TRANSPOSE_MPI_FFT_BUFFERS
            fftw_execute_dft( fft_T_to, (fftw_complex*)buffer_ggm_many, (fftw_complex*)buffer_ggp_many );
            fftw_mpi_plan_execute( p, buffer_ggp_many, buffer_ggm_many, true );
            fftw_execute_dft( fft_T_from, (fftw_complex*)buffer_ggm_many, (fftw_complex*)buffer_ggp_many );
            #else
            fftw_mpi_plan_execute( p, buffer_ggm_many, buffer_ggp_many, false );
            #endif
        } else {
            double tick = 0, tock = 0;
            if (extra_timings_enabled) tick = diverge_mpi_wtime();
            fftw_execute_dft( pp ? fft_many_forward : fft_many_backward,
                (fftw_complex*)buffer_ggm_many, (fftw_complex*)buffer_ggp_many );
            if (extra_timings_enabled) tock = diverge_mpi_wtime();
            extra_tvec->at(4) += tock-tick;
        }

        // put into loop
        #pragma omp parallel for num_threads(diverge_omp_num_threads()) schedule(dynamic)
        for (index_t i=0; i<chunksize; ++i) {
            const index_t& s1 = tasks[i].s1, s2 = tasks[i].s2, s3 = tasks[i].s3, s4 = tasks[i].s4, m = tasks[i].m;
            index_t mo1 = tu_data->unique_mi_l[m+taskoff];
            index_t mo3 = tu_data->unique_mi_r[m+taskoff];
            for(index_t q = 0; q<my_nk;++q) {
                index_t idx_to = pp ?
                    mo1+n_orbff*(s1+s2*n_spin+POW2(n_spin)*(mo3+n_orbff*(s3 + s4*n_spin+ POW2(n_spin)*q))) :
                    mo1+n_orbff*(s1+s4*n_spin+POW2(n_spin)*(mo3+n_orbff*(s3 + s2*n_spin+ POW2(n_spin)*q)));
                if constexpr(subtract)
                    loop[idx_to] -= MY_BUFFER_GGP_IDX( i, c2fmap[q] );
                else
                    loop[idx_to] = MY_BUFFER_GGP_IDX( i, c2fmap[q] );
            }
        }
    }

    // XXX algorithm choice: end XXX
    }

    if(rescale) {
        reconstruct_from_reduced(loop);
        scale_loop(loop, prefac, my_nk*POW4(n_spin)*POW2(n_orbff));
    }
}

template<bool pp, bool subtract>
void tu_loop_t::S_loop_static_rs(complex128_t* L) const {
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
        if constexpr(pp) {
            if constexpr(subtract)
                L[mo1 + n_orbff*((s1+n_spin*s2) + POW2(n_spin)*(mo3+n_orbff*(s3+n_spin*s4)))] -=
                    prefac*(GF_fft_conj_plus[o1+n_orb*(s1+n_spin*(o3+n_orb*s3))]
                    *GF_fft_plus[ob1+n_orb*(s2+n_spin*(ob3+n_orb*s4))]
                    +GF_fft_plus[o1+n_orb*(s1+n_spin*(o3+n_orb*s3))]
                    *GF_fft_conj_plus[ob1+n_orb*(s2+n_spin*(ob3+n_orb*s4))]);
            else
                L[mo1 + n_orbff*((s1+n_spin*s2) + POW2(n_spin)*(mo3+n_orbff*(s3+n_spin*s4)))] =
                    prefac*(GF_fft_conj_plus[o1+n_orb*(s1+n_spin*(o3+n_orb*s3))]
                    *GF_fft_plus[ob1+n_orb*(s2+n_spin*(ob3+n_orb*s4))]
                    +GF_fft_plus[o1+n_orb*(s1+n_spin*(o3+n_orb*s3))]
                    *GF_fft_conj_plus[ob1+n_orb*(s2+n_spin*(ob3+n_orb*s4))]);
        } else {
            if constexpr(subtract)
                L[mo1 + n_orbff*((s1+n_spin*s4) + POW2(n_spin)*(mo3+n_orbff*(s3+n_spin*s2)))] -=
                    prefac*(GF_fft_plus[o1+n_orb*(s1+n_spin*(o3+n_orb*s3))]
                    *GF_fft_plus[ob3+n_orb*(s2+n_spin*(ob1+n_orb*s4))]
                    +GF_fft_conj_plus[o1+n_orb*(s1+n_spin*(o3+n_orb*s3))]
                    *GF_fft_conj_plus[ob3+n_orb*(s2+n_spin*(ob1+n_orb*s4))]);
            else
                L[mo1 + n_orbff*((s1+n_spin*s4) + POW2(n_spin)*(mo3+n_orbff*(s3+n_spin*s2)))] =
                    prefac*(GF_fft_plus[o1+n_orb*(s1+n_spin*(o3+n_orb*s3))]
                    *GF_fft_plus[ob3+n_orb*(s2+n_spin*(ob1+n_orb*s4))]
                    +GF_fft_conj_plus[o1+n_orb*(s1+n_spin*(o3+n_orb*s3))]
                    *GF_fft_conj_plus[ob3+n_orb*(s2+n_spin*(ob1+n_orb*s4))]);
        }
    }
}

