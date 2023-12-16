/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

// XXX
// THIS IS AN UNUSED, UNTESTED PIECE OF CODE. IT IS SUPPOSED TO REPLACE THE
// CURRENT IMPLEMENTATION AS IT FEATURES MASSIVELY REDUCED COMMUNICATION
// OVERHEAD AS WELL AS CLEANER CODE (ONLY THIS ONE ALGORITHM IN ONE FUNCTION).
// THIS IS THEREFORE CLASSIFIED AS WORK-IN-PROGRESS!

#include <omp.h>

typedef struct {
    index_t s1, s2, s3, s4, m;
} task_t;

#define MY_BUFFER_GGM_IDX( i__, kff__ ) buffer_ggm_many[ i__ * nktot + kff__ ]
#define MY_BUFFER_GGP_IDX( i__, kff__ ) buffer_ggp_many[ i__ * nktot + kff__ ]

struct {
    tu_loop_mpi_data_t( tu_loop_t* loop );
    ~tu_loop_mpi_data_t();

    int rank;
    int size;
    MPI_Comm comm;

    const index_t* vnk;
    const index_t* vnkf;
    const index_t howmany;

    index_t dims[3];

    index_t kx_count, kx_displ;
    fftw_complex* ggm_buf_f;
    fftw_complex* ggp_buf_f;
    complex128_t* ggm_buf;
    complex128_t* ggp_buf;

    index_t alloc;

    fftw_plan fft_forward,
              fft_backward;

    int* kx_counts;
    int* kx_displs;

    void perform_fft( bool forward ) {
        fftw_mpi_execute_dft( forward ? fft_forward : fft_backward, ggm_buf_f, ggp_buf_f );
    }
} tu_loop_mpi_data_t;

typedef struct {
    index_t k;
    index_t q;
    int source_rank;
    int target_rank;
} kmap_t;

tu_loop_mpi_data_t::tu_loop_mpi_data_t( tu_loop_t* loop ) :
    vnk(loop->model->nk),
    vnkf(loop->model->nkf),
    howmany(loop->chunksize_max)
{
    diverge_mpi_barrier();
    rank = diverge_mpi_comm_rank(),
    size = diverge_mpi_comm_size();
    MPI_Comm comm = diverge_mpi_get_comm();
    for (int i=0; i<3; ++i) dims[i] = vnk[i]*vnkf[i];

    alloc = fftw_mpi_local_size_many( 3, dims, howmany,
            FFTW_MPI_DEFAULT_BLOCK, comm, &kx_count, &kx_displ );
    ggm_buf_f = fftw_alloc_complex(alloc);
    ggp_buf_f = fftw_alloc_complex(alloc);
    ggm_buf = (complex128_t*)ggm_buf_f;
    ggp_buf = (complex128_t*)ggp_buf_f;

    fft_forward = fftw_mpi_plan_many_dft( 3, dims, howmany,
            FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, ggm_buf, ggp_buf,
            comm, FFTW_FORWARD, FFTW_MEASURE );
    fft_backward = fftw_mpi_plan_many_dft( 3, dims, howmany,
            FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, ggm_buf, ggp_buf,
            comm, FFTW_BACKWARD, FFTW_MEASURE );

    kx_counts = (int*)calloc(size, sizeof(int));
    kx_displs = (int*)calloc(size, sizeof(int));
    int kx_count_i = kx_count;
    int kx_displ_i = kx_displ;
    diverge_mpi_allgather_int( &kx_count_i, kx_counts, 1 );
    diverge_mpi_allgather_int( &kx_displ_i, kx_displs, 1 );

    // kcomms contains the missing indices on *this rank*
    auto find_rank = [&]( index_t k )->int {
        for (int r=0; r<size; ++r)
            if ((kx_displs[r]*dims[1]*dims[2] <= k) &&
                (kx_displs[r]*dims[1]*dims[2] + kx_counts[r]*dims[1]*dims[2] > kx))
                return r;
        return -1;
    };
    vector<kmap_t> kcomms;
    kcomms.reserve(loop->my_nk);
    #pragma omp parallel for
    for (index_t q=0; q<loop->my_nk; ++q) {
        index_t k_want = loop->c2fmap[q];
        int r = find_rank( k_want );
        if (r != rank) {
            #pragma omp critical
            kcomms.push_back( {k_want, q, r, rank} );
        }
    }
    kcomms.shrink_to_fit();

    // and now kcomms contains *all indices that must be communicated*
    vector<int> kcommcounts(size);
    int kcommcount = kcomms.size();
    diverge_mpi_allgather_int( &kcommcount, kcommcounts.data(), 1 );
    vector<int> kcommdispls(size, 0);
    for (int i=1; i<size; ++i) kcommdispls[i] = kcommdispls[i-1] + kcommcounts[i-1];
    kcomms.resize( kcommcounts.back() + kcommdispls.back() );
    // move our share to the right location before gathering everything
    memmove( kcomms.data() + kcommdispls[r], kcomms.data(), sizeof(kmap_t)*kcommcounts[r] );
    diverge_mpi_allgatherv_bytes( kcomms.data(), sizeof(kmap_t), kcommcounts.data(), kcommdispls.data() );

    // have all the information, i.e., which elements to send where and what to
    // receive from where. starting communication (i.e. send/recv) setup
    vector<kmap_t> kc_send, kc_recv;
    for (const kmap_t& m: kcomms) {
        if (m.target_rank == rank) kc_recv.push_back(m);
        if (m.source_rank == rank) kc_send.push_back(m);
    }
    index_t nk_recv = kc_recv.size(), nk_send = kc_send.size();

    // stupid realloc
    index_t recv_size = nk_recv * howmany;
    index_t send_size = nk_send * howmany;
    if (recv_size > alloc) {
        mpi_wrn_printf( "more memory for recv (%.2fGiB) than for FFT (%.2fGiB)\n",
                (double)(recv_size*sizeof(complex128_t))/(double)(1024*1024*1024),
                (double)(alloc*sizeof(complex128_t))/(double)(1024*1024*1024) );
        fftw_free(ggp_buf_f);
        ggp_buf_f = fftw_alloc_complex(recv_size);
        ggp_buf = (complex128_t*)ggp_buf_f;
    }
    if (send_size > alloc) {
        mpi_wrn_printf( "more memory for send (%.2fGiB) than for FFT (%.2fGiB)\n",
                (double)(send_size*sizeof(complex128_t))/(double)(1024*1024*1024),
                (double)(alloc*sizeof(complex128_t))/(double)(1024*1024*1024) );
        fftw_free(ggm_buf_f);
        ggm_buf_f = fftw_alloc_complex(recv_size);
        ggm_buf = (complex128_t*)ggm_buf_f;
    }

    // sort the send and receive descriptors
    qsort( kc_send.data(), kc_send.size(), sizeof(kmap_t), []( const void* pa, const void* pb )->int{
            const kmap_t& a = *(const kmap_t*)(pa);
            const kmap_t& b = *(const kmap_t*)(pb);
            int result = 0;
            if (result = (a.target_rank - b.target_rank)) return result;
            if (result = (a.k - b.k)) return result;
            if (result = (a.q - b.q)) return result;
            return result;
    } );
    qsort( kc_recv.data(), kc_recv.size(), sizeof(kmap_t), []( const void* pa, const void* pb )->int{
            const kmap_t& a = *(const kmap_t*)(pa);
            const kmap_t& b = *(const kmap_t*)(pb);
            int result = 0;
            if (result = (a.source_rank - b.source_rank)) return result;
            if (result = (a.k - b.k)) return result;
            if (result = (a.q - b.q)) return result;
            return result;
    } );
    // and transfer this information to the counts/displs
    vector<int> sendcounts(size), senddispls(size,0),
                recvcounts(size), recvdispls(size,0);
    for (const kmap_t& m: kc_send) sendcounts[m.target_rank]++;
    for (const kmap_t& m: kc_recv) recvcounts[m.source_rank]++;
    for (int i=1; i<size; ++i) {
        senddispls[i] = senddispls[i-1] + sendcounts[i-1];
        recvdispls[i] = recvdispls[i-1] + recvcounts[i-1];
    }

    auto kidx_to_local = [&]( index_t k )->index_t{
        index_t kx = k / (dims[1]*dims[2]),
                kyz = k % (dims[1]*dims[2]);
        return (kx - kx_start) * dims[1]*dims[2] + kyz;
    };
    index_t elemsize = howmany;

    // put from result_buf into send_buf
    for (uindex_t i=0; i<kc_send.size(); ++i) {
        index_t kidx = kidx_to_local( kc_send[i].k );
        for (uindex_t j=0; j<elemsize; ++j)
            send_buf[i*elemsize+j] = result_buf[kidx*elemsize+j];
    }

    // send from send_buf to result_buf
    vector<int> xsendc = sendcounts, xsendd = senddispls,
                xrecvc = recvcounts, xrecvd = recvdispls;
    for (int i=0; i<size; ++i) {
        xsendc *= elemsize, xsendd *= elemsize;
        xrecvc *= elemsize, xrecvd *= elemsize;
    }
    diverge_mpi_alltoallv_complex( send_buf, xsendc.data(), xsendd.data(),
                                   result_buf, xrecvc.data(), xrecvd.data() );

    auto loop_idx = [&]( index_t q, index_t j )->index_t{
        return 0; // TODO
    };
    // put from result_buf into loop
    for (uindex_t i=0; i<kc_recv.size(); ++i) {
        index_t kidx = kidx_to_local( kc_recv[i].k );
        for (uindex_t j=0; j<elemsize; ++j)
            loop[loop_idx(kc_recv[i].q, j)] = result_buf[i*elemsize+j]
    }

    diverge_mpi_barrier();
}

tu_loop_mpi_data_t::~tu_loop_mpi_data_t() {
    free(kx_counts);
    free(kx_displs);
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    fftw_free(ggp_buf_f);
    fftw_free(ggm_buf_f);
}


template<bool pp, bool simple, bool subtract>
void tu_loop_t::S_loop_static_mpi(complex128_t* loop, index_t ntasks, index_t taskoff, bool rescale) const {
    if (ntasks == -1) {
        ntasks = simple ? POW2(n_orbff) : (index_t)tu_data->unique_mi_l.size();
        taskoff = 0;
    }
    const double prefac = -1.0/(2.0*M_PI);
    const index_t* vnk = model->nk;
    const index_t* vnkf = model->nkf;
    int dims[] = {(int)(vnk[0]*vnkf[0]), (int)(vnk[1]*vnkf[1]), (int)(vnk[2]*vnkf[2])};

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

    fftw_mpi_plan* p = pp ? fft_many_forward_mpi : fft_many_backward_mpi;
    const index_t kx_start = p->get_local_0_start();
    const index_t kx_stop = p->get_local_0_start() + p->get_local_n0();

    while (stack.size() > 0) {
        index_t chunksize = MIN((index_t)stack.size(), chunksize_max);
        for (index_t i=0; i<chunksize; ++i) {
            tasks[i] = stack.back();
            stack.pop_back();
        }

        // fill buffers
        #pragma omp parallel for num_threads(diverge_omp_num_threads()) schedule(dynamic)
        for (index_t i=0; i<chunksize; ++i) {
            const index_t& s1 = tasks[i].s1, s2 = tasks[i].s2, s3 = tasks[i].s3, s4 = tasks[i].s4, m = tasks[i].m;

            index_t mo3 = simple ? ((taskoff+m)/n_orbff) :
                                   tu_data->unique_mi_r[m+taskoff];
            index_t mo1 = simple ? ((taskoff+m) - mo3*n_orbff) :
                                   tu_data->unique_mi_l[m+taskoff];
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

        fftw_execute_dft( fft_T_to, (fftw_complex*)buffer_ggm_many, (fftw_complex*)buffer_ggp_many );
        p->execute( buffer_ggp_many, buffer_ggm_many, true );
        fftw_execute_dft( fft_T_from, (fftw_complex*)buffer_ggm_many, (fftw_complex*)buffer_ggp_many );

        // put into loop
        #pragma omp parallel for num_threads(diverge_omp_num_threads()) schedule(dynamic)
        for (index_t i=0; i<chunksize; ++i) {
            const index_t& s1 = tasks[i].s1, s2 = tasks[i].s2, s3 = tasks[i].s3, s4 = tasks[i].s4, m = tasks[i].m;

            index_t mo3 = simple ? ((taskoff+m)/n_orbff) :
                                   tu_data->unique_mi_r[m+taskoff];
            index_t mo1 = simple ? ((taskoff+m) - mo3*n_orbff) :
                                   tu_data->unique_mi_l[m+taskoff];
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

    if (rescale) {
        if (!simple) reconstruct_from_reduced(loop);
        scale_loop(loop, prefac, my_nk*POW4(n_spin)*POW2(n_orbff)*n_frq);
    }
}

