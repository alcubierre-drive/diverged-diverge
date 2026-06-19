/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "propagator_details.h"
#include "propagator_cpu_mpi.h"
#include <omp.h>
#include "../diverge_model.h"
#include "../diverge_internals_struct.h"
#include "../misc/shared_mem.h"
#include "../misc/mergesort.h"

// global var
unsigned propagator_cpu_mpi_planning_patience = FFTW_MEASURE;

#if defined(USE_MPI) && defined(USE_MPI_FFTW)

#ifdef MPI_VERBOSE_ALLOCATION_PRINT
#define alloc_print( sz ) mpi_vrb_printf( "allocating %.2fGiB (%s:%i)\n", \
        (double)(sz)/(double)(1024.*1024.*1024), __FILE__, __LINE__ )
#else
#define alloc_print( sz ) {}
#endif

#include <fftw3-mpi.h>

static fftw_plan plan_transpose( int rows, int cols, complex128_t* in, complex128_t* out ) {
    fftw_iodim howmany_dims[2];
    howmany_dims[0].n  = rows;
    howmany_dims[0].is = cols;
    howmany_dims[0].os = 1;
    howmany_dims[1].n  = cols;
    howmany_dims[1].is = 1;
    howmany_dims[1].os = rows;
    return fftw_plan_guru_dft( 0, NULL, 2, howmany_dims, (fftw_complex*)in,
            (fftw_complex*)out, 0, propagator_cpu_mpi_planning_patience);
}

static GF_FFTW(plan) GF_plan_transpose( int rows, int cols, gf_complex_t* in_out ) {
    GF_FFTW(iodim) howmany_dims[2];
    howmany_dims[0].n  = rows;
    howmany_dims[0].is = cols;
    howmany_dims[0].os = 1;
    howmany_dims[1].n  = cols;
    howmany_dims[1].is = 1;
    howmany_dims[1].os = rows;
    return GF_FFTW(plan_guru_dft)( 0, NULL, 2, howmany_dims, in_out, in_out, 0,
            propagator_cpu_mpi_planning_patience );
}


static inline int find_rank( const tu_mpi_loop_t* t, index_t k ) {
    for (int r=0; r<t->size; ++r)
        if ((t->kx_displs[r]*t->dims[1]*t->dims[2] <= k) &&
            ((t->kx_displs[r]*t->dims[1]*t->dims[2] + t->kx_counts[r]*t->dims[1]*t->dims[2]) > k))
                return r;
    return -1;
}

static inline index_t kidx_to_local( const tu_mpi_loop_t* t, index_t k ) {
    index_t kx = k / (t->dims[1]*t->dims[2]),
            kyz = k % (t->dims[1]*t->dims[2]);
    return (kx - t->kx_displ) * t->dims[1]*t->dims[2] + kyz;
}

static inline index_t kmap_cmp_send(const kmap_t* a, const kmap_t* b) {
    index_t result = 0;
    if ((result = (a->target_rank - b->target_rank))) return result;
    if ((result = (a->q - b->q))) return result;
    if ((result = (a->k - b->k))) return result;
    return result;
}
#undef MERGESORT_COMP
#define MERGESORT_COMP(x,y) kmap_cmp_send(&(x),&(y))
MERGESORT_IMPL(kmap_sendsort, kmap_t)

static inline index_t kmap_cmp_recv(const kmap_t* a, const kmap_t* b) {
    index_t result = 0;
    if ((result = (a->source_rank - b->source_rank))) return result;
    if ((result = (a->q - b->q))) return result;
    if ((result = (a->k - b->k))) return result;
    return result;
}
#undef MERGESORT_COMP
#define MERGESORT_COMP(x,y) kmap_cmp_recv(&(x),&(y))
MERGESORT_IMPL(kmap_recvsort, kmap_t)

tu_mpi_loop_t* tu_mpi_loop_init( const index_t nk[3], const index_t nkf[3],
        index_t chunksize_max, const index_t* c2fmap, index_t my_nk,
        index_t n_spin, index_t n_orb, index_t n_orbff,
        const index_t* unique_mi_l, index_t unique_mi_l_size,
        const index_t* unique_mi_r, const index_t* mi_to_ofrom,
        const index_t* mi_to_oto, const index_t* mi_to_R,
        complex128_t** p_GF_fft_plus, complex128_t** p_GF_fft_conj_plus,
        void* handle, diverge_model_t* model ) {

    diverge_mpi_barrier();
    double tick = 0, tock = 0;

    tick = diverge_mpi_wtime();
    tu_mpi_loop_t* t = calloc(1, sizeof(tu_mpi_loop_t));
    alloc_print( sizeof(tu_mpi_loop_t) );
    t->enable_timings = 1;
    t->rank = diverge_mpi_comm_rank(),
    t->size = diverge_mpi_comm_size();
    t->comm = diverge_mpi_get_comm();
    for (int i=0; i<3; ++i) {
        t->vnk[i] = nk[i];
        t->vnkf[i] = nkf[i];
        t->dims[i] = nk[i]*nkf[i];
    }
    t->chunksize_max = chunksize_max;
    t->my_nk = my_nk;
    t->c2fmap = c2fmap;
    t->n_spin = n_spin;
    t->n_orb =  n_orb;
    t->n_orbff =  n_orbff;
    t->unique_mi_l =  unique_mi_l;
    t->unique_mi_l_size =  unique_mi_l_size;
    t->unique_mi_r =  unique_mi_r;
    t->mi_to_ofrom =  mi_to_ofrom;
    t->mi_to_oto =  mi_to_oto;
    t->mi_to_R =  mi_to_R;
    t->p_GF_fft_plus =  p_GF_fft_plus;
    t->p_GF_fft_conj_plus =  p_GF_fft_conj_plus;
    t->tu_loop_handle = handle;
    t->model = model;
    tock = diverge_mpi_wtime();
    mpi_vrb_printf( "TU-MPI prop: %.2fs for data\n", tock-tick );

    tick = diverge_mpi_wtime();
    t->alloc = fftw_mpi_local_size_many( 3, t->dims, t->chunksize_max,
            FFTW_MPI_DEFAULT_BLOCK, t->comm, &t->kx_count, &t->kx_displ );
    t->ggm_buf = fftw_alloc_complex(t->alloc);
    alloc_print( sizeof(complex128_t)*t->alloc );
    t->ggp_buf = fftw_alloc_complex(t->alloc);
    alloc_print( sizeof(complex128_t)*t->alloc );

    t->fft_forward = fftw_mpi_plan_many_dft( 3, t->dims, t->chunksize_max,
            FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, t->ggm_buf, t->ggp_buf,
            t->comm, FFTW_FORWARD, propagator_cpu_mpi_planning_patience );
    t->fft_backward = fftw_mpi_plan_many_dft( 3, t->dims, t->chunksize_max,
            FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, t->ggm_buf, t->ggp_buf,
            t->comm, FFTW_BACKWARD, propagator_cpu_mpi_planning_patience );
    t->fft_transpose = plan_transpose( t->chunksize_max,
            t->kx_count*t->dims[1]*t->dims[2], t->ggm_buf, t->ggp_buf );
    tock = diverge_mpi_wtime();
    mpi_vrb_printf( "TU-MPI prop: %.2fs for FFT plans\n", tock-tick );

    tick = diverge_mpi_wtime();
    t->kx_counts = (int*)calloc(t->size, sizeof(int));
    alloc_print( sizeof(int)*t->size );
    t->kx_displs = (int*)calloc(t->size, sizeof(int));
    alloc_print( sizeof(int)*t->size );
    int kx_count_i = t->kx_count;
    int kx_displ_i = t->kx_displ;
    diverge_mpi_allgather_int( &kx_count_i, t->kx_counts, 1 );
    diverge_mpi_allgather_int( &kx_displ_i, t->kx_displs, 1 );

    // kcomms contains the missing indices on *this rank*
    kmap_t* kcomms = calloc(t->my_nk, sizeof(kmap_t));
    index_t kcomms_size = 0;
    t->c2fmap_local = bitary_alloc(t->my_nk);
    alloc_print( bitary_nbytes(t->my_nk) );
    for (index_t q=0; q<t->my_nk; ++q) {
        index_t k_want = t->c2fmap[q];
        int r = find_rank( t, k_want );
        if (r != t->rank) {
            kcomms[kcomms_size++] = (kmap_t){.k=k_want, .q=q, .source_rank=r, .target_rank=t->rank};
        } else {
            bitary_set(t->c2fmap_local, q);
        }
    }
    // shrink kcomms
    kcomms = realloc( kcomms, sizeof(kmap_t)*kcomms_size );

    // prepare for communication of kcomms indices
    int *kcommcounts = calloc(t->size, sizeof(int)),
        *kcommdispls = calloc(t->size, sizeof(int));
    int kcommcount = kcomms_size;
    diverge_mpi_allgather_int( &kcommcount, kcommcounts, 1 );
    for (int i=1; i<t->size; ++i) kcommdispls[i] = kcommdispls[i-1] + kcommcounts[i-1];

    kcomms_size = kcommcounts[t->size-1]+kcommdispls[t->size-1];
    kcomms = realloc( kcomms, sizeof(kmap_t)*kcomms_size );
    // move our share to the right location before gathering everything
    memmove( kcomms + kcommdispls[t->rank], kcomms, sizeof(kmap_t)*kcommcounts[t->rank] );
    // and now kcomms contains *all indices that must be communicated*
    diverge_mpi_allgatherv_bytes( kcomms, sizeof(kmap_t), kcommcounts, kcommdispls );

    // free auxiliary arrays
    free( kcommcounts );
    free( kcommdispls );

    // have all the information, i.e., which elements to send where and what to
    // receive from where. starting communication (i.e. send/recv) setup
    t->kc_send_count = 0,
    t->kc_recv_count = 0;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t kc=0; kc<kcomms_size; ++kc) {
        if (kcomms[kc].target_rank == t->rank)
            #pragma omp atomic
            t->kc_recv_count++;
        if (kcomms[kc].source_rank == t->rank)
            #pragma omp atomic
            t->kc_send_count++;
    }
    t->kc_send = calloc(t->kc_send_count, sizeof(kmap_t));
    alloc_print( t->kc_send_count * sizeof(kmap_t) );
    t->kc_recv = calloc(t->kc_recv_count, sizeof(kmap_t));
    alloc_print( t->kc_recv_count * sizeof(kmap_t) );
    t->kc_send_count = 0;
    t->kc_recv_count = 0;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t kc=0; kc<kcomms_size; ++kc) {
        if (kcomms[kc].target_rank == t->rank)
            #pragma omp critical
            t->kc_recv[t->kc_recv_count++] = kcomms[kc];
        if (kcomms[kc].source_rank == t->rank)
            #pragma omp critical
            t->kc_send[t->kc_send_count++] = kcomms[kc];
    }
    free( kcomms );

    // stupid realloc
    index_t realloc_size = MAX(t->kc_recv_count, t->kc_send_count) * t->chunksize_max;
    if (realloc_size > t->alloc) {
        mpi_wrn_printf( "more memory for recv (%.2fGiB) than for FFT (%.2fGiB)\n",
                (double)(realloc_size*sizeof(complex128_t))/(double)(1024*1024*1024),
                (double)(t->alloc*sizeof(complex128_t))/(double)(1024*1024*1024) );
        fftw_free(t->ggp_buf);
        t->ggp_buf = fftw_alloc_complex(realloc_size);
        fftw_free(t->ggm_buf);
        t->ggm_buf = fftw_alloc_complex(realloc_size);
    }
    tock = diverge_mpi_wtime();
    mpi_vrb_printf( "TU-MPI prop: %.2fs for kcomms\n", tock-tick );

    // sort the send and receive descriptors
    tick = diverge_mpi_wtime();
    kmap_sendsort( t->kc_send, t->kc_send_count, diverge_omp_num_threads(), 0 );
    kmap_recvsort( t->kc_recv, t->kc_recv_count, diverge_omp_num_threads(), 0 );
    tock = diverge_mpi_wtime();
    mpi_vrb_printf( "TU-MPI prop: %.2fs for sort\n", tock-tick );

    tick = diverge_mpi_wtime();
    t->sendcounts = calloc(t->size, sizeof(int));
    t->senddispls = calloc(t->size, sizeof(int));
    t->recvcounts = calloc(t->size, sizeof(int));
    t->recvdispls = calloc(t->size, sizeof(int));
    for (index_t m=0; m<t->kc_send_count; ++m) t->sendcounts[t->kc_send[m].target_rank]++;
    for (index_t m=0; m<t->kc_recv_count; ++m) t->recvcounts[t->kc_recv[m].source_rank]++;
    for (int i=1; i<t->size; ++i) {
        t->senddispls[i] = t->senddispls[i-1] + t->sendcounts[i-1];
        t->recvdispls[i] = t->recvdispls[i-1] + t->recvcounts[i-1];
    }
    t->xsendc = calloc(t->size, sizeof(int));
    t->xsendd = calloc(t->size, sizeof(int));
    t->xrecvc = calloc(t->size, sizeof(int));
    t->xrecvd = calloc(t->size, sizeof(int));
    for (int i=0; i<t->size; ++i) {
        t->xsendc[i] = t->sendcounts[i] * t->chunksize_max;
        t->xsendd[i] = t->senddispls[i] * t->chunksize_max;
        t->xrecvc[i] = t->recvcounts[i] * t->chunksize_max;
        t->xrecvd[i] = t->recvdispls[i] * t->chunksize_max;
    }
    tock = diverge_mpi_wtime();
    mpi_vrb_printf( "TU-MPI prop: %.2fs for finalize\n", tock-tick );

    diverge_mpi_barrier();
    return t;
}

void tu_mpi_loop_destroy( tu_mpi_loop_t* t ) {
    diverge_mpi_barrier();
    free( t->kx_counts );
    free( t->kx_displs );
    fftw_destroy_plan( t->fft_forward );
    fftw_destroy_plan( t->fft_backward );
    fftw_destroy_plan( t->fft_transpose );
    fftw_free( t->ggp_buf );
    fftw_free( t->ggm_buf );
    free( t->c2fmap_local );
    free( t->kc_send );
    free( t->kc_recv );
    free( t->sendcounts );
    free( t->senddispls );
    free( t->recvcounts );
    free( t->recvdispls );
    free( t->xsendc );
    free( t->xsendd );
    free( t->xrecvc );
    free( t->xrecvd );
    free( t );
}

void tu_mpi_loop_calc( tu_mpi_loop_t* t, complex128_t* loop, index_t ntasks,
        index_t taskoff, bool rescale, bool pp, bool simple, bool subtract ) {
    double tick = 0, tock = 0;
    if (t->enable_timings) tick = diverge_mpi_wtime();
    if (ntasks == -1) {
        ntasks = simple ? POW2(t->n_orbff) : t->unique_mi_l_size;
        taskoff = 0;
    }
    const double prefac = -1.0/(2.0*M_PI);
    const index_t* dims = t->dims;
    const index_t nktot = dims[0]*dims[1]*dims[2];

    index_t stack_size = POW4(t->n_spin)*ntasks;
    ptask_t* stack = calloc( stack_size, sizeof(ptask_t) );
    alloc_print( stack_size * sizeof(ptask_t) );
    index_t idx = stack_size;
    for(index_t s3 = 0; s3<t->n_spin;++s3)
    for(index_t s2 = 0; s2<t->n_spin;++s2)
    for(index_t s4 = 0; s4<t->n_spin;++s4)
    for(index_t s1 = 0; s1<t->n_spin;++s1)
    for(index_t m = 0; m<ntasks;++m)
        stack[--idx] = (ptask_t){.s1=s1, .s2=s2, .s3=s3, .s4=s4, .m=m};
    ptask_t* tasks = calloc( t->chunksize_max, sizeof(ptask_t) );

    fftw_plan p = pp ? t->fft_forward : t->fft_backward;
    const index_t kx_start = t->kx_displ;
    const index_t kx_stop = t->kx_count + t->kx_displ;

    gf_complex_t* GF_fft_plus = (gf_complex_t*)(*t->p_GF_fft_plus);
    gf_complex_t* GF_fft_conj_plus = (gf_complex_t*)(*t->p_GF_fft_conj_plus);
    if (t->enable_timings) tock = diverge_mpi_wtime();
    t->timings[tu_mpi_loop_timing_prepare] += tock-tick;

    while (stack_size > 0) {
        tick = tock;
        index_t chunksize = MIN((index_t)stack_size, t->chunksize_max);
        for (index_t i=0; i<chunksize; ++i)
            tasks[i] = stack[--stack_size];

#define MY_BUFFER_GGM_IDX( i__, kff__ ) t->ggm_buf[ i__ * t->kx_count*dims[1]*dims[2] + kff__ ]
#define MY_BUFFER_GGP_IDX( i__, kff__ ) t->ggp_buf[ i__ * t->kx_count*dims[1]*dims[2] + kff__ ]
#define MY_BUFFER_GGM_IDX_POST( i__, kff__ ) t->ggm_buf[ kff__ * t->chunksize_max + i__ ]
#define MY_BUFFER_GGP_IDX_POST( i__, kff__ ) t->ggp_buf[ kff__ * t->chunksize_max + i__ ]

        // fill buffers
        #pragma omp parallel for num_threads(diverge_omp_num_threads()) schedule(dynamic)
        for (index_t i=0; i<chunksize; ++i) {
            const index_t s1 = tasks[i].s1,
                          s2 = tasks[i].s2,
                          s3 = tasks[i].s3,
                          s4 = tasks[i].s4,
                          m = tasks[i].m;

            index_t mo3 = simple ? ((taskoff+m)/t->n_orbff) : t->unique_mi_r[m+taskoff];
            index_t mo1 = simple ? ((taskoff+m)-mo3*t->n_orbff) : t->unique_mi_l[m+taskoff];
            const index_t o1 = t->mi_to_ofrom[mo1];
            const index_t ob1 = t->mi_to_oto[mo1];
            const index_t* off1 = t->mi_to_R+3*mo1;
            const index_t o3 = t->mi_to_ofrom[mo3];
            const index_t ob3 = t->mi_to_oto[mo3];
            const index_t* off3 = t->mi_to_R+3*mo3;

            for (index_t kffx=kx_start; kffx<kx_stop; ++kffx)
            for (index_t kffy=0; kffy<dims[1]; ++kffy)
            for (index_t kffz=0; kffz<dims[2]; ++kffz) {
                index_t kz = ((pp ? 1:-1)*kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = ((pp ? 1:-1)*kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = ((pp ? 1:-1)*kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx),
                        kff_local = kffz + dims[2] * (kffy + dims[1] * (kffx - kx_start));
                MY_BUFFER_GGM_IDX( i, kff_local ) = pp ?
                    GF_fft_conj_plus[kid + nktot*(o1+t->n_orb*(s1+t->n_spin*(o3+t->n_orb*s3)))] :
                    GF_fft_plus[kid + nktot*(o1+t->n_orb*(s1+t->n_spin*(o3+t->n_orb*s3)))];
                MY_BUFFER_GGM_IDX( i, kff_local ) *= pp ?
                    GF_fft_plus[kff+nktot*(ob1+t->n_orb*(s2+t->n_spin*(ob3+t->n_orb*s4)))] :
                    GF_fft_plus[kff+nktot*(ob3+t->n_orb*(s2+t->n_spin*(ob1+t->n_orb*s4)))];
            }

            for (index_t kffx=kx_start; kffx<kx_stop; ++kffx)
            for (index_t kffy=0; kffy<dims[1]; ++kffy)
            for (index_t kffz=0; kffz<dims[2]; ++kffz) {
                index_t kz = ((pp?1:-1)*kffz - off1[2] + off3[2]+4*dims[2])%dims[2];
                index_t ky = ((pp?1:-1)*kffy - off1[1] + off3[1]+4*dims[1])%dims[1];
                index_t kx = ((pp?1:-1)*kffx - off1[0] + off3[0]+4*dims[0])%dims[0];
                index_t kid = kz + dims[2] * (ky + dims[1] *kx);
                index_t kff = kffz + dims[2] * (kffy + dims[1] *kffx),
                        kff_local = kffz + dims[2] * (kffy + dims[1] * (kffx - kx_start));
                MY_BUFFER_GGP_IDX( i, kff_local ) = pp ?
                    GF_fft_plus[kid + nktot*(o1+t->n_orb*(s1+t->n_spin*(o3+t->n_orb*s3)))] :
                    GF_fft_conj_plus[kid + nktot*(o1+t->n_orb*(s1+t->n_spin*(o3+t->n_orb*s3)))];
                MY_BUFFER_GGP_IDX( i, kff_local ) *= pp ?
                    GF_fft_conj_plus[kff+nktot*(ob1+t->n_orb*(s2+t->n_spin*(ob3+t->n_orb*s4)))] :
                    GF_fft_conj_plus[kff+nktot*(ob3+t->n_orb*(s2+t->n_spin*(ob1+t->n_orb*s4)))];
            }

            for (index_t kffx=kx_start; kffx<kx_stop; ++kffx)
            for (index_t kffy=0; kffy<dims[1]; ++kffy)
            for (index_t kffz=0; kffz<dims[2]; ++kffz) {
                index_t kff_local = kffz + dims[2] * (kffy + dims[1] * (kffx - kx_start));
                MY_BUFFER_GGM_IDX( i, kff_local ) += MY_BUFFER_GGP_IDX( i, kff_local );
            }
        }
        if (t->enable_timings) tock = diverge_mpi_wtime();
        t->timings[tu_mpi_loop_timing_reorder] += tock-tick;

        tick = tock;
        fftw_execute_dft( t->fft_transpose, t->ggm_buf, t->ggp_buf );
        if (t->enable_timings) tock = diverge_mpi_wtime();
        t->timings[tu_mpi_loop_timing_transpose] += tock-tick;

        tick = tock;
        fftw_mpi_execute_dft( p, t->ggp_buf, t->ggm_buf );
        if (t->enable_timings) tock = diverge_mpi_wtime();
        t->timings[tu_mpi_loop_timing_fft] += tock-tick;

        tick = tock;
        #pragma omp parallel for num_threads(diverge_omp_num_threads()) schedule(dynamic)
        for (index_t i=0; i<chunksize; ++i) {
            const index_t s1 = tasks[i].s1,
                          s2 = tasks[i].s2,
                          s3 = tasks[i].s3,
                          s4 = tasks[i].s4,
                          m = tasks[i].m;

            index_t mo3 = simple ? ((taskoff+m)/t->n_orbff) : t->unique_mi_r[m+taskoff];
            index_t mo1 = simple ? ((taskoff+m)-mo3*t->n_orbff) : t->unique_mi_l[m+taskoff];
            for (index_t q=0; q<t->my_nk; ++q) {
                if (bitary_get(t->c2fmap_local, q)) {
                    index_t idx_to = pp ?
                        mo1+t->n_orbff*(s1+s2*t->n_spin+POW2(t->n_spin)*(mo3+t->n_orbff*(
                                        s3+s4*t->n_spin+POW2(t->n_spin)*q))) :
                        mo1+t->n_orbff*(s1+s4*t->n_spin+POW2(t->n_spin)*(mo3+t->n_orbff*(
                                        s3+s2*t->n_spin+POW2(t->n_spin)*q)));
                    index_t qff_local = kidx_to_local( t, t->c2fmap[q] );
                    if (subtract)
                        loop[idx_to] -= MY_BUFFER_GGM_IDX_POST( i, qff_local );
                    else
                        loop[idx_to] = MY_BUFFER_GGM_IDX_POST( i, qff_local );
                }
            }
        }
        if (t->enable_timings) tock = diverge_mpi_wtime();
        t->timings[tu_mpi_loop_timing_insert] += tock-tick;

        tick = tock;
        // communicate missing
        index_t elemsize = t->chunksize_max;
        complex128_t *send_buf = t->ggp_buf,
                     *result_buf = t->ggm_buf;

        // put from result_buf into send_buf
        for (index_t i=0; i<t->kc_send_count; ++i) {
            index_t kidx = kidx_to_local( t, t->kc_send[i].k );
            #ifndef NDEBUG
            if (find_rank( t, t->kc_send[i].k ) != t->rank) {
                printf( "rank of k index (%i) not me (%i) :(\n", find_rank( t, t->kc_send[i].k ), t->rank );
            }
            #endif
            for (index_t j=0; j<chunksize; ++j)
                send_buf[IDX2(i,j,elemsize)] = MY_BUFFER_GGM_IDX_POST( j, kidx );
        }
        if (t->enable_timings) tock = diverge_mpi_wtime();
        t->timings[tu_mpi_loop_timing_mpibufprep] += tock-tick;

        tick = tock;
        // send from send_buf to result_buf
        diverge_mpi_alltoallv_complex( send_buf, t->xsendc, t->xsendd, result_buf, t->xrecvc, t->xrecvd );
        if (t->enable_timings) tock = diverge_mpi_wtime();
        t->timings[tu_mpi_loop_timing_comm] += tock-tick;

        tick = tock;
        #pragma omp parallel for num_threads(diverge_omp_num_threads()) schedule(dynamic)
        for (index_t j=0; j<chunksize; ++j) {
            const index_t s1 = tasks[j].s1,
                          s2 = tasks[j].s2,
                          s3 = tasks[j].s3,
                          s4 = tasks[j].s4,
                          m = tasks[j].m;
            index_t mo3 = simple ? ((taskoff+m)/t->n_orbff) : t->unique_mi_r[m+taskoff];
            index_t mo1 = simple ? ((taskoff+m)-mo3*t->n_orbff) : t->unique_mi_l[m+taskoff];
            // put from result_buf into loop
            for (index_t i=0; i<t->kc_recv_count; ++i) {
                index_t q = t->kc_recv[i].q;
                if (!bitary_get(t->c2fmap_local, q)) {
                    index_t idx_to = pp ?
                        mo1+t->n_orbff*(s1+s2*t->n_spin+POW2(t->n_spin)*(mo3+t->n_orbff*(
                                        s3+s4*t->n_spin+POW2(t->n_spin)*q))) :
                        mo1+t->n_orbff*(s1+s4*t->n_spin+POW2(t->n_spin)*(mo3+t->n_orbff*(
                                        s3+s2*t->n_spin+POW2(t->n_spin)*q)));
                    if (subtract)
                        loop[idx_to] -= result_buf[IDX2(i,j,elemsize)];
                    else
                        loop[idx_to] = result_buf[IDX2(i,j,elemsize)];
                }
            }
        }
        if (t->enable_timings) tock = diverge_mpi_wtime();
        t->timings[tu_mpi_loop_timing_insert_missing] += tock-tick;
    }

    tick = tock;
    if (rescale) {
        if (!simple) reconstruct_from_reduced( t->tu_loop_handle, loop );
        scale_loop(loop, prefac, t->my_nk*POW4(t->n_spin)*POW2(t->n_orbff));
    }
    free(stack);
    free(tasks);
    if (t->enable_timings) tock = diverge_mpi_wtime();
    t->timings[tu_mpi_loop_timing_postproc] += tock-tick;
}

tu_mpi_gf_t* tu_mpi_gf_init( const index_t nk[3], const index_t nkf[3], index_t nb,
        gf_complex_t* gf_buf, diverge_model_t* model ) {
    tu_mpi_gf_t* t = calloc(1, sizeof(tu_mpi_gf_t));
    for (int i=0; i<3; ++i) t->dims[i] = (t->nk[i] = nk[i]) * (t->nkf[i] = nkf[i]);
    t->nb = nb;
    t->enable_timings = 1;
    t->comm = diverge_mpi_get_comm();
    t->rank = diverge_mpi_comm_rank();
    t->size = diverge_mpi_comm_size();
    t->model = model;
    index_t alloc = GF_FFTW(mpi_local_size_many)( 3, t->dims, POW2(t->nb),
            FFTW_MPI_DEFAULT_BLOCK, t->comm, &t->kx_count, &t->kx_displ );
    // make buffer large enough to also be used for transposition
    alloc = MAX( alloc, POW2(t->nb)*diverge_omp_num_threads() );
    t->fft_buf = GF_FFTW(alloc_complex)( alloc );
    alloc_print( sizeof(gf_complex_t)*alloc );
    t->fft = GF_FFTW(mpi_plan_many_dft)( 3, t->dims, POW2(t->nb),
            FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, t->fft_buf,
            t->fft_buf, t->comm, FFTW_BACKWARD, propagator_cpu_mpi_planning_patience );

    int displ = t->kx_displ * t->dims[1] * t->dims[2] * POW2(t->nb);
    int count = t->kx_count * t->dims[1] * t->dims[2] * POW2(t->nb);
    t->counts = calloc(t->size, sizeof(int));
    t->displs = calloc(t->size, sizeof(int));
    diverge_mpi_allgather_int( &count, t->counts, 1 );
    diverge_mpi_allgather_int( &displ, t->displs, 1 );

    t->transpose[0] = GF_plan_transpose( t->dims[0]*t->dims[1]*t->dims[2], POW2(t->nb), gf_buf );
    t->transpose[1] = GF_plan_transpose( t->dims[0]*t->dims[1]*t->dims[2], POW2(t->nb), gf_buf +
                                         t->dims[0]*t->dims[1]*t->dims[2]*POW2(t->nb) );
    return t;
}

void tu_mpi_gf_fft_and_transpose( tu_mpi_gf_t* t, gf_complex_t* gf_buf_ ) {
    double tick = 0, tock = 0;

    index_t kdimtot = t->dims[0] * t->dims[1] * t->dims[2];
    index_t scale = t->dims[1]*t->dims[2]*POW2(t->nb);
    gf_complex_t weight = 1./(double)kdimtot;
    gf_complex_t* gf_bufs[2] = { gf_buf_, gf_buf_ + kdimtot * POW2(t->nb) };

    index_t* shm_counts = calloc(2, sizeof(index_t));
    shm_counts[0] = kdimtot;
    index_t* shm_displs = shm_counts+1;
    int shm_size = 1, shm_rank = 0;
    const int use_shm = t->model->internals->greens_shared;

    if (use_shm) {
        free( shm_counts );
        shm_size = shared_malloc_size();
        shm_rank = shared_malloc_rank();
        shm_counts = diverge_distribute( kdimtot, shm_size );
        shm_displs = shm_counts + shm_size;
    }

    for (int gf_idx=0; gf_idx<2; ++gf_idx) {
        gf_complex_t* gf_buf = gf_bufs[gf_idx];
        if (t->enable_timings) tick = diverge_mpi_wtime();
        memcpy( t->fft_buf, gf_buf+t->kx_displ*scale, sizeof(gf_complex_t)*t->kx_count*scale );
        if (t->enable_timings) tock = diverge_mpi_wtime();
        t->timings[tu_mpi_gf_timing_trans] += tock-tick;

        if (t->enable_timings) tick = diverge_mpi_wtime();
        GF_FFTW(mpi_execute_dft)( t->fft, t->fft_buf, t->fft_buf );
        if (t->enable_timings) tock = diverge_mpi_wtime();
        t->timings[tu_mpi_gf_timing_fft] += tock-tick;

        if (t->enable_timings) tick = diverge_mpi_wtime();
        memcpy( gf_buf+t->kx_displ*scale, t->fft_buf, sizeof(gf_complex_t)*t->kx_count*scale );
        if (t->enable_timings) tock = diverge_mpi_wtime();
        t->timings[tu_mpi_gf_timing_trans] += tock-tick;

        if (t->enable_timings) tick = diverge_mpi_wtime();
        diverge_mpi_allgatherv_bytes( gf_buf, sizeof(gf_complex_t), t->counts, t->displs );
        if (t->enable_timings) tock = diverge_mpi_wtime();
        t->timings[tu_mpi_gf_timing_comm] += tock-tick;

        if (use_shm) shared_malloc_barrier();
        // we transpose o/b first
        if (t->enable_timings) tick = diverge_mpi_wtime();
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t krel=0; krel<shm_counts[shm_rank]; ++krel) {
            index_t k = krel + shm_displs[shm_rank];
            gf_complex_t* scratch = t->fft_buf + omp_get_thread_num()*POW2(t->nb);
            gf_complex_t* tgf_buf = gf_buf + k*POW2(t->nb);
            memcpy( scratch, tgf_buf, sizeof(gf_complex_t)*POW2(t->nb) );
            for (index_t b1=0; b1<t->nb; ++b1)
            for (index_t b2=0; b2<t->nb; ++b2)
                tgf_buf[IDX2(b1,b2,t->nb)] = scratch[IDX2(b2,b1,t->nb)] * weight;
        }
        if (use_shm) shared_malloc_barrier();
        if (t->enable_timings) tock = diverge_mpi_wtime();
        t->timings[tu_mpi_gf_timing_trans] += tock-tick;

        // and then ob/k
        if (t->enable_timings) tick = diverge_mpi_wtime();
        if (use_shm) {
            if (shared_exclusive_enter(gf_buf_))
                GF_FFTW(execute_dft)( t->transpose[gf_idx], gf_buf, gf_buf );
            shared_exclusive_wait(gf_buf_);
        } else {
            GF_FFTW(execute_dft)( t->transpose[gf_idx], gf_buf, gf_buf );
        }
        if (t->enable_timings) tock = diverge_mpi_wtime();
        t->timings[tu_mpi_gf_timing_ffttrans] += tock-tick;
    }

    free( shm_counts );
}

void tu_mpi_gf_destroy( tu_mpi_gf_t* t ) {
    GF_FFTW(destroy_plan)( t->fft );
    GF_FFTW(destroy_plan)( t->transpose[0] );
    GF_FFTW(destroy_plan)( t->transpose[1] );
    GF_FFTW(free)( t->fft_buf );
    free( t->counts );
    free( t->displs );
    free( t );
}

#else

tu_mpi_loop_t* tu_mpi_loop_init( const index_t nk[3], const index_t nkf[3],
        index_t chunksize_max, const index_t* c2fmap, index_t my_nk,
        index_t n_spin, index_t n_orb, index_t n_orbff,
        const index_t* unique_mi_l, index_t unique_mi_l_size,
        const index_t* unique_mi_r, const index_t* mi_to_ofrom,
        const index_t* mi_to_oto, const index_t* mi_to_R,
        complex128_t** GF_fft_plus, complex128_t** GF_fft_conj_plus,
        void* handle, diverge_model_t* model ) {
    (void)nk;
    (void)nkf;
    (void)chunksize_max;
    (void)c2fmap;
    (void)my_nk;
    (void)n_spin;
    (void)n_orb;
    (void)n_orbff;
    (void)unique_mi_l;
    (void)unique_mi_l_size;
    (void)unique_mi_r;
    (void)mi_to_ofrom;
    (void)mi_to_oto;
    (void)mi_to_R;
    (void)GF_fft_plus;
    (void)GF_fft_conj_plus;
    (void)handle;
    (void)model;
    mpi_err_printf( "new MPI loop not compiled in\n" );
    return NULL;
}

void tu_mpi_loop_calc( tu_mpi_loop_t* t, complex128_t* loop,
        index_t ntasks, index_t taskoff, bool rescale, bool pp, bool simple,
        bool subtract ) {
    (void)t;
    (void)loop;
    (void)ntasks;
    (void)taskoff;
    (void)rescale;
    (void)pp;
    (void)simple;
    (void)subtract;
    mpi_err_printf( "new MPI loop not compiled in\n" );
}

void tu_mpi_loop_destroy( tu_mpi_loop_t* t ) {
    (void)t;
    mpi_err_printf( "new MPI loop not compiled in\n" );
}

tu_mpi_gf_t* tu_mpi_gf_init( const index_t nk[3], const index_t nkf[3], index_t nb,
        gf_complex_t* gf_buf, diverge_model_t* model ) {
    (void)nk;
    (void)nkf;
    (void)nb;
    (void)gf_buf;
    (void)model;
    mpi_err_printf( "new MPI GF not compiled in\n" );
    return NULL;
}

void tu_mpi_gf_fft_and_transpose( tu_mpi_gf_t* t, gf_complex_t* gf_buf ) {
    (void)t;
    (void)gf_buf;
    mpi_err_printf( "new MPI GF not compiled in\n" );
}

void tu_mpi_gf_destroy( tu_mpi_gf_t* t ) {
    (void)t;
    mpi_err_printf( "new MPI GF not compiled in\n" );
}

#endif
