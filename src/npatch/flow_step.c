/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "flow_step.h"

#include "vertex.h"
#include "../diverge_model.h"
#include "symmetrize_vertex.h"
#include "../misc/mpi_functions.h"

#include <pthread.h>

#include "kernels_cpu.h"
#include "kernels_gpu.h"
#include "kernels_loop_gpu.h"

static bool h_flow_use_D1 = true,
            h_flow_use_D2 = true,
            h_flow_use_D3 = true;

static bool h_npatch_flow_step_ignore_nonSU2_exchange_symmetry = false;
static bool h_npatch_flow_step_resymmetrize_vertex = true;

static index_t start_at_idx;
static index_t stop_at_idx;
static index_t size_of_portion;

static index_t cpu_start_at_idx;
static index_t cpu_stop_at_idx;
static index_t cpu_size_of_portion;

static index_t gpu_start_at_idx;
static index_t gpu_stop_at_idx;
static index_t gpu_size_of_portion;

static int *displs = NULL, *counts = NULL;

static complex128_t* dPhi = NULL;
static complex128_t* dPhi_sym = NULL;
static bool static_variables_set = false;

static float npatch_flow_step_gpu_workshare = -1.;
static float npatch_flow_step_loop_gpu_workshare = -0.1;

static void npatch_flow_step_SU2_P    ( npatch_vertex_t* vertex, npatch_loop_t* loop, double* time_gpu, double time_gpus[], double* wtime_cpu );
static void npatch_flow_step_SU2_C    ( npatch_vertex_t* vertex, npatch_loop_t* loop, double* time_gpu, double time_gpus[], double* wtime_cpu );
static void npatch_flow_step_SU2_D1   ( npatch_vertex_t* vertex, npatch_loop_t* loop, double* time_gpu, double time_gpus[], double* wtime_cpu );
static void npatch_flow_step_SU2_D2   ( npatch_vertex_t* vertex, npatch_loop_t* loop, double* time_gpu, double time_gpus[], double* wtime_cpu );
static void npatch_flow_step_SU2_D3   ( npatch_vertex_t* vertex, npatch_loop_t* loop, double* time_gpu, double time_gpus[], double* wtime_cpu );
static void npatch_flow_step_nonSU2_P ( npatch_vertex_t* vertex, npatch_loop_t* loop, double* time_gpu, double time_gpus[], double* wtime_cpu );
static void npatch_flow_step_nonSU2_C ( npatch_vertex_t* vertex, npatch_loop_t* loop, double* time_gpu, double time_gpus[], double* wtime_cpu );
static void npatch_flow_step_nonSU2_D1( npatch_vertex_t* vertex, npatch_loop_t* loop, double* time_gpu, double time_gpus[], double* wtime_cpu );

#define FLOW_STEP_OF_CHAN( su, chan ) { \
    npatch_flow_step_##su##_##chan ( vertex, loop, &wtime_gpu, wtime_gpu_list, &wtime_cpu ); \
}

static void add_to_ary( complex128_t* ary, const complex128_t* add, index_t sz );
static void add_and_scale_to_ary( complex128_t* ary, const complex128_t* add, complex128_t scale, index_t sz );
static void flow_internal_setup( index_t sz, index_t sz_factor, float gpu_workshare );
static double find_max( const complex128_t* ary, index_t sz );

void npatch_flow_step_set_ignore_exchange_nonSU2( bool ignore ) {
    h_npatch_flow_step_ignore_nonSU2_exchange_symmetry = ignore;
}

void npatch_flow_step_set_vertex_resym( bool resym ) {
    h_npatch_flow_step_resymmetrize_vertex = resym;
}

npatch_flow_step_conf_t* npatch_flow_step_conf_init( bool SU2 ) {
    npatch_flow_step_conf_t* r = (npatch_flow_step_conf_t*)calloc(1,sizeof(npatch_flow_step_conf_t));
    r->SU2 = SU2;
    r->P = r->C = r->D = true;
    r->wtime_gpu = r->wtime_cpu = r->wtime_comm = r->wtime_max =
                   r->wtime_loop_cpu = r->wtime_loop_gpu = r->wtime_loop_comm = 0.0;
    for (int i=0; i<128; ++i) {
        r->wtime_gpu_list[i] = 0.0;
        r->wtime_loop_gpu_list[i] = 0.0;
        r->wtime_loop_gpu_gf_list[i] = 0.0;
    }
    r->wtime_full = 0.0;
    r->niter = 0;
    return r;
}

void npatch_setup_arch( const int* gpus, int ngpus, const int nblocks[3], float gpu_workshare, float loop_gpu_workshare ) {

    bool free_gpu_ary = false;
    if (gpus == NULL) {
        ngpus = npatch_flow_gpu_device_count();
        if (ngpus > 0) {
            mpi_vrb_printf("detected %i devices for npatch GPU flow\n", ngpus);
            free_gpu_ary = true;
            int* gp = (int*)malloc(sizeof(int)*ngpus);
            for (int i=0; i<ngpus; ++i) gp[i] = i;
            gpus = gp;
        } else {
            #ifdef USE_CUDA
            mpi_wrn_printf("automatic device search returned 0 devices. resetting GPU workshare to 0\n");
            #endif
            npatch_flow_step_gpu_workshare = 0.0;
            npatch_flow_step_loop_gpu_workshare = 0.0;
            return;
        }
    }

    int nbl[] = {1024, 1024, 1};
    if (nblocks == NULL) {
        nblocks = nbl;
        mpi_vrb_printf("automatic GPU block setup done with {%i,%i,%i}\n", nbl[0], nbl[1], nbl[2]);
    }

    npatch_flow_step_gpu_workshare = gpu_workshare;
    npatch_flow_step_loop_gpu_workshare = loop_gpu_workshare > -0.1 ? loop_gpu_workshare : gpu_workshare;

    if (npatch_flow_step_loop_gpu_workshare >= 1.0) npatch_flow_step_loop_gpu_workshare = 1.0;
    if (npatch_flow_step_loop_gpu_workshare <= 0.0) npatch_flow_step_loop_gpu_workshare = 0.0;
    if (npatch_flow_step_gpu_workshare >= 1.0) npatch_flow_step_gpu_workshare = 1.0;
    if (npatch_flow_step_gpu_workshare <= 0.0) npatch_flow_step_gpu_workshare = 0.0;

    npatch_flow_gpu_setup( gpus, ngpus, nblocks );

    if (free_gpu_ary)
        free((int*)gpus);
}

#define RESET_BUF( buf ) { \
    if (buf) free(buf); \
    buf = NULL; \
}
void npatch_setup_reset( void ) {
    npatch_flow_gpu_reset();
    static_variables_set = false;
    RESET_BUF( dPhi );
    RESET_BUF( dPhi_sym );
    RESET_BUF( counts );
    RESET_BUF( displs );
}

void npatch_setup_teardown( void ) {
    npatch_setup_reset();
    npatch_flow_gpu_teardown();
    npatch_loop_gpu_teardown();
}

void npatch_flow_step_set_sub_D( bool D1, bool D2, bool D3 ) {
    h_flow_use_D1 = D1;
    h_flow_use_D2 = D2;
    h_flow_use_D3 = D3;
    mpi_log_printf( "enabled D channel diagrams: RPA(%i) VC1(%i) VC2(%i)\n", D1, D2, D3 );
}

void npatch_setup_query( const char* property, void** data, index_t* psize ) {
    index_t size = 0;
    if        (!strcmp(property, "start_at_idx")) {
        *data = malloc( (size = sizeof(index_t)) );
        *(index_t*)(*data) = start_at_idx;
    } else if (!strcmp(property, "stop_at_idx")) {
        *data = malloc( (size = sizeof(index_t)) );
        *(index_t*)(*data) = stop_at_idx;
    } else if (!strcmp(property, "size_of_portion")) {
        *data = malloc( (size = sizeof(index_t)) );
        *(index_t*)(*data) = size_of_portion;
    } else if (!strcmp(property, "cpu_start_at_idx")) {
        *data = malloc( (size = sizeof(index_t)) );
        *(index_t*)(*data) = cpu_start_at_idx;
    } else if (!strcmp(property, "cpu_stop_at_idx")) {
        *data = malloc( (size = sizeof(index_t)) );
        *(index_t*)(*data) = cpu_stop_at_idx;
    } else if (!strcmp(property, "cpu_size_of_portion")) {
        *data = malloc( (size = sizeof(index_t)) );
        *(index_t*)(*data) = cpu_size_of_portion;
    } else if (!strcmp(property, "gpu_start_at_idx")) {
        *data = malloc( (size = sizeof(index_t)) );
        *(index_t*)(*data) = gpu_start_at_idx;
    } else if (!strcmp(property, "gpu_stop_at_idx")) {
        *data = malloc( (size = sizeof(index_t)) );
        *(index_t*)(*data) = gpu_stop_at_idx;
    } else if (!strcmp(property, "gpu_size_of_portion")) {
        *data = malloc( (size = sizeof(index_t)) );
        *(index_t*)(*data) = gpu_size_of_portion;
    } else if (!strcmp(property, "displs")) {
        *data = malloc( (size = sizeof(int) * diverge_mpi_comm_size()) );
        for (int r=0; r<diverge_mpi_comm_size(); ++r) ((int*)(*data))[r] = displs[r];
    } else if (!strcmp(property, "counts")) {
        *data = malloc( (size = sizeof(int) * diverge_mpi_comm_size()) );
        for (int r=0; r<diverge_mpi_comm_size(); ++r) ((int*)(*data))[r] = counts[r];
    } else if (!strcmp(property, "npatch_flow_step_gpu_workshare")) {
        *data = malloc( (size = sizeof(float)) );
        *(float*)(*data) = npatch_flow_step_gpu_workshare;
    } else if (!strcmp(property, "npatch_flow_step_loop_gpu_workshare")) {
        *data = malloc( (size = sizeof(float)) );
        *(float*)(*data) = npatch_flow_step_loop_gpu_workshare;
    } else if (!strcmp(property, "ngpus") ||
               !strcmp(property, "gpus") ||
               !strcmp(property, "nblocks") ||
               !strcmp(property, "nthreads")) {
        npatch_setup_query_gpu( property, data, psize );
        if (psize) size = *psize;
    } else {
        mpi_wrn_printf( "property '%s' not found.", property );
        *data = malloc( (size = sizeof(char)) );
        *(char*)(*data) = '\0';
    }
    if (psize) *psize = size;
}

void npatch_flow_step( npatch_flow_step_conf_t* config, npatch_vertex_t* vertex,
        npatch_loop_t* loop, double* Lambda, double dLambda, double* chanmax_ ) {

    double npatch_flow_step_tick = diverge_mpi_wtime();

    index_t np = loop->np;
    index_t nb = loop->nb;
    flow_internal_setup(np*np*np, nb*nb*nb*nb, npatch_flow_step_gpu_workshare);

    double chanmax[3] = {0,0,0};
    memset( vertex->dV, 0, sizeof(complex128_t)*vertex->size );

    complex128_t cLambda = 0.0;
    ((double*)(&cLambda))[1] = *Lambda;
    npatch_loop_gen( loop, cLambda );

    double wtime_gpu = 0.0;
    double wtime_cpu = 0.0;
    double wtime_comm = 0.0;
    double wtime_max = 0.0;
    double wtime_gpu_list[128];
    for (int i=0; i<128; ++i)
        wtime_gpu_list[i] = 0.0;

    double tick, tock;

    /* complex128_t* buf = malloc(sizeof(complex128_t)*vertex->size); */
    /* symmetrize_npatch_vertex( loop->dmod, vertex->V, loop->dpat->patches, np, dPhi_sym ); */
    /* memcpy( buf, vertex->V, sizeof(complex128_t)*vertex->size ); */
    /* symmetrize_npatch_vertex( loop->dmod, vertex->V, loop->dpat->patches, np, dPhi_sym ); */
    /* for (index_t i=0; i<vertex->size; ++i) */
    /*     if (cabs(buf[i] - vertex->V[i]) > 1e-7) { */
    /*         printf("problem 2/1 @%li\n", i); */
    /*         break; */
    /*     } */
    /* free(buf); */

    if (config->P) {
        memset( dPhi, 0, sizeof(complex128_t)*size_of_portion );
        if (config->SU2) FLOW_STEP_OF_CHAN( SU2, P )
        else             FLOW_STEP_OF_CHAN( nonSU2, P )
        tick = diverge_mpi_wtime();
        chanmax[0] = find_max( dPhi, size_of_portion );
        add_to_ary( vertex->dV + start_at_idx, dPhi, size_of_portion );
        tock = diverge_mpi_wtime();
        wtime_max += tock - tick;
    }
    if (config->C) {
        memset( dPhi, 0, sizeof(complex128_t)*size_of_portion );
        if (config->SU2) FLOW_STEP_OF_CHAN( SU2, C )
        else             FLOW_STEP_OF_CHAN( nonSU2, C )
        tick = diverge_mpi_wtime();
        chanmax[1] = find_max( dPhi, size_of_portion );
        add_to_ary( vertex->dV + start_at_idx, dPhi, size_of_portion );
        tock = diverge_mpi_wtime();
        wtime_max += tock - tick;
    }
    if (config->D) {
        memset( dPhi, 0, sizeof(complex128_t)*size_of_portion );
        if (config->SU2) {
            if (h_flow_use_D1) { FLOW_STEP_OF_CHAN( SU2, D1 ) }
            if (h_flow_use_D2) { FLOW_STEP_OF_CHAN( SU2, D2 ) }
            if (h_flow_use_D3) { FLOW_STEP_OF_CHAN( SU2, D3 ) }
        } else {
            if (h_flow_use_D1) { FLOW_STEP_OF_CHAN( nonSU2, D1 ) }
        }
        tick = diverge_mpi_wtime();
        chanmax[2] = find_max( dPhi, size_of_portion );
        add_to_ary( vertex->dV + start_at_idx, dPhi, size_of_portion );
        tock = diverge_mpi_wtime();
        wtime_max += tock - tick;
    }

    tick = diverge_mpi_wtime();
    diverge_mpi_allgatherv( vertex->dV, counts, displs );
    tock = diverge_mpi_wtime();
    wtime_comm += tock - tick;

    tick = diverge_mpi_wtime();
    if ((config->P || config->C || config->D) && (!config->SU2) &&
        (!h_npatch_flow_step_ignore_nonSU2_exchange_symmetry)) {
        // we don't want to destroy dV, that's why we symmetrize here and need one more buffer
        if (!dPhi_sym)
            dPhi_sym = (complex128_t*)calloc( vertex->size + loop->nk, sizeof(complex128_t) );
        npatch_flow_step_enforce_exchange( vertex->dV, dPhi_sym, loop->k3map, np, nb, 0.25 );
    }
    tock = diverge_mpi_wtime();
    wtime_max += tock - tick;

    tick = diverge_mpi_wtime();
    double nrm = dLambda;
    add_and_scale_to_ary( vertex->V, vertex->dV, nrm, vertex->size );
    for (int i=0; i<3; ++i) chanmax[i] *= fabs(nrm);
    tock = diverge_mpi_wtime();
    wtime_max += tock - tick;

    tick = diverge_mpi_wtime();
    if (h_npatch_flow_step_resymmetrize_vertex) {
        // same as above. we don't want to destroy dV
        if (!dPhi_sym)
            dPhi_sym = (complex128_t*)calloc( vertex->size + loop->nk, sizeof(complex128_t) );
        symmetrize_npatch_vertex( loop->dmod, vertex->V, loop->dpat->patches, np, dPhi_sym );
    }
    tock = diverge_mpi_wtime();
    wtime_max += tock - tick;

    tick = diverge_mpi_wtime();
    for (int i=0; i<3; ++i)
        chanmax[i] = diverge_mpi_max( chanmax+i );
    tock = diverge_mpi_wtime();
    wtime_comm += tock - tick;

    tick = diverge_mpi_wtime();
    double Lpmax = find_max( loop->Lp, loop->L_sz ),
           Lmmax = find_max( loop->Lm, loop->L_sz ),
           Vmax = find_max( vertex->V, vertex->size );
    tock = diverge_mpi_wtime();
    wtime_max += tock - tick;

    if (chanmax_ != NULL) {
        chanmax_[0] = Lpmax;
        chanmax_[1] = Lmmax;
        chanmax_[2] = chanmax[0];
        chanmax_[3] = chanmax[1];
        chanmax_[4] = chanmax[2];
        chanmax_[5] = Vmax;
    }

    *Lambda += dLambda;

    config->wtime_cpu += wtime_cpu;
    config->wtime_gpu += wtime_gpu;
    config->wtime_loop_cpu += loop->wtime_cpu;
    config->wtime_loop_gpu += loop->wtime_gpu;
    config->wtime_loop_comm += loop->wtime_comm;
    config->wtime_comm += wtime_comm;
    config->wtime_max += wtime_max;
    for (int i=0; i<128; ++i) {
        config->wtime_gpu_list[i] += wtime_gpu_list[i];
        config->wtime_loop_gpu_list[i] += loop->wtime_gpu_list[i];
        config->wtime_loop_gpu_gf_list[i] += loop->wtime_gpu_gf_list[i];
    }
    config->niter++;

    double npatch_flow_step_tock = diverge_mpi_wtime();
    config->wtime_full += npatch_flow_step_tock - npatch_flow_step_tick;

}

#define FLOW_STEP_OF_CHAN_IMPL( su, chan ) \
static void npatch_flow_step_##su##_##chan ( npatch_vertex_t* vertex, npatch_loop_t* loop, \
        double* wtime_gpu, double wtime_gpu_list[], double* wtime_cpu ) { \
    struct npatch_cpu_kernel_start_t cpu_kstart; \
    struct npatch_gpu_kernel_start_t gpu_kstart; \
\
    cpu_kstart.start_at = cpu_start_at_idx; \
    cpu_kstart.stop_at = cpu_stop_at_idx; \
    cpu_kstart.size = cpu_size_of_portion; \
    cpu_kstart.V = vertex; \
    cpu_kstart.L = loop; \
    cpu_kstart.dPhi = dPhi - start_at_idx; \
    cpu_kstart.wtime = 0; \
\
    gpu_kstart.start_at = gpu_start_at_idx; \
    gpu_kstart.stop_at = gpu_stop_at_idx; \
    gpu_kstart.size = gpu_size_of_portion; \
    gpu_kstart.V = vertex; \
    gpu_kstart.L = loop; \
    gpu_kstart.dPhi = dPhi - start_at_idx; \
    gpu_kstart.wtime = 0; \
    memset( gpu_kstart.wtime_gpus, 0, sizeof(gpu_kstart.wtime_gpus) ); \
\
    pthread_t cpu_thread, gpu_thread; \
    \
    pthread_create( &cpu_thread, NULL, npatch_cpu_flow_step_##su##_##chan , &cpu_kstart ); \
    if (gpu_kstart.size > 0) \
        pthread_create( &gpu_thread, NULL, npatch_gpu_flow_step_##su##_##chan , &gpu_kstart ); \
    pthread_join( cpu_thread, NULL ); \
    if (gpu_kstart.size > 0) \
        pthread_join( gpu_thread, NULL ); \
\
    *wtime_gpu += gpu_kstart.wtime; \
    *wtime_cpu += cpu_kstart.wtime; \
    for (int i=0; i<128; ++i) \
        wtime_gpu_list[i] += gpu_kstart.wtime_gpus[i]; \
}

FLOW_STEP_OF_CHAN_IMPL( SU2, P )
FLOW_STEP_OF_CHAN_IMPL( SU2, C )
FLOW_STEP_OF_CHAN_IMPL( SU2, D1 )
FLOW_STEP_OF_CHAN_IMPL( SU2, D2 )
FLOW_STEP_OF_CHAN_IMPL( SU2, D3 )
FLOW_STEP_OF_CHAN_IMPL( nonSU2, P )
FLOW_STEP_OF_CHAN_IMPL( nonSU2, C )
FLOW_STEP_OF_CHAN_IMPL( nonSU2, D1 )

static void flow_internal_setup(index_t sz, index_t sz_factor, float gpu_workshare) {
    if (static_variables_set) return;
    index_t* howmany = diverge_mpi_distribute( sz );
    int nranks = diverge_mpi_comm_size();
    displs = (int*)malloc(sizeof(index_t)*nranks);
    counts = (int*)malloc(sizeof(index_t)*nranks);
    displs[0] = 0;
    counts[0] = howmany[0];
    for (int c=1; c<nranks; ++c) {
        displs[c] = counts[c-1] + displs[c-1];
        counts[c] = howmany[c];
    }
    int r = diverge_mpi_comm_rank();
    start_at_idx = displs[r];
    stop_at_idx = displs[r] + counts[r];
    size_of_portion = stop_at_idx - start_at_idx;

    if (gpu_workshare > 1)
        gpu_workshare = 1;
    if (gpu_workshare < 0)
        gpu_workshare = 0;
    gpu_size_of_portion = size_of_portion * gpu_workshare;
    if (gpu_size_of_portion < 0)
        gpu_size_of_portion = 0;
    if (gpu_size_of_portion > size_of_portion)
        gpu_size_of_portion = size_of_portion;

    cpu_size_of_portion = size_of_portion - gpu_size_of_portion;
    cpu_start_at_idx = start_at_idx;
    cpu_stop_at_idx = start_at_idx + cpu_size_of_portion;

    gpu_start_at_idx = cpu_stop_at_idx;
    gpu_stop_at_idx = gpu_start_at_idx + gpu_size_of_portion;

        start_at_idx *= sz_factor;
    cpu_start_at_idx *= sz_factor;
    gpu_start_at_idx *= sz_factor;
        stop_at_idx *= sz_factor;
    cpu_stop_at_idx *= sz_factor;
    gpu_stop_at_idx *= sz_factor;
        size_of_portion *= sz_factor;
    cpu_size_of_portion *= sz_factor;
    gpu_size_of_portion *= sz_factor;
    for (int c=0; c<nranks; ++c) {
        displs[c] *= sz_factor;
        counts[c] *= sz_factor;
    }

    dPhi = (complex128_t*) calloc( size_of_portion, sizeof(complex128_t) );
    static_variables_set = true;
    free(howmany);
}

static void add_to_ary( complex128_t* ary, const complex128_t* add, index_t sz ) {
    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<sz; ++i) {
        ary[i] += add[i];
    }
}

static void add_and_scale_to_ary( complex128_t* ary, const complex128_t* add, complex128_t scale, index_t sz ) {
    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<sz; ++i) {
        ary[i] += scale * add[i];
    }
}

void npatch_flow_step_enforce_exchange( complex128_t* V, complex128_t* aux,
        const index_t* k3map, index_t np, index_t nb, double factor ) {

    bool free_aux = !aux;
    if (!aux) aux = (complex128_t*)malloc(sizeof(complex128_t)*POW3(np)*POW4(nb));

    #define RAVEL_V( k1, k2, k3, o1, o2, o3, o4 ) ((((((k1*np+k2)*np+k3)*nb+o1)*nb+o2)*nb+o3)*nb+o4)

    memcpy( aux, V, sizeof(complex128_t)*POW3(np)*POW4(nb) );
    #pragma omp parallel for collapse(7) num_threads(diverge_omp_num_threads())
    for (index_t k1=0; k1<np; ++k1)
    for (index_t k2=0; k2<np; ++k2)
    for (index_t k3=0; k3<np; ++k3)
    for (index_t o1=0; o1<nb; ++o1)
    for (index_t o2=0; o2<nb; ++o2)
    for (index_t o3=0; o3<nb; ++o3)
    for (index_t o4=0; o4<nb; ++o4) {
        index_t k4 = k3map[k1*np*np + k2*np + k3];

        index_t i1234 = RAVEL_V( k1, k2, k3, o1, o2, o3, o4 ),
                i1243 = RAVEL_V( k1, k2, k4, o1, o2, o4, o3 ),
                i2134 = RAVEL_V( k2, k1, k3, o2, o1, o3, o4 ),
                i2143 = RAVEL_V( k2, k1, k4, o2, o1, o4, o3 );
        V[i1234] = factor *  (aux[i1234] - aux[i1243] + aux[i2143] - aux[i2134]);
    }

    if (free_aux) free(aux);
}

static double find_max( const complex128_t* ary, index_t sz ) {
    double ary_max = 0.0;
    #pragma omp parallel for reduction(max:ary_max) num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<sz; ++i)
        ary_max = MAX(cabs(ary[i]), ary_max);
    return ary_max;
}

