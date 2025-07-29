/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "loop.h"
#include "flow_step.h"

#include "../misc/mpi_functions.h"
#include "../diverge_model.h"
#include "../diverge_internals_struct.h"

#include <pthread.h>
#include "kernels_helpers.h"
#include "kernels_loop_cpu.h"
#include "kernels_loop_gpu.h"

#define POW2(x) ((x)*(x))
#define POW3(x) ((x)*(x)*(x))
#define POW4(x) ((x)*(x)*(x)*(x))

static void loop_workshare_setup(npatch_loop_workshare_t* ws, index_t sz);
static void loop_workshare_scale(npatch_loop_workshare_t* ws, index_t scale);

struct npatch_loop_workshare_t {
    index_t start_at_idx;
    index_t stop_at_idx;
    index_t size_of_portion;

    index_t cpu_start_at_idx;
    index_t cpu_stop_at_idx;
    index_t cpu_size_of_portion;

    index_t gpu_start_at_idx;
    index_t gpu_stop_at_idx;
    index_t gpu_size_of_portion;

    int *displs;
    int *counts;
};

npatch_loop_t* npatch_loop_init( diverge_model_t* dmodel ) {
    npatch_loop_t* r = (npatch_loop_t*)malloc(sizeof(npatch_loop_t));

    r->dmod = dmodel;
    r->dint = dmodel->internals;
    r->dpat = dmodel->patching;

    r->nk = r->dint->patch_nk;
    r->nb = r->dint->patch_nb;

    r->G_sz = r->nk*r->nb*r->nb;
    r->Gp = r->dint->greens;
    r->Gm = r->dint->greens + r->G_sz;

    r->np = r->dpat->n_patches;
    r->p_idx = r->dpat->patches;
    r->p_cnt = r->dpat->p_count;
    r->p_dis = r->dpat->p_displ;
    r->p_map = r->dpat->p_map;
    r->f_wgh = r->dpat->p_weights;
    r->nk_flat[0] = r->dmod->nk[0];
    r->nk_flat[1] = r->dmod->nk[1];
    r->nk_flat[2] = 1;
    r->k3map = r->dint->patch_k3map;
    r->wgh = r->dpat->weights;

    r->E_real = r->dint->E;
    r->E = NULL;
    r->U = r->dint->U;

    r->L_sz = POW3(r->np) * POW4(r->nb);
    r->Lp = malloc(sizeof(complex128_t)*r->L_sz);
    r->Lm = malloc(sizeof(complex128_t)*r->L_sz);

    r->workshare = (npatch_loop_workshare_t*)malloc(sizeof(npatch_loop_workshare_t));
    loop_workshare_setup(r->workshare, r->np);
    loop_workshare_scale(r->workshare, POW4(r->nb) * POW2(r->np));

    r->is_proj_dispatcher = 0;
    r->is_proj = 0;
    r->projLoop = NULL;

    if (r->dmod->gproj != NULL) {
        // set up infrastructure for projection dispatch
        npatch_loop_t* p = (npatch_loop_t*)malloc(sizeof(npatch_loop_t));
        memcpy( p, r, sizeof(npatch_loop_t) );

        r->is_proj_dispatcher = 1;
        p->is_proj = 1;
        r->projLoop = p;

        // and allocate memory on the dispatched loop
        p->Lp = (complex128_t*)calloc(p->L_sz, sizeof(complex128_t));
        p->Lm = (complex128_t*)calloc(p->L_sz, sizeof(complex128_t));
    }
    return r;
}

void npatch_loop_free( npatch_loop_t* loop ) {
    if (loop->is_proj_dispatcher) {
        npatch_loop_free( loop->projLoop );
    } else {
        free(loop->workshare->displs);
        free(loop->workshare->counts);
        free(loop->workshare);
    }
    free( loop->Lp );
    free( loop->Lm );
    free( loop );
}

void npatch_loop_gen( npatch_loop_t* loop, complex128_t Lambda ) {

    // recurse one level if we are the dispatcher!
    if (loop->is_proj_dispatcher)
        npatch_loop_gen( loop->projLoop, Lambda );

    bool loop_use_gpu_gf = false;

    greensfunc_generator_t ggen = loop->is_proj ? loop->dmod->gproj : loop->dmod->gfill;
    switch ((*ggen)( loop->dmod, Lambda, loop->dint->greens )) {
        case greensfunc_op_gpu:
            loop_use_gpu_gf = (loop->workshare->cpu_start_at_idx == loop->workshare->cpu_stop_at_idx);
            if (!loop_use_gpu_gf)
                mpi_err_printf("bogus GF. GPU only valid with loop gpu work 100%% (i.e. USE_CUDA=1)\n");
            if (loop->is_proj)
                mpi_err_printf("cannot use GPU GF on projected loop\n");
            if (loop->is_proj_dispatcher)
                mpi_err_printf("cannot use GPU GF on projected loop dispatcher\n");
            break;
        case greensfunc_op_cpu:
        default:
            break;
    }

    complex128_t* restrict Lp = loop->Lp;
    complex128_t* restrict Lm = loop->Lm;
    memset( Lp, 0, sizeof(complex128_t)*loop->L_sz );
    memset( Lm, 0, sizeof(complex128_t)*loop->L_sz );

    npatch_loop_kernel_info_t cpu, gpu;

    cpu.L = loop;
    cpu.start = loop->workshare->cpu_start_at_idx;
    cpu.stop = loop->workshare->cpu_stop_at_idx;
    cpu.device = -1;
    cpu.needs_gpu_copy = false;
    cpu.calc_gf_on_gpu = loop_use_gpu_gf;
    cpu.Lambda = Lambda;
    cpu.wtime = 0.0;
    memset( cpu.wtime_gpus, 0, sizeof(cpu.wtime_gpus) );
    memset( cpu.wtime_gfs, 0, sizeof(cpu.wtime_gfs) );

    gpu.L = loop;
    gpu.start = loop->workshare->gpu_start_at_idx;
    gpu.stop = loop->workshare->gpu_stop_at_idx;
    gpu.device = -1;
    gpu.needs_gpu_copy = false;
    gpu.calc_gf_on_gpu = loop_use_gpu_gf;
    gpu.Lambda = Lambda;
    gpu.wtime = 0.0;
    memset( gpu.wtime_gpus, 0, sizeof(gpu.wtime_gpus) );
    memset( gpu.wtime_gfs, 0, sizeof(gpu.wtime_gfs) );

    pthread_t cpu_t, gpu_t;

    pthread_create( &cpu_t, NULL, npatch_loop_gen_cpu, &cpu );
    if (loop->workshare->gpu_size_of_portion > 0)
        pthread_create( &gpu_t, NULL, npatch_loop_gen_gpu, &gpu );

    pthread_join( cpu_t, NULL );
    if (loop->workshare->gpu_size_of_portion > 0)
        pthread_join( gpu_t, NULL );

    double tick = diverge_mpi_wtime();
    diverge_mpi_allgatherv( Lp, loop->workshare->counts, loop->workshare->displs );
    diverge_mpi_allgatherv( Lm, loop->workshare->counts, loop->workshare->displs );
    double tock = diverge_mpi_wtime();
    loop->wtime_comm = tock - tick;

    loop->wtime_gpu = gpu.wtime;
    loop->wtime_cpu = cpu.wtime;
    for (int i=0; i<128; ++i) {
        loop->wtime_gpu_list[i] = gpu.wtime_gpus[i];
        loop->wtime_gpu_gf_list[i] = gpu.wtime_gfs[i];
    }

    // do the accumulation if we are the dispatcher
    if (loop->is_proj_dispatcher) {
        tick = diverge_mpi_wtime();
        complex128_t *Lm_proj = loop->projLoop->Lm,
                     *Lp_proj = loop->projLoop->Lp,
                     *Lm = loop->Lm,
                     *Lp = loop->Lp;
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<loop->L_sz; ++i)
            Lp[i] -= Lp_proj[i];
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t i=0; i<loop->L_sz; ++i)
            Lm[i] -= Lm_proj[i];
        tock = diverge_mpi_wtime();

        // and update the timings a bit
        npatch_loop_t* dloop = loop->projLoop;
        loop->wtime_gpu += dloop->wtime_gpu;
        loop->wtime_cpu += dloop->wtime_cpu;
        loop->wtime_comm += dloop->wtime_comm + tock-tick;
        for (int i=0; i<128; ++i) {
            loop->wtime_gpu_list[i] += dloop->wtime_gpu_list[i];
            loop->wtime_gpu_gf_list[i] += dloop->wtime_gpu_gf_list[i];
        }
    }
}

static void loop_workshare_setup(npatch_loop_workshare_t* ws, index_t sz) {
    float* p_gpu_workshare;
    index_t p_gpu_workshare_sz;
    npatch_setup_query( "npatch_flow_step_loop_gpu_workshare", (void**)&p_gpu_workshare, &p_gpu_workshare_sz );

    index_t* howmany = diverge_mpi_distribute( sz );
    int nranks = diverge_mpi_comm_size();
    ws->displs = (int*)malloc(sizeof(index_t)*nranks);
    ws->counts = (int*)malloc(sizeof(index_t)*nranks);
    ws->displs[0] = 0;
    ws->counts[0] = howmany[0];
    for (int c=1; c<nranks; ++c) {
        ws->displs[c] = ws->counts[c-1] + ws->displs[c-1];
        ws->counts[c] = howmany[c];
    }
    int r = diverge_mpi_comm_rank();
    ws->start_at_idx = ws->displs[r];
    ws->stop_at_idx = ws->displs[r] + ws->counts[r];
    ws->size_of_portion = ws->stop_at_idx - ws->start_at_idx;

    if (*p_gpu_workshare > 1)
        *p_gpu_workshare = 1;
    if (*p_gpu_workshare < 0)
        *p_gpu_workshare = 0;
    ws->gpu_size_of_portion = ws->size_of_portion * (*p_gpu_workshare);
    if (ws->gpu_size_of_portion < 0)
        ws->gpu_size_of_portion = 0;
    if (ws->gpu_size_of_portion > ws->size_of_portion)
        ws->gpu_size_of_portion = ws->size_of_portion;

    ws->cpu_size_of_portion = ws->size_of_portion - ws->gpu_size_of_portion;
    ws->cpu_start_at_idx = ws->start_at_idx;
    ws->cpu_stop_at_idx = ws->start_at_idx + ws->cpu_size_of_portion;

    ws->gpu_start_at_idx = ws->cpu_stop_at_idx;
    ws->gpu_stop_at_idx = ws->gpu_start_at_idx + ws->gpu_size_of_portion;
    free(howmany);
    free(p_gpu_workshare);
}

static void loop_workshare_scale(npatch_loop_workshare_t* ws, index_t scale) {
    ws->start_at_idx *= scale;
    ws->stop_at_idx *= scale;
    ws->size_of_portion *= scale;

    ws->cpu_start_at_idx *= scale;
    ws->cpu_stop_at_idx *= scale;
    ws->cpu_size_of_portion *= scale;

    ws->gpu_start_at_idx *= scale;
    ws->gpu_stop_at_idx *= scale;
    ws->gpu_size_of_portion *= scale;

    for (int r=0; r<diverge_mpi_comm_size(); ++r) {
        ws->displs[r] *= scale;
        ws->counts[r] *= scale;
    }

}

