/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_hack.h"
#include "diverge_model.h"
#include "diverge_internals_struct.h"
#include "diverge_Eigen3.hpp"

#include "tu/diverge_interface.hpp"
#include "npatch/vertex.h"
#include "npatch/flow_step.h"
#include "grid/vertex_memory.hpp"
#include "misc/kmeans_lloyd.h"
#include "misc/shared_mem.h"
#include "tu/propagator_cpu_mpi.h"
#include "tu/propagator_cpu_mpi_fft.h"

#include <string>
#include <set>

using std::set;
using std::string;
using std::stringstream;

static int npatch_gpus[128];
static int npatch_nblocks[3];
static int npatch_gpu_weights[128];
static int npatch_loop_gpu_weights[128];

extern double npatch_average_factor;
extern int autofine_ngroups;
extern double autofine_alpha;
extern double autofine_beta;
extern double autofine_gamma;

static int parse_vector_int( int* vec, int len, const char* str );
static greensfunc_op_t diverge_shared_gf(const diverge_model_t* model, complex128_t Lambda, gf_complex_t* buf);

static set<string> hacking_keys;

#define CHECKKEY( ckey ) \
    hacking_keys.insert( ckey ); \
    if (key == ckey) key_set = true; \
    if (key == ckey)

void diverge_model_hack( diverge_model_t* m, const char* key_, const char* val ) {

    #define INTERNALS (m->internals)
    #define TU_INTERNALS ((tu_data_t*)m->internals->tu_data)

    string key;
    if (key_) key = key_;
    else key = "";
    bool key_set = false;

    CHECKKEY("tu_cpu_offload") {
        TU_INTERNALS->offload_cpu = atof(val);
        if(TU_INTERNALS->offload_cpu > 1.0)
            TU_INTERNALS->offload_cpu = 0.0;
    }
    CHECKKEY("tu_percent_mem_GPU_blocked") {
        TU_INTERNALS->percent_mem_GPU_blocked = atof(val);
        if(TU_INTERNALS->percent_mem_GPU_blocked > 1.0)
            TU_INTERNALS->percent_mem_GPU_blocked = 0.9;
    }
    CHECKKEY("tu_use_cuda_mpi_loop") {
        TU_INTERNALS->use_cuda_mpi_loop = true;
    }
    CHECKKEY("tu_selfenergy_flow") {
        TU_INTERNALS->tu_selfenergy_flow = (bool)atoi(val);
    }
    CHECKKEY("tu_mpifft_chunksize") {
        TU_INTERNALS->mpi_fft_chunksize = atoi(val);
    }
    CHECKKEY("tu_which_fft_greens") {
        TU_INTERNALS->which_fft_greens = (tu_fft_algorithm_t)atoi(val);
    }
    CHECKKEY("tu_which_fft_simple") {
        TU_INTERNALS->which_fft_simple = (tu_fft_algorithm_t)atoi(val);
    }
    CHECKKEY("tu_which_fft_red") {
        TU_INTERNALS->which_fft_red = (tu_fft_algorithm_t)atoi(val);
    }
    CHECKKEY("tu_extra_propagator_timings") {
        TU_INTERNALS->propagator_extra_cpu_timings = atoi(val);
    }
    CHECKKEY("npatch_gpu_workshare") {
        INTERNALS->patch_gpu_workshare = atof(val);
    }
    CHECKKEY("npatch_gpu_loop_workshare") {
        INTERNALS->patch_gpu_loop_workshare = atof(val);
    }
    CHECKKEY("npatch_gpus") {
        parse_vector_int( npatch_gpus, sizeof(npatch_gpus)/sizeof(npatch_gpus[0]), val );
        INTERNALS->patch_gpus = npatch_gpus;
    }
    CHECKKEY("npatch_nblocks") {
        parse_vector_int( npatch_nblocks, sizeof(npatch_nblocks)/sizeof(npatch_nblocks[0]), val );
        INTERNALS->patch_nblocks = npatch_nblocks;
    }
    CHECKKEY("npatch_ngpus") {
        INTERNALS->patch_ngpus = atoi(val);
    }
    CHECKKEY("npatch_loop_gpu_weights") {
        parse_vector_int( npatch_loop_gpu_weights, sizeof(npatch_loop_gpu_weights)/sizeof(npatch_loop_gpu_weights[0]), val );
        INTERNALS->patch_loop_gpu_weights = npatch_loop_gpu_weights;
    }
    CHECKKEY("npatch_gpu_weights") {
        parse_vector_int( npatch_gpu_weights, sizeof(npatch_gpu_weights)/sizeof(npatch_gpu_weights[0]), val );
        INTERNALS->patch_gpu_weights = npatch_gpu_weights;
    }
    CHECKKEY("npatch_ignore_exchange_nonSU2") {
        npatch_flow_step_set_ignore_exchange_nonSU2( atoi(val) );
    }
    CHECKKEY("npatch_set_sub_D") {
        int ivec[3] = {0};
        parse_vector_int( ivec, 3, val );
        npatch_flow_step_set_sub_D( ivec[0], ivec[1], ivec[2] );
    }
    CHECKKEY("grid_set_sub_D") {
        int ivec[3] = {0};
        parse_vector_int( ivec, 3, val );
        grid::VertexMemory* V = (grid::VertexMemory*)INTERNALS->grid_vertex;
        if (V) V->set_sub_D( ivec );
    }
    CHECKKEY("npatch_vertex_resym") {
        npatch_flow_step_set_vertex_resym( atoi(val) );
    }
    CHECKKEY("npatch_average_factor") {
        npatch_average_factor = atof(val);
    }
    CHECKKEY("autofine_ngroups") {
        autofine_ngroups = atoi(val);
    }
    CHECKKEY("autofine_alpha") {
        autofine_alpha = atof(val);
    }
    CHECKKEY("autofine_beta") {
        autofine_beta = atof(val);
    }
    CHECKKEY("autofine_gamma") {
        autofine_gamma = atof(val);
    }
    CHECKKEY("dkm_random_seed") {
        dkm_set_use_random_seed( atoi(val) );
    }
    CHECKKEY("model_free_hamiltonian") {
        shared_free(INTERNALS->ham);
        INTERNALS->ham = NULL;
        if (!strcmp(val, "eigen")) {
            shared_free(INTERNALS->E);
            INTERNALS->E = NULL;
            shared_free(INTERNALS->U);
            INTERNALS->U = NULL;
        }
    }
    CHECKKEY("model_shared_gf") {
        if (m->gfill != &diverge_greensfunc_generator_default) {
            mpi_wrn_printf( "shared GF buffer for custom GF generator must be taken care of manually\n" );
        }
        m->gfill = &diverge_shared_gf;
        free(INTERNALS->greens);
        INTERNALS->greens = (gf_complex_t*)shared_malloc(sizeof(gf_complex_t) *
                kdimtot(m->nk, m->nkf) * POW2(m->n_orb*m->n_spin) * 2);
        INTERNALS->greens_shared = 1;
    }
    CHECKKEY("model_internals_eigensolve_ignore") {
        INTERNALS->ignore_eigensolver_in_common = 1;
    }
    CHECKKEY("model_internals_eigensolve_svd_algo") {
        INTERNALS->use_batched_svd_algo_in_common = 1;
        if (atoi(val) == -1)
            INTERNALS->use_batched_svd_algo_in_common = -1;
    }

    CHECKKEY("fftw_mpi_plan_patience") {
        // either set with 'G' (GX) to address fftw_mpi_plan_patience (for
        // GPU/MPI) or without (X) to address
        // propagator_cpu_mpi_planning_patience (for MPI)
        unsigned* addr = &propagator_cpu_mpi_planning_patience;
        if (strlen(val) > 0) {
            if (toupper(val[0]) == 'G') addr = &fftw_mpi_plan_patience;
            val++;
        }
        switch (atoi(val)) {
            case 0: *addr = FFTW_ESTIMATE; break;
            case 1: *addr = FFTW_MEASURE; break;
            case 2: *addr = FFTW_PATIENT; break;
            case 3: *addr = FFTW_EXHAUSTIVE; break;
            default: break;
        }
    }

    if (key_) {
        if (key_set) {
            mpi_vrb_printf("hacking: %s=%s\n", key_, val);
        } else {
            mpi_wrn_printf("hacking: %s not a valid key\n", key_);
        }
    }
}

void diverge_model_print_hacks( void ) {
    diverge_model_hack( NULL, NULL, NULL );
    mpi_scc_printf( "available hacking keys:\n" );
    for (string s: hacking_keys)
        mpi_scc_printf( "'%s'\n", s.c_str() );
}

static int parse_vector_int( int* vec, int len, const char* str_ ) {
    string input(str_);
    stringstream input_stream(input);

    string parsed;
    vector<string> components;
    while (std::getline( input_stream, parsed, ',' ))
        components.push_back( parsed );

    int parsed_len = components.size();
    if (parsed_len > len) {
        components.resize(len);
        parsed_len = len;
    }

    for (int i=0; i<parsed_len; ++i)
        vec[i] = atoi(components[i].c_str());
    for (int j=parsed_len; j<len; ++j)
        vec[j] = 0;
    return parsed_len;
}

static greensfunc_op_t diverge_shared_gf(const diverge_model_t* model, complex128_t Lambda, gf_complex_t* buf) {
    const index_t nkf = kdimtot(model->nk, model->nkf);
    const index_t nb = model->n_spin * model->n_orb;
    const complex128_t* U = (const complex128_t*) model->internals->U;
    const double* E = model->internals->E;

    // calc displs/counts
    index_t kcount = 0, kdispl = 0;
    {
        int size = shared_malloc_size(),
            rank = shared_malloc_rank();
        index_t* counts = (index_t*)calloc(size*2, sizeof(index_t));
        index_t* displs = counts + size;
        for (index_t k=0; k<nkf; ++k) counts[k%size]++;
        for (int c=1; c<size; ++c) displs[c] = displs[c-1] + counts[c-1];
        kcount = counts[rank];
        kdispl = displs[rank];
        free( counts );
    }

    // do loop
    shared_malloc_barrier();
    for (int sign=0; sign<2; ++sign) {
        complex128_t L = sign ? Lambda : std::conj(Lambda);
        #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
        for (index_t krel=0; krel<kcount; ++krel)
        for (index_t o=0; o<nb; ++o)
        for (index_t p=0; p<nb; ++p) {
            index_t k = krel + kdispl;
            complex128_t tmp = 0.0;
            for (index_t b=0; b<nb; ++b)
                tmp += U[IDX3(k,b,o,nb,nb)] * std::conj(U[IDX3(k,b,p,nb,nb)]) / (L - E[IDX2(k,b,nb)]);
            buf[IDX4(sign, k, o, p, nkf, nb, nb)] = tmp;
        }
    }
    shared_malloc_barrier();
    return greensfunc_op_cpu;
}

