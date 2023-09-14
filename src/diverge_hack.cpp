/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_hack.h"
#include "diverge_model.h"
#include "diverge_internals_struct.h"
#include "diverge_Eigen3.hpp"

#include "tu/diverge_interface.hpp"
#include "npatch/vertex.h"
#include "npatch/flow_step.h"
#include "grid/vertex_memory.hpp"
#include "misc/kmeans_lloyd.h"

#include <string>
#include <set>

using std::set;
using std::string;
using std::stringstream;

static int npatch_gpus[128];
static int npatch_nblocks[3];
static int npatch_gpu_weights[128];
static int npatch_loop_gpu_weights[128];

static int parse_vector_int( int* vec, int len, const char* str );

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
    }
    CHECKKEY("tu_selfenergy_flow") {
        TU_INTERNALS->tu_selfenergy_flow = (bool)atoi(val);
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
        int ivec[3];
        parse_vector_int( ivec, 3, val );
        npatch_flow_step_set_sub_D( ivec[0], ivec[1], ivec[2] );
    }
    CHECKKEY("npatch_vertex_resym") {
        npatch_flow_step_set_vertex_resym( atoi(val) );
    }
    CHECKKEY("dkm_random_seed") {
        dkm_set_use_random_seed( atoi(val) );
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
