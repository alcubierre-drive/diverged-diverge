/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "kernels_cpu.h"
#include "kernels_helpers.h"
#include "../diverge_model.h"
#include "../misc/mpi_functions.h"
#include <omp.h>

#define DO_PRAGMA(x) _Pragma (#x)
#define ESC_PARENS(...) __VA_ARGS__

#define INVOKE_UNRAVEL_V(...) UNRAVEL_V(__VA_ARGS__)
#define INVOKE_RAVEL_V(...) RAVEL_V(__VA_ARGS__)
#define INVOKE_RAVEL_L3(...) RAVEL_L3(__VA_ARGS__)

#define K3ADD( k1, k2, k3 ) k3map[(k1*np+k2)*np+k3]

#define CONTRACT_P_Vl ESC_PARENS(k1, k2, k, o1, o2, p1, p2)
#define CONTRACT_P_Vr ESC_PARENS(k, K3ADD( k1, k2, k ), k3, p3, p4, o3, o4)
#define CONTRACT_P_L3 ESC_PARENS(k, k1, k2, p1, p2, p3, p4)
#define CONTRACT_P_nm const short o1 = n, o2 = m;
#define CONTRACT_P_op const short o3 = o, o4 = p;

#define CONTRACT_C_Vl ESC_PARENS(k1, K3ADD( k2, k, k3 ), k, o1, p2, p1, o4)
#define CONTRACT_C_Vr ESC_PARENS(k, k2, k3,                 p3, o2, o3, p4)
#define CONTRACT_C_L3 ESC_PARENS(k, k2, k3, p1, p4, p3, p2)
#define CONTRACT_C_nm const short o1 = n, o4 = m;
#define CONTRACT_C_op const short o2 = o, o3 = p;

#define CONTRACT_D1_Vl ESC_PARENS(k1, K3ADD( k, k3, k1 ), k3, o1, p1, o3, p2)
#define CONTRACT_D1_Vr ESC_PARENS(k, k2, K3ADD( k, k3, k1 ), p4, o2, p3, o4)
#define CONTRACT_D1_L3 ESC_PARENS(k, k3, k1, p2, p3, p4, p1)
#define CONTRACT_D1_nm const short o1 = n, o3 = m;
#define CONTRACT_D1_op const short o2 = o, o4 = p;

#define CONTRACT_D2_Vl ESC_PARENS(k1, K3ADD( k, k3, k1 ), k3, o1, p1, o3, p2)
#define CONTRACT_D2_Vr ESC_PARENS(k, k2, K3ADD( k1, k2, k3 ), p4, o2, o4, p3)
#define CONTRACT_D2_L3 ESC_PARENS(k, k3, k1, p2, p3, p4, p1)
#define CONTRACT_D2_nm const short o1 = n, o3 = m;
#define CONTRACT_D2_op const short o2 = o, o4 = p;

#define CONTRACT_D3_Vl ESC_PARENS(k1, K3ADD( k, k3, k1 ), k, o1, p2, p1, o3)
#define CONTRACT_D3_Vr ESC_PARENS(k, k2, K3ADD( k, k3, k1 ), p3, o2, p4, o4)
#define CONTRACT_D3_L3 ESC_PARENS(k, k3, k1, p1, p4, p3, p2)
#define CONTRACT_D3_nm const short o1 = n, o3 = m;
#define CONTRACT_D3_op const short o2 = o, o4 = p;

#define PREPARE_IMPL_FULL \
    struct npatch_cpu_kernel_start_t* params = (struct npatch_cpu_kernel_start_t*) _params; \
    const complex128_t* restrict V = params->V->V; \
    const index_t np = params->L->np; \
    const index_t nb = params->L->nb; \
    const index_t start = params->start_at; \
    const index_t stop = params->stop_at; \
    const index_t* restrict k3map = params->L->k3map; \
    const double* restrict W = params->L->wgh; \
    complex128_t* dPhi = params->dPhi;

#define MAGIC_SPIN_VALUE (-1234567890)

#define IMPL_FULL( su, chan, factor ) \
void* npatch_cpu_flow_step_##su##_##chan ( void* _params ) { \
    double tick = diverge_mpi_wtime(); \
    PREPARE_IMPL_FULL \
    complex128_t * restrict L = !strcmp( #chan ,"P") ? params->L->Lm : params->L->Lp; \
    \
    const double fac = (factor == MAGIC_SPIN_VALUE) ? \
            (1./fabs((double)(params->L->dmod->n_spin))) : \
            factor; \
    \
    const index_t Nb4 = nb*nb*nb*nb; \
    complex128_t* mem_per_thread = (complex128_t*)malloc(sizeof(complex128_t)*omp_get_max_threads()*Nb4); \
    \
    const index_t k123start = start/Nb4, \
                  k123stop = stop/Nb4; \
    \
    DO_PRAGMA(omp parallel for schedule(static) collapse(2) num_threads(diverge_omp_num_threads())) \
    for (index_t phi_k_idx=k123start; phi_k_idx<k123stop; ++phi_k_idx) \
    for (index_t k=0; k<np; ++k) { \
        \
        const double nrm = W[k] * fac; \
        \
        complex128_t* cache = mem_per_thread + Nb4*omp_get_thread_num(); \
        memset(cache,0,sizeof(complex128_t)*Nb4); \
        \
        UNRAVEL_K3(phi_k_idx, k1, k2, k3) \
        \
        for (short n=0; n<nb; ++n) \
        for (short m=0; m<nb; ++m) \
        for (short p3=0; p3<nb; ++p3) \
        for (short p4=0; p4<nb; ++p4) { \
            CONTRACT_##chan##_nm \
            const index_t oidx = RAVEL_O4( n, m, p3, p4 ); \
            for (short p1=0; p1<nb; ++p1) \
            for (short p2=0; p2<nb; ++p2) { \
                const index_t lidx = INVOKE_RAVEL_L3( CONTRACT_##chan##_L3 ); \
                const index_t vidx = INVOKE_RAVEL_V( CONTRACT_##chan##_Vl ); \
                cache[oidx] += V[vidx] * L[lidx]; \
            } \
            cache[oidx] *= nrm; \
        } \
        \
        for (short n=0; n<nb; ++n) \
        for (short m=0; m<nb; ++m) \
        for (short o=0; o<nb; ++o) \
        for (short p=0; p<nb; ++p) { \
            CONTRACT_##chan##_nm \
            CONTRACT_##chan##_op \
            const index_t pidx = RAVEL_V( k1, k2, k3, o1, o2, o3, o4 ); \
            for (short p3=0; p3<nb; ++p3) \
            for (short p4=0; p4<nb; ++p4) { \
                const index_t oidx = RAVEL_O4( n, m, p3, p4 ); \
                const index_t vidx = INVOKE_RAVEL_V( CONTRACT_##chan##_Vr ); \
                complex128_t_atomic_add_byval( dPhi + pidx, cache[oidx] * V[vidx] ); \
            } \
        } \
    } \
    free(mem_per_thread); \
    double tock = diverge_mpi_wtime(); \
    params->wtime = tock - tick; \
    return NULL; \
}

IMPL_FULL( SU2, P, 1.0 )
IMPL_FULL( SU2, C, 1.0 )
IMPL_FULL( SU2, D1, -2.0 )
IMPL_FULL( SU2, D2, 1.0 )
IMPL_FULL( SU2, D3, 1.0 )

IMPL_FULL( nonSU2, P, MAGIC_SPIN_VALUE )
IMPL_FULL( nonSU2, C, 1.0 )
IMPL_FULL( nonSU2, D1, -1.0 )

