#include "flow_step.h"
#include "vertex_memory.h"
#include "loops.h"
#include "../misc/batched_gemms.h"

static void grid_flow_step_set_dv_zero( grid_flow_step_t* s );
static double grid_flow_step_get_dv( grid_flow_step_t* s, char channel, double Lambda, double dLambda );
static double grid_flow_step_add_dv_to_v( grid_flow_step_t* s );
static double kfnormgen( const double* kfweights, index_t nkf );

struct grid_flow_step_t {
    grid_vertex_t* V;
    grid_loop_t* L;
    bool P, C, D;
    bool SU2;
    double kfnorm;

    complex128_t* LC;
    complex128_t* LP;

    double norms[5];
    grid_timing_t t[6];
    double maxs[6];
    index_t num_calls;
};

grid_flow_step_t* grid_flow_step_init( grid_vertex_t* V, grid_loop_t* L,
        bool P/*= true*/, bool C/*= true*/, bool D/*= true*/ ) {
    grid_flow_step_t* s = calloc(1, sizeof*s);

    s->P = P;
    s->C = C;
    s->D = D;
    s->SU2 = V->SU2;
    s->V = V;
    s->L = L;
    s->kfnorm = kfnormgen( L->kfweights, L->nkf );

    for (int i=0; i<5; ++i) s->norms[i] = 1.0/(double)s->L->nk/s->kfnorm;

    if (s->SU2) {
        s->norms[2] *= -2.0;
        int nmod = 0;
        if (!V->sub_D[0]) s->norms[2] = 0, nmod++;
        if (!V->sub_D[1]) s->norms[3] = 0, nmod++;
        if (!V->sub_D[2]) s->norms[4] = 0, nmod++;
        if (nmod)
            mpi_log_printf( "enabled D channel diagrams: RPA(%i) VC1(%i) VC2(%i)\n",
                    V->sub_D[0], V->sub_D[1], V->sub_D[2] );
    } else {
        if (s->C != s->D)
            mpi_err_printf( "non-SU2 only uses D *and* C\n" );
        s->norms[0] *=  0.5;
        s->norms[2] *= -1.0;
    }

    strcpy( s->t[0].nam, "reorder" );
    strcpy( s->t[1].nam, "loop" );
    strcpy( s->t[2].nam, "loop-symm" );
    strcpy( s->t[3].nam, "loop-gemm" );
    strcpy( s->t[4].nam, "contract" );
    strcpy( s->t[5].nam, "max" );
    return s;
}

void grid_flow_step_free( grid_flow_step_t* s ) { free(s); }
grid_timing_t* grid_flow_step_timings( grid_flow_step_t* s ) { return s->t; }
index_t grid_flow_step_num_calls( const grid_flow_step_t* s ) { return s->num_calls; }

const double* grid_flow_step_euler( grid_flow_step_t* s, double Lambda, double dLambda ) {
    grid_flow_step_set_dv_zero( s );
    memset( s->maxs, 0, sizeof s->maxs );
    if (s->SU2) {
        if (s->P) s->maxs[0] = grid_flow_step_get_dv( s, 'P', Lambda, dLambda );
        if (s->C) s->maxs[1] = grid_flow_step_get_dv( s, 'C', Lambda, dLambda );
        if (s->D) s->maxs[2] = grid_flow_step_get_dv( s, 'D', Lambda, dLambda );
    } else {
        if (s->P) s->maxs[0] = grid_flow_step_get_dv( s, 'P', Lambda, dLambda );
        if (s->D) s->maxs[1] = grid_flow_step_get_dv( s, 'D', Lambda, dLambda );
        s->maxs[2] = s->maxs[1]; // C == D
    }
    s->maxs[3] = grid_flow_step_add_dv_to_v( s );
    s->maxs[4] = s->L->max_P;
    s->maxs[5] = s->L->max_C;
    s->num_calls++;

    for (int i=0; i<6; ++i)
        diverge_mpi_allreduce_double_max( MPI_IN_PLACE, &s->maxs[i], 1 );

    return s->maxs;
}

static double kfnormgen( const double* kfweights, index_t nkf ) {
    double norm = 0.0;
    for (index_t kf=0; kf<nkf; ++kf) norm += kfweights[kf];
    return norm;
}

static void grid_flow_step_set_dv_zero( grid_flow_step_t* s ) {
    ZERO_OUT( s->V->vertex_buffers[1], s->V->size );
    s->LP = s->LC = NULL;
}

static double grid_flow_step_get_dv( grid_flow_step_t* s, char channel, double Lambda, double dLambda ) {
    index_t qsize = s->L->qsize;
    index_t nk = s->L->nk;
    index_t nb = s->L->nb;
    complex128_t* v = s->V->vertex_buffers[0];
    complex128_t* left = s->V->vertex_buffers[2];
    complex128_t* right = s->V->vertex_buffers[3];
    complex128_t* dv = s->V->vertex_buffers[1];
    double vmax = 0.0;
    index_t Vsize = s->V->size;
    double *t_reorder = &s->t[0].t,
           *t_loop = &s->t[1].t,
           *t_loop_symm = &s->t[2].t,
           *t_loop_gemm = &s->t[3].t,
           *t_contract = &s->t[4].t,
           *t_max = &s->t[5].t;
    if (channel == 'P') {
        TICK_TOCK( *t_loop,      if (!s->LP) s->LP = grid_loop_calc(s->L, 'P', Lambda); );
        TICK_TOCK( *t_reorder,   grid_vertex_project_name( s->V, v, left,  "D_to_P", false, 1.0 ) );
        TICK_TOCK( *t_reorder,   grid_vertex_project_name( s->V, v, right, "D_to_P", false, 1.0 ) );
        TICK_TOCK( *t_loop_gemm, batched_gemm_vertex_loop( left, s->LP, qsize, nk, nb ) );
        TICK_TOCK( *t_contract,  batched_gemm_overlapping( left, right, right, nk*nb*nb, 1.0, 0.0, qsize ) );
        TICK_TOCK( *t_max,       vmax = absmax( right, Vsize ) * s->norms[0] * dLambda );
        TICK_TOCK( *t_reorder,   grid_vertex_project_name( s->V, right, dv, "P_to_D", true, s->norms[0]*dLambda ) );
    }
    if (channel == 'C') {
        if (!s->SU2) return 0;
        TICK_TOCK( *t_loop,      if (!s->LC) s->LC = grid_loop_calc(s->L, 'C', Lambda); );
        TICK_TOCK( *t_reorder,   grid_vertex_project_name( s->V, v, left,  "D_to_C", false, 1.0 ) );
        TICK_TOCK( *t_reorder,   grid_vertex_project_name( s->V, v, right, "D_to_C", false, 1.0 ) );
        TICK_TOCK( *t_loop_gemm, batched_gemm_vertex_loop( left, s->LC, qsize, nk, nb ) );
        TICK_TOCK( *t_contract,  batched_gemm_overlapping( left, right, right, nk*nb*nb, 1.0, 0.0, qsize ) );
        TICK_TOCK( *t_max,       vmax = absmax( right, Vsize ) * s->norms[1] * dLambda );
        TICK_TOCK( *t_reorder,   grid_vertex_project_name( s->V, right, dv, "C_to_D", true, s->norms[1]*dLambda ) );
    }
    if (channel == 'D') {
        TICK_TOCK( *t_loop,      if (!s->LC) s->LC = grid_loop_calc(s->L, 'C', Lambda); );
        if (s->SU2) {
            // 'right' is the result

            // vertex corrections
            // left vertex C channel
            TICK_TOCK( *t_reorder,   grid_vertex_project_name( s->V, v, left, "D_to_C", false, 1.0 ) );
            TICK_TOCK( *t_loop_gemm, batched_gemm_vertex_loop( left, s->LC, qsize, nk, nb ) );
            // make 'v' the right vertex and 'right' the result
            TICK_TOCK( *t_contract,  batched_gemm( left, v, right, nk*nb*nb, s->norms[3]*dLambda, 0.0, qsize ) );
            // right vertex C channel
            // make 'left' right
            TICK_TOCK( *t_reorder,   grid_vertex_project_name( s->V, v, left, "D_to_C", false, 1.0 ) );
            // make 'send_pack' left (*after* reordering call)
            TICK_TOCK( *t_reorder,   grid_vertex_copy( s->V, v, s->V->send_pack, false, 1.0 ) );
            TICK_TOCK( *t_loop_gemm, batched_gemm_vertex_loop( s->V->send_pack, s->LC, qsize, nk, nb ) );
            // write result in 'right'
            TICK_TOCK( *t_contract,  batched_gemm( s->V->send_pack, left, right, nk*nb*nb, s->norms[4]*dLambda, 1.0, qsize ) );

            // rpa
            // use 'send_pack' as left vertex *with* loop and 'v' as right
            // write result to 'right'
            TICK_TOCK( *t_contract,  batched_gemm( s->V->send_pack, v, right, nk*nb*nb, s->norms[2]*dLambda, 1.0, qsize ) );
            // get max of result ('right')
            TICK_TOCK( *t_max,       vmax = absmax( right, Vsize ) );
            // add result ('right') to dv
            TICK_TOCK( *t_reorder,   grid_vertex_copy( s->V, right, dv, true, 1.0 ) );
        } else {
            TICK_TOCK( *t_reorder,   grid_vertex_copy( s->V, v, left, false, 1.0 ) );
            TICK_TOCK( *t_loop_gemm, batched_gemm_vertex_loop( left, s->LC, qsize, nk, nb ) );
            TICK_TOCK( *t_contract,  batched_gemm( left, v, right, nk*nb*nb, 1.0, 0.0, qsize ) );
            TICK_TOCK( *t_max,       vmax = absmax( right, Vsize ) * s->norms[2] * dLambda );
            TICK_TOCK( *t_reorder,   grid_vertex_copy( s->V, right, dv, true, s->norms[2]*dLambda ) );
            TICK_TOCK( *t_reorder,   grid_vertex_project_name( s->V, right, dv, "D_swap_3_and_4", true, s->norms[1]*dLambda ) );
        }
    }
    *t_loop_symm = s->L->t_sym;
    return vmax;
}

static double grid_flow_step_add_dv_to_v( grid_flow_step_t* s ) {
    complex128_t* vertex = s->V->vertex_buffers[0];
    const complex128_t* d_vertex = s->V->vertex_buffers[1];
    const index_t Vsize = s->V->size;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<Vsize; ++i) vertex[i] += d_vertex[i];
    double Vmax = absmax( vertex, Vsize );
    return Vmax;
}
