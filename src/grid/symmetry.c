#include "symmetry.h"
#include "../diverge_model.h"
#include "../diverge_momentum_gen.h"
#include "../diverge_internals_struct.h"
#include "../misc/mpi_functions.h"
#include "../misc/mergesort.h"
#define LINALG_NO_LAPACKE
#include "../misc/linalg.h"
LINALG_IMPLEMENT

// some private types and functions //
typedef struct {
    int current_rank;
    int target_rank;
    index_t current_index;
    index_t target_index;
    complex128_t phase;
} grid_symm_op_t;

typedef struct {
    index_t* reorder_before_send; // _loop_size
    complex128_t* phase_before_send; // _loop_size
    index_t *send_counts, *send_displs, *recv_counts, *recv_displs;
    index_t* reorder_after_recv; // _loop_size

    const complex128_t* orbital_matrix; // (nb, nb), must not be freed
    index_t nb;
    char chan;
} grid_symm_comm_t;

// free the memory of a communicator
static void grid_symm_comm_free( grid_symm_comm_t S );

typedef struct {
    int idx_pre_symm;
    int idx_post_symm;
    int R1, R2, R3;
} grid_symm_prop_t;

struct grid_symm_transform_t {
    index_t nk, nb;
    index_t nsym;

    const double* kmesh; // (nk, 3)
    const double* rs_sym; // (nsym, 3,3)
    const complex128_t* orb_sym; // (nsym, nb,nb)

    grid_symm_prop_t* symmprops; // (nsym, nb)
    index_t* k_symmaps; // (nsym, nk)

    const index_t *k1m2, *k1p2;

    claM3d_t lattice, rlattice;

    // communicator
    index_t *qdispls, *qcounts, qdispl, qcount;
    index_t loop_size;
    int myrank, nranks;
    grid_symm_op_t *symops_P, *symops_C; // (nsym, loop_size)

    // comm structs/loop
    grid_symm_comm_t *comms_P, *comms_C;
    complex128_t *send_buf, *recv_buf, *loop_buf;
};

// first part of initialization, symm maps are generated
static grid_symm_transform_t* grid_symm_transform_init_transform( diverge_model_t* model );

// symm maps to MPI loopmaps transform
static void grid_symm_transform_get_trafo( grid_symm_transform_t* s, index_t sym, index_t qdispl,
        index_t qcount, complex128_t* phasesP /*(qcount, nk, POW4(nb))*/,
        complex128_t* phasesC, index_t* mapP /*(qcount, nk, POW4(nb))*/, index_t* mapC );

// communication infrastructure for MPI
static void grid_symm_transform_init_communicator( grid_symm_transform_t* s,
        index_t qdispl, index_t qcount );

// communicator setup for specific symmetry
static grid_symm_comm_t grid_symm_transform_comm( grid_symm_transform_t* s, index_t symm, char chan );

// loop symmetrization infrastructure
static void grid_symm_transform_init_loop( grid_symm_transform_t* s );

// apply a single symmetry to loop
static void grid_symm_transform_loop_single( grid_symm_transform_t* s, complex128_t* loop_in,
        complex128_t* loop_out, index_t sym, char chan );



static void gen_grid_k_symmmap( const double* kmesh, index_t nk,
        claM3d_t reciprocal, claM3d_t symm, index_t* result/*(nk)*/ ) {
    for (index_t i=0; i<nk; ++i) {
        claV3d_t ksym = claM3d_matvec(symm, claV3d_map(kmesh+3*i));
        for (index_t j=0; j<nk; ++j) {
            claV3d_t ksym_ij = claV3d_sub(ksym, claV3d_map(kmesh+3*j));
            for (int g1=-2; g1<=2; ++g1)
            for (int g2=-2; g2<=2; ++g2)
            for (int g3=-2; g3<=2; ++g3) {
                claV3d_t ksym_ij_g1g2 = claV3d_add(ksym_ij, claM3d_vecmat((claV3d_t){{g1,g2,g3}}, reciprocal));
                if (claV3d_norm2(ksym_ij_g1g2) <= 1.e-7) {
                    result[i] = j;
                    goto endloop_k_symmap;
                }
            }
        }
endloop_k_symmap: {}
    }
}

static void gen_grid_symm_prop( const claV3d_t* basis, index_t nb, claM3d_t lattice,
    claM3d_t sym, grid_symm_prop_t* result/*(nb)*/ ) {
    for (index_t i=0; i<nb; ++i) {
        result[i].idx_pre_symm = i;
        for (index_t j=0; j<nb; ++j) {
            for (int r1=-2; r1<=2; ++r1)
            for (int r2=-2; r2<=2; ++r2)
            for (int r3=-2; r3<=2; ++r3) {
                claV3d_t A = claM3d_vecmat((claV3d_t){{r1,r2,r3}}, lattice);
                if (claV3d_norm2(claV3d_sub( claV3d_add(claM3d_matvec(sym, basis[i]), A), basis[j] )) <= 1.e-7) {
                    result[i].idx_post_symm = j;
                    result[i].R1 = r1; result[i].R2 = r2; result[i].R3 = r3;
                    goto endloop_grid_symm_prop_t;
                }
            }
        }
endloop_grid_symm_prop_t: {}
    }
}

static grid_symm_transform_t* grid_symm_transform_init_transform( diverge_model_t* model ) {
    grid_symm_transform_t* s = calloc(1, sizeof*s);
    s->nk = kdim(model->nk);
    s->nb = model->n_orb * model->n_spin;
    s->nsym = model->n_sym;

    s->kmesh = model->internals->kmesh;
    s->rs_sym = model->rs_symmetries[0][0];
    s->orb_sym = model->orb_symmetries;
    s->k1m2 = model->internals->grid_k1m2;
    s->k1p2 = model->internals->grid_k1p2;

    s->symmprops = calloc(s->nsym*s->nb, sizeof*s->symmprops);
    s->k_symmaps = calloc(s->nsym*s->nk, sizeof*s->k_symmaps);

    // we have to pad the basis with spin
    claV3d_t basis[s->nb];
    for (index_t ss=0; ss<model->n_spin; ++ss)
    for (index_t oo=0; oo<model->n_orb; ++oo)
        basis[IDX2(ss,oo,model->n_orb)] = claV3d_map(model->positions[oo]);

    s->lattice = claM3d_map(model->lattice[0]);
    diverge_model_generate_mom_basis(s->lattice.m[0], s->rlattice.m[0]);

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t sym=0; sym<s->nsym; ++sym) {
        gen_grid_symm_prop( basis, s->nb, s->lattice,
                claM3d_map(model->rs_symmetries[sym][0]), s->symmprops + sym*s->nb );
        gen_grid_k_symmmap( s->kmesh, s->nk, s->rlattice,
                claM3d_map(model->rs_symmetries[sym][0]), s->k_symmaps + sym*s->nk );
    }

    // _reset_bond_symmprops due to diverge style of having the indices encoded
    // in the matrices
    for (index_t symb=0; symb<s->nsym * s->nb; ++symb)
        s->symmprops[symb].idx_post_symm = s->symmprops[symb].idx_pre_symm;

    return s;
}

static void grid_symm_transform_get_trafo( grid_symm_transform_t* s, index_t sym, index_t qdispl,
        index_t qcount, complex128_t* phasesP /*(qcount, nk, POW4(nb))*/,
        complex128_t* phasesC, index_t* mapP /*(qcount, nk, POW4(nb))*/, index_t* mapC ) {

    index_t LoopSize = qcount*s->nk*POW4(s->nb);
    const index_t dim[6] = {qcount, s->nk, s->nb, s->nb, s->nb, s->nb};

    // P
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t Loopindex=0; Loopindex<LoopSize; ++Loopindex) {
        index_t _qko[6]; unravel(Loopindex, dim, 6, _qko);
        const index_t q = _qko[0] + qdispl, k = _qko[1],
                      o1 = _qko[2], o2 = _qko[3], o3 = _qko[4], o4 = _qko[5];

        const index_t k1 = k,
                      k2 = s->k1m2[q*s->nk+k];

        const index_t k1_T = s->k_symmaps[IDX2(sym,k1,s->nk)],
                      o1_T = s->symmprops[IDX2(sym,o1,s->nb)].idx_post_symm,
                      o3_T = s->symmprops[IDX2(sym,o3,s->nb)].idx_post_symm,
                      k2_T = s->k_symmaps[IDX2(sym,k2,s->nk)],
                      o2_T = s->symmprops[IDX2(sym,o2,s->nb)].idx_post_symm,
                      o4_T = s->symmprops[IDX2(sym,o4,s->nb)].idx_post_symm;

        const index_t k_T = k1_T,
                      q_T = s->k1p2[k2_T*s->nk+k1_T];

        const claV3d_t k1vT = claV3d_map(s->kmesh + k1_T*3),
                       k2vT = claV3d_map(s->kmesh + k2_T*3);
        const claV3d_t So1 = claM3d_vecmat( (claV3d_t){{ s->symmprops[IDX2(sym,o1,s->nb)].R1,
                                                         s->symmprops[IDX2(sym,o1,s->nb)].R2,
                                                         s->symmprops[IDX2(sym,o1,s->nb)].R3, }}, s->lattice );
        const claV3d_t So2 = claM3d_vecmat( (claV3d_t){{ s->symmprops[IDX2(sym,o2,s->nb)].R1,
                                                         s->symmprops[IDX2(sym,o2,s->nb)].R2,
                                                         s->symmprops[IDX2(sym,o2,s->nb)].R3, }}, s->lattice );
        const claV3d_t So3 = claM3d_vecmat( (claV3d_t){{ s->symmprops[IDX2(sym,o3,s->nb)].R1,
                                                         s->symmprops[IDX2(sym,o3,s->nb)].R2,
                                                         s->symmprops[IDX2(sym,o3,s->nb)].R3, }}, s->lattice );
        const claV3d_t So4 = claM3d_vecmat( (claV3d_t){{ s->symmprops[IDX2(sym,o4,s->nb)].R1,
                                                         s->symmprops[IDX2(sym,o4,s->nb)].R2,
                                                         s->symmprops[IDX2(sym,o4,s->nb)].R3, }}, s->lattice );

        complex128_t phase_first =  cexp( -I * claV3d_dot(k1vT, claV3d_sub(So1, So3)) );
        complex128_t phase_second = cexp( -I * claV3d_dot(k2vT, claV3d_sub(So2, So4)) );

        mapP[Loopindex] = IDX6( q_T,k_T,o1_T,o2_T,o3_T,o4_T, s->nk,s->nb,s->nb,s->nb,s->nb );
        phasesP[Loopindex] = phase_first * phase_second;
    }

    // C
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t Loopindex=0; Loopindex<LoopSize; ++Loopindex) {
        index_t _qko[6]; unravel(Loopindex, dim, 6, _qko);
        const index_t q = _qko[0] + qdispl, k = _qko[1],
                      O1 = _qko[2], O2 = _qko[3], O3 = _qko[4], O4 = _qko[5];
        const index_t o1 = O2, o2 = O3, o3 = O4, o4 = O1; // reorder

        const index_t k1 = k,
                      k2 = s->k1m2[k*s->nk+q];

        const index_t k1_T = s->k_symmaps[IDX2(sym,k1,s->nk)],
                      o1_T = s->symmprops[IDX2(sym,o1,s->nb)].idx_post_symm,
                      o3_T = s->symmprops[IDX2(sym,o3,s->nb)].idx_post_symm,
                      k2_T = s->k_symmaps[IDX2(sym,k2,s->nk)],
                      o2_T = s->symmprops[IDX2(sym,o2,s->nb)].idx_post_symm,
                      o4_T = s->symmprops[IDX2(sym,o4,s->nb)].idx_post_symm;

        const index_t k_T = k1_T,
                      q_T = s->k1m2[k1_T*s->nk+k2_T];

        const claV3d_t k1vT = claV3d_map(s->kmesh + k1_T*3),
                       k2vT = claV3d_map(s->kmesh + k2_T*3);
        const claV3d_t So1 = claM3d_vecmat( (claV3d_t){{ s->symmprops[IDX2(sym,o1,s->nb)].R1,
                                                         s->symmprops[IDX2(sym,o1,s->nb)].R2,
                                                         s->symmprops[IDX2(sym,o1,s->nb)].R3, }}, s->lattice );
        const claV3d_t So2 = claM3d_vecmat( (claV3d_t){{ s->symmprops[IDX2(sym,o2,s->nb)].R1,
                                                         s->symmprops[IDX2(sym,o2,s->nb)].R2,
                                                         s->symmprops[IDX2(sym,o2,s->nb)].R3, }}, s->lattice );
        const claV3d_t So3 = claM3d_vecmat( (claV3d_t){{ s->symmprops[IDX2(sym,o3,s->nb)].R1,
                                                         s->symmprops[IDX2(sym,o3,s->nb)].R2,
                                                         s->symmprops[IDX2(sym,o3,s->nb)].R3, }}, s->lattice );
        const claV3d_t So4 = claM3d_vecmat( (claV3d_t){{ s->symmprops[IDX2(sym,o4,s->nb)].R1,
                                                         s->symmprops[IDX2(sym,o4,s->nb)].R2,
                                                         s->symmprops[IDX2(sym,o4,s->nb)].R3, }}, s->lattice );

        complex128_t phase_first =  cexp( -I * claV3d_dot(k1vT, claV3d_sub(So1,So3)) );
        complex128_t phase_second = cexp( -I * claV3d_dot(k2vT, claV3d_sub(So2,So4)) );

        const index_t O1p = o4_T, O2p = o1_T, O3p = o2_T, O4p = o3_T; // reorder

        mapC[Loopindex] = IDX6( q_T,k_T,O1p,O2p,O3p,O4p, s->nk,s->nb,s->nb,s->nb,s->nb );
        phasesC[Loopindex] = phase_first * phase_second;
    }
}

void grid_symm_transform_free( grid_symm_transform_t* s ) {
    free( s->symmprops );
    free( s->k_symmaps );

    free( s->qdispls );
    free( s->qcounts );
    free( s->symops_P );
    free( s->symops_C );

    for (index_t i=0; i<s->nsym; ++i)
        grid_symm_comm_free( s->comms_P[i] ),
        grid_symm_comm_free( s->comms_C[i] );
    free( s->comms_P );
    free( s->comms_C );
    free( s->send_buf );
    free( s->recv_buf );
    free( s->loop_buf );

    free( s );
}

static inline int rank_from_index( index_t* qdispls, index_t* qcounts, int nranks, index_t q ) {
    for (int i=0; i<nranks; ++i)
        if (q >= qdispls[i] && q < (qdispls[i]+qcounts[i])) return i;
    return -1;
}

#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) ( (x).target_rank - (y).target_rank )
MERGESORT_IMPL( mergesort_symop, grid_symm_op_t )

static void grid_symm_transform_init_communicator( grid_symm_transform_t* s,
        index_t qdispl, index_t qcount ) {
    s->qdispl = qdispl;
    s->qcount = qcount;
    s->nranks = diverge_mpi_comm_size();
    s->myrank = diverge_mpi_comm_rank();
    s->qcounts = calloc( s->nranks, sizeof*s->qcounts );
    s->qdispls = calloc( s->nranks, sizeof*s->qdispls );
    diverge_mpi_allgather_index( &s->qdispl, s->qdispls, 1 );
    diverge_mpi_allgather_index( &s->qcount, s->qcounts, 1 );

    s->loop_size = s->qcount * s->nk * POW4(s->nb);

    complex128_t *pP = (complex128_t*)calloc( s->loop_size, sizeof*pP );
    complex128_t *pC = (complex128_t*)calloc( s->loop_size, sizeof*pC );
    index_t *mP = (index_t*)calloc( s->loop_size, sizeof*mP );
    index_t *mC = (index_t*)calloc( s->loop_size, sizeof*mC );

    s->symops_P = calloc( s->nsym * s->loop_size, sizeof*s->symops_P );
    s->symops_C = calloc( s->nsym * s->loop_size, sizeof*s->symops_C );

    for (index_t i=0; i<s->nsym; ++i) {
        grid_symm_transform_get_trafo( s, i, qdispl, qcount, pP, pC, mP, mC );
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t l=0; l<s->loop_size; ++l) {
            s->symops_P[IDX2(i,l,s->loop_size)] = (grid_symm_op_t){
                .current_rank = s->myrank,
                .current_index = qdispl*s->nk*POW4(s->nb) + l,
                .phase = pP[l],
                .target_index = mP[l],
                .target_rank = rank_from_index( s->qdispls, s->qcounts, s->nranks, mP[l]/(s->nk*POW4(s->nb)) ),
            };
            s->symops_C[IDX2(i,l,s->loop_size)] = (grid_symm_op_t){
                .current_rank = s->myrank,
                .current_index = qdispl*s->nk*POW4(s->nb) + l,
                .phase = pC[l],
                .target_index = mC[l],
                .target_rank = rank_from_index( s->qdispls, s->qcounts, s->nranks, mC[l]/(s->nk*POW4(s->nb)) ),
            };
        }
        mergesort_symop( s->symops_P + i*s->loop_size, s->loop_size, diverge_omp_num_threads(), 0 );
        mergesort_symop( s->symops_C + i*s->loop_size, s->loop_size, diverge_omp_num_threads(), 0 );
    }

    free( pP );
    free( pC );
    free( mP );
    free( mC );
}

static grid_symm_comm_t grid_symm_transform_comm( grid_symm_transform_t* s, index_t symm, char chan ) {

    const grid_symm_op_t* symop = (chan == 'P' ? s->symops_P : s->symops_C) + symm * s->loop_size;

    int nranks = s->nranks;
    int rank = s->myrank;
    index_t* send_counts = calloc(nranks, sizeof*send_counts);
    for (index_t i=0; i<s->loop_size; ++i) send_counts[symop[i].target_rank]++;
    index_t* send_displs = calloc(nranks, sizeof*send_displs);
    for (int i=1; i<nranks; ++i) send_displs[i] = send_displs[i-1] + send_counts[i-1];

    index_t* all_send_counts = calloc(POW2(nranks), sizeof*all_send_counts);
    diverge_mpi_allgather_index( send_counts, all_send_counts, nranks );

    index_t* recv_counts = calloc(nranks, sizeof*recv_counts);
    index_t* recv_displs = calloc(nranks, sizeof*recv_displs);
    for (int i=0; i<nranks; ++i) recv_counts[i] = all_send_counts[IDX2(i,rank,nranks)];
    for (int i=1; i<nranks; ++i) recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];

    grid_symm_op_t* recv_op = calloc(s->loop_size, sizeof*recv_op);

    diverge_mpi_alltoallv_bytes( symop, send_counts, send_displs, recv_op,
            recv_counts, recv_displs, sizeof(grid_symm_op_t) );

    grid_symm_comm_t result = {
        .reorder_before_send = calloc(s->loop_size, sizeof*result.reorder_before_send),
        .phase_before_send = calloc(s->loop_size, sizeof*result.phase_before_send),
        .send_counts = send_counts,
        .send_displs = send_displs,
        .recv_counts = recv_counts,
        .recv_displs = recv_displs,
        .reorder_after_recv = calloc(s->loop_size, sizeof*result.reorder_after_recv),
        .orbital_matrix = s->orb_sym + POW2(s->nb) * symm,
        .nb = s->nb,
        .chan = chan,
    };

    free( all_send_counts );

    index_t qfac = s->nk * POW4(s->nb);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t x=0; x<s->loop_size; ++x) {
        result.reorder_before_send[x] = symop[x].current_index - s->qdispls[symop[x].current_rank]*qfac;
        result.phase_before_send[x] = symop[x].phase;
        result.reorder_after_recv[x] = recv_op[x].target_index - s->qdispls[recv_op[x].target_rank]*qfac;
    }

    free( recv_op );

    return result;
}

static void grid_symm_comm_free( grid_symm_comm_t S ) {
    free( S.reorder_before_send );
    free( S.phase_before_send );
    free( S.send_counts );
    free( S.send_displs );
    free( S.recv_counts );
    free( S.recv_displs );
    free( S.reorder_after_recv );
}

static void grid_symm_transform_init_loop( grid_symm_transform_t* s ) {
    s->comms_P = calloc(s->nsym, sizeof*s->comms_P);
    s->comms_C = calloc(s->nsym, sizeof*s->comms_P);
    for (index_t i=0; i<s->nsym; ++i) {
        s->comms_P[i] = grid_symm_transform_comm(s, i, 'P');
        s->comms_C[i] = grid_symm_transform_comm(s, i, 'C');
    }
    s->send_buf = calloc(s->loop_size, sizeof*s->send_buf);
    s->recv_buf = calloc(s->loop_size, sizeof*s->recv_buf);
    s->loop_buf = calloc(s->loop_size, sizeof*s->loop_buf);
}

static void grid_symm_transform_loop_single( grid_symm_transform_t* s, complex128_t* loop_in,
        complex128_t* loop_out, index_t sym, char chan ) {

    const grid_symm_comm_t comm = chan == 'P' ? s->comms_P[sym] : s->comms_C[sym];

    const index_t* send_map = comm.reorder_before_send;
    const complex128_t* send_phase = comm.phase_before_send;

    const index_t* send_counts = comm.send_counts;
    const index_t* send_displs = comm.send_displs;
    const index_t* recv_counts = comm.recv_counts;
    const index_t* recv_displs = comm.recv_displs;

    const index_t* recv_map = comm.reorder_after_recv;

    const index_t nb = comm.nb;
    const complex128_t* orbital_matrix = comm.orbital_matrix;
    ZERO_OUT( s->send_buf, s->loop_size );
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t qk=0; qk<s->loop_size/POW4(s->nb); ++qk) {
        index_t x = qk*POW4(s->nb);
        for (index_t o=0; o<POW4(s->nb); ++o)
        for (index_t op=0; op<POW4(s->nb); ++op) {

            const index_t dim[4] = {nb,nb,nb,nb};
            index_t O[4]; unravel( o, dim, 4, O );
            index_t Op[4]; unravel( op, dim, 4, Op );
            index_t o1,o2,o3,o4,
                    o1p,o2p,o3p,o4p;
            if (chan == 'P') {
                o1 = O[0], o2 = O[1], o3 = O[2], o4 = O[3],
                o1p = Op[0], o2p = Op[1], o3p = Op[2], o4p = Op[3];
            } else {
                o1 = O[1], o2 = O[2], o3 = O[3], o4 = O[0],
                o1p = Op[1], o2p = Op[2], o3p = Op[3], o4p = Op[0];
            }

            s->send_buf[x+op] += loop_in[send_map[x+o]] *
                     conj(orbital_matrix[IDX2(o1p,o1,nb)]) *
                     conj(orbital_matrix[IDX2(o2p,o2,nb)]) *
                         (orbital_matrix[IDX2(o3p,o3,nb)]) *
                         (orbital_matrix[IDX2(o4p,o4,nb)]) * send_phase[x+o];
        }
    }

    diverge_mpi_alltoallv_bytes( s->send_buf, send_counts, send_displs,
            s->recv_buf, recv_counts, recv_displs, sizeof(complex128_t) );

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t x=0; x<s->loop_size; ++x) {
        loop_out[recv_map[x]] += s->recv_buf[x];
    }
}

complex128_t* grid_symm_transform_loop( grid_symm_transform_t* s, complex128_t* loop, char chan ) {
    complex128_t* out = s->loop_buf;
    ZERO_OUT( out, s->loop_size );
    for (index_t i=0; i<s->nsym; ++i)
        grid_symm_transform_loop_single( s, loop, out, i, chan );
    double norm = 1./(double)s->nsym;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<s->loop_size; ++i) out[i] *= norm;
    return out;
}

bool grid_symm_transform_loop_checksyms( grid_symm_transform_t* s, complex128_t* loop, char chan ) {
    complex128_t* buf = s->loop_buf;
    bool errors = false;
    for (index_t i=0; i<s->nsym; ++i) {
        ZERO_OUT( buf, s->loop_size );
        grid_symm_transform_loop_single( s, loop, buf, i, chan );
        for (index_t x=0; x<s->loop_size; ++x) {
            double err = cabs( buf[x] - loop[x] );
            if (err > 1.e-13) {
                mpi_wrn_printf( "chan '%c', sym %i loop not symmetric (%li; %.5e%+.5ej - %.5e%+.5ej)\n",
                        chan, i, x, creal(buf[x]), cimag(buf[x]), creal(loop[x]), cimag(loop[x]) );
                errors = true;
            }
        }
    }
    return errors;
}

grid_symm_transform_t* grid_symm_transform_init( diverge_model_t* m, index_t qdispl, index_t qcount ) {
    grid_symm_transform_t* s = grid_symm_transform_init_transform(m);
    grid_symm_transform_init_communicator(s, qdispl, qcount);
    grid_symm_transform_init_loop(s);
    return s;
}
