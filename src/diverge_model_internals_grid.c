/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_common.h"
#include "diverge_internals_struct.h"
#include "diverge_model_internals.h"
#include "diverge_momentum_gen.h"

#include "grid/vertex_memory.h"
#include "misc/mpi_functions.h"
#define LINALG_NO_LAPACKE
#include "misc/linalg.h"
LINALG_IMPLEMENT
#include "misc/mergesort.h"

static void grid_vertex_destructor( void* _V ) { grid_vertex_free((grid_vertex_t*)_V); }

typedef struct { VECTOR_DECL( index_t, v ) } vec_index_t;
typedef struct { VECTOR_DECL( double, v ) } vec_double;

#undef SERISORT_COMP_R
#define SERISORT_COMP_R( i, j, dists ) (((double*)(dists))[i] < ((double*)(dists))[j] ? -1 : \
                                        ((double*)(dists))[i] > ((double*)(dists))[j] ? 1 : 0)
SERISORT_IMPL_R( distsort, index_t )
#undef SERISORT_COMP_R
#define SERISORT_COMP_R( i, j, mesh ) (((index_t*)(mesh))[i] < ((index_t*)(mesh))[j] ? -1 : \
                                       ((index_t*)(mesh))[i] > ((index_t*)(mesh))[j] ? 1 : 0)
SERISORT_IMPL_R( idxsort, index_t )
SERISORT_IMPL( isort, index_t )

static inline index_t binsearch( const index_t* vals, index_t sz, index_t val ) {
    index_t low = 0, high = sz - 1;
    while (low <= high) {
        index_t mid = low + (high - low) / 2;
        if (vals[mid] == val) {
            return mid;
        } else if (vals[mid] < val) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return -1;
}

static inline index_t unique_idx( index_t* arr, index_t size ) {
    if (size == 0) return 0;
    index_t j = 0;
    for (index_t i=1; i<size; ++i) if (arr[i] != arr[j]) arr[++j] = arr[i];
    return j+1;
}


static void generate_refinement( const double* fine/*(fine_sz,3)*/, index_t fine_sz,
        const claV3d_t* central_w_nghbrs, index_t central_w_nghbrs_sz,
        index_t coarse_center, const index_t* nk, const index_t* nkf,
        /*output*/ vec_index_t* mesh, vec_double* weights, vec_index_t* rev ) {

    index_t nkcf[3] = {nk[0]*nkf[0], nk[1]*nkf[1], nk[2]*nkf[2]};

    VECTOR_RESERVE( mesh->v, fine_sz );
    VECTOR_RESERVE( weights->v, fine_sz );
    VECTOR_RESERVE( rev->v, fine_sz );

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<fine_sz; ++i) {

        // calc dist to every coarse pt
        double dists[central_w_nghbrs_sz];
        for (index_t c=0; c<central_w_nghbrs_sz; ++c)
            dists[c] = claV3d_norm(claV3d_sub(claV3d_map(fine+3*i),central_w_nghbrs[c]));

        index_t idists[central_w_nghbrs_sz];
        for (index_t d=0; d<central_w_nghbrs_sz; ++d) idists[d] = d;

        distsort_r( idists, central_w_nghbrs_sz, dists );

        // count how many points are closest
        index_t nclose = 0;
        while (fabs(dists[idists[nclose]] - dists[idists[0]]) < DIVERGE_EPS_MESH)
            if (++nclose == central_w_nghbrs_sz) break;

        // check whether first point is among the closest
        int is_fine_pt = 0;
        for (index_t n=0; n<nclose; ++n) is_fine_pt += (idists[n] == 0);

        if (is_fine_pt) {
        #pragma omp critical
        {
            index_t kf_rel = k1m2(i, kidxc2f(coarse_center, nk, nkf), nkcf);
            VECTOR_PUSH_BACK( mesh->v, kf_rel );
            VECTOR_PUSH_BACK( weights->v, 1./(double)nclose );
        }}
    }

    // argsort && sort
    index_t* mesh_i = calloc(mesh->v_sz, sizeof*mesh_i);
    for (index_t i=0; i<mesh->v_sz; ++i) mesh_i[i] = i;
    idxsort_r( mesh_i, mesh->v_sz, mesh->v );
    index_t* mesh_cp = calloc(mesh->v_sz, sizeof*mesh_cp);
    double* mesh_wg_cp = calloc(mesh->v_sz, sizeof*mesh_wg_cp);
    memcpy( mesh_cp, mesh->v, sizeof*mesh_cp * mesh->v_sz );
    memcpy( mesh_wg_cp, weights->v, sizeof*mesh_wg_cp * mesh->v_sz );
    for (index_t i=0; i<mesh->v_sz; ++i) {
        mesh->v[i] = mesh_cp[mesh_i[i]];
        weights->v[i] = mesh_wg_cp[mesh_i[i]];
    }
    free( mesh_cp );
    free( mesh_wg_cp );
    free( mesh_i );

    // generate kfrev
    for (index_t i=0; i<mesh->v_sz; ++i) {
        index_t inv_i = binsearch(mesh->v, mesh->v_sz, k1m2(0, mesh->v[i], nkcf));
        VECTOR_PUSH_BACK( rev->v, inv_i );
    }

}

void diverge_model_internals_grid( diverge_model_t* model ) {
    if (!model->internals->has_common_internals) {
        mpi_err_printf("model needs common internals before grid internals!\n");
        diverge_model_internals_common(model);
    }
    internals_t* intls = model->internals;
    strcpy( intls->backend, "grid" );
    const index_t* nk = model->nk;
    const index_t* nkf = model->nkf;

    // set internals' nkpts
    const index_t ns = model->n_spin;
    const index_t no = model->n_orb;
    intls->grid_nk = model->nk[0] * model->nk[1] * model->nk[2];
    intls->grid_nb = model->n_orb * ns;
    index_t nkcf[3] = {nk[0]*nkf[0], nk[1]*nkf[1], nk[2]*nkf[2]};

    // find the central point (first element of vector) and all its neighbors
    // (all the other elements of vector)
    index_t central_coarse_idx = IDX3( model->nk[0]/2, model->nk[1]/2, model->nk[2]/2,
                                       model->nk[1], model->nk[2] );
    index_t cn_idx[30];
    index_t cn_idx_sz = 0;
    for (index_t kx=-1; kx<=1; ++kx)
    for (index_t ky=-1; ky<=1; ++ky)
    for (index_t kz=-1; kz<=1; ++kz) {
        index_t dk = IDX3( (kx+nk[0])%nk[0], (ky+nk[1])%nk[1], (kz+nk[2])%nk[2], nk[1], nk[2] );
        index_t c_dk = k1p2( dk, central_coarse_idx, nk );
        if (c_dk != central_coarse_idx)
            cn_idx[cn_idx_sz++] = c_dk;
    }
    cn_idx[cn_idx_sz++] = -1; // first element after sort
    isort( cn_idx, cn_idx_sz );
    cn_idx_sz = unique_idx( cn_idx, cn_idx_sz );
    cn_idx[0] = central_coarse_idx;

    double* coarse_pts = intls->kmesh;
    claV3d_t central_w_nghbrs[cn_idx_sz];
    for (index_t ii=0; ii<cn_idx_sz; ++ii)
        central_w_nghbrs[ii] = claV3d_map(coarse_pts+3*cn_idx[ii]);

    vec_index_t fine_mesh = {0}, fine_rev = {0};
    vec_double fine_weights = {0};
    generate_refinement( intls->kfmesh, kdim(nkcf), central_w_nghbrs, cn_idx_sz,
            central_coarse_idx, nk, nkf, &fine_mesh, &fine_weights, &fine_rev );

    intls->grid_nkf = fine_mesh.v_sz;
    intls->grid_kfrev = fine_rev.v;
    intls->grid_weights = fine_weights.v;

    // the kfmesh needs some care, because it must save *all* points
    intls->grid_kfmesh = malloc(sizeof(index_t) * fine_mesh.v_sz*intls->grid_nk);
    index_t counter = 0;
    for (index_t c=0; c<intls->grid_nk; ++c)
    for (index_t i=0; i<intls->grid_nkf; ++i)
        intls->grid_kfmesh[counter++] = k1p2(fine_mesh.v[i], kidxc2f(c, nk, nkf), nkcf);
    VECTOR_FREE( fine_mesh.v );

    intls->grid_k1m2 = malloc(sizeof(index_t)*intls->grid_nk*intls->grid_nk);
    intls->grid_k1p2 = malloc(sizeof(index_t)*intls->grid_nk*intls->grid_nk);
    #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t k1=0; k1<intls->grid_nk; ++k1)
    for (index_t k2=0; k2<intls->grid_nk; ++k2) {
        intls->grid_k1m2[IDX2(k1,k2,intls->grid_nk)] = k1m2(k1, k2, model->nk);
        intls->grid_k1p2[IDX2(k1,k2,intls->grid_nk)] = k1p2(k1, k2, model->nk);
    }

    index_t nb = model->n_orb * model->n_spin;
    intls->grid_greens_p_buf = calloc(intls->grid_nk*intls->grid_nkf*nb*nb, sizeof(complex128_t));
    intls->grid_greens_m_buf = calloc(intls->grid_nk*intls->grid_nkf*nb*nb, sizeof(complex128_t));

    grid_vertex_t* V = grid_vertex_init(intls->grid_nk, intls->grid_nb,
            intls->grid_k1m2, intls->grid_k1p2, model->SU2 > 0);

    // DEVELOPERS
    //
    // example for ffill/vfill usage
    if (model->ffill != NULL) {
        mpi_vrb_printf("using full vertex generator\n");
        complex128_t* vbuf = V->vertex_buffers[2];

        const index_t nk = intls->grid_nk;
        const index_t no4 = POW4(nb);
        #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
        for (index_t k1=0; k1<V->qcount; ++k1)
        for (index_t k2=0; k2<nk; ++k2)
        for (index_t k3=0; k3<nk; ++k3)
            (*(model->ffill))( model, V->qdispl+k1, k2, k3, vbuf + IDX3(k1, k2, k3, nk, nk)*no4 );

        grid_vertex_overwrite_transform( V, 0, 1, "D_to_V" );
        grid_vertex_project_name( V, vbuf, V->vertex_buffers[0], "V_to_D", false, 1.0 );
        grid_vertex_overwrite_transform( V, 0, 1, "D_to_P" );
    } else {
        mpi_vrb_printf("using channel vertex generators\n");
        ZERO_OUT( V->vertex_buffers[0], V->size );
        complex128_t* qbuf_in = calloc(intls->grid_nk * POW2(ns*ns*no), sizeof(complex128_t));
        for (const char* pchan="CPD"; *pchan; pchan++) {
            ZERO_OUT( qbuf_in, intls->grid_nk * POW2(ns*ns*no) );
            if ((*(model->vfill))( model, *pchan, qbuf_in ))
                grid_vertex_from_qbuf( V, qbuf_in, *pchan, ns );
        }
        free(qbuf_in);
    }

    // DEVELOPERS
    //
    // example for exchange symmetry enforcment
    if (intls->enforce_exchange) {
        mpi_vrb_printf( "enforcing exchange symmetry of initial vertex\n" );
        grid_vertex_exchange_symmetry( V, 0.5 );
    }
    intls->grid_vertex = (void*)V;
    intls->grid_vertex_destructor = &grid_vertex_destructor;
}
