/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_common.h"
#include "diverge_internals_struct.h"
#include "diverge_model_internals.h"
#include "diverge_Eigen3.hpp"
#include "diverge_momentum_gen.h"
#include "grid/vertex_memory.hpp"

#include <vector>
#include <array>
using std::vector;
using std::array;

typedef Eigen::Matrix<double,3,3,Eigen::RowMajor> CMat3d;
typedef Eigen::Matrix<complex128_t,-1,-1,Eigen::RowMajor> CMatXcd;

static void VertexMemory_destructor( void* _V ) {
    grid::VertexMemory* V = (grid::VertexMemory*)_V;
    delete V;
}

static void generate_refinement( const vector<Vec3d>& fine,
        const vector<Vec3d>& central_w_nghbrs, index_t coarse_center,
        const index_t* nk, const index_t* nkf, vector<index_t>& mesh,
        vector<double>& weights, vector<index_t>& rev ) {

    index_t nkcf[3] = {nk[0]*nkf[0], nk[1]*nkf[1], nk[2]*nkf[2]};

    mesh.reserve( fine.size() );
    weights.reserve( fine.size() );
    rev.reserve( fine.size() );

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (uindex_t i=0; i<fine.size(); ++i) {

        // calc dist to every coarse pt
        vector<double> dists;
        dists.reserve(central_w_nghbrs.size());
        for (const Vec3d& c: central_w_nghbrs)
            dists.push_back( (fine[i] - c).norm() );

        // argsort the distance list
        vector<index_t> idists( dists.size() );
        for (uindex_t d=0; d<dists.size(); ++d)
            idists[d] = d;
        std::sort( idists.begin(), idists.end(), [&dists]( index_t i, index_t j )->bool{
            return dists[i] < dists[j];
        } );

        // count how many points are closest
        index_t nclose = 0;
        while (std::abs(dists[idists[nclose]] - dists[idists[0]]) < DIVERGE_EPS_MESH) {
            nclose++;
            if ((uindex_t)nclose == idists.size())
                break;
        }

        // check whether first point is among the closest
        int is_fine_pt = 0;
        for (index_t n=0; n<nclose; ++n) is_fine_pt += (idists[n] == 0);

        if (is_fine_pt) {
        #pragma omp critical
        {
            index_t kf_rel = k1m2(i, kidxc2f(coarse_center, nk, nkf), nkcf);
            mesh.push_back(kf_rel);
            weights.push_back(1./(double)nclose);
        }}

    }

    // argsort && sort
    vector<index_t> mesh_i(mesh.size());
    for (uindex_t i=0; i<mesh.size(); ++i) mesh_i[i] = i;
    std::sort( mesh_i.begin(), mesh_i.end(), [&mesh](index_t i, index_t j)->bool{
            return mesh[i] < mesh[j];
    } );
    vector<index_t> mesh_cp = mesh;
    vector<double> mesh_wg = weights;
    for (uindex_t i=0; i<mesh.size(); ++i) {
        mesh[i] = mesh_cp[mesh_i[i]];
        weights[i] = mesh_wg[mesh_i[i]];
    }

    // generate kfrev
    for (uindex_t i=0; i<mesh.size(); ++i) {
        index_t mk = k1m2(0, mesh[i], nkcf);
        auto mk_ptr = std::find(mesh.begin(), mesh.end(), mk);
        rev.push_back( std::distance(mesh.begin(), mk_ptr) );
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
    index_t initial_nkf = model->nkf[0] * model->nkf[1] * model->nkf[2];
    index_t nkcf[3] = {nk[0]*nkf[0], nk[1]*nkf[1], nk[2]*nkf[2]};

    // fill fine kpts
    double* kpts = intls->kfmesh;
    vector<Vec3d> fine_pts( initial_nkf * intls->grid_nk );
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<initial_nkf * intls->grid_nk; ++k)
        fine_pts[k] = Map<Vec3d>( kpts+k*3 );

    // fill coarse kpts
    double* kcpts = intls->kmesh;
    vector<Vec3d> coarse_pts( intls->grid_nk );
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<intls->grid_nk; ++k)
        coarse_pts[k] = Map<Vec3d>( kcpts+k*3 );

    // find the central point (first element of vector) and all its neighbors
    // (all the other elements of vector)
    index_t central_coarse_idx = IDX3( model->nk[0]/2, model->nk[1]/2,
            model->nk[2]/2, model->nk[1], model->nk[2] );
    vector<index_t> cn_idx;
    for (index_t kx=-1; kx<=1; ++kx)
    for (index_t ky=-1; ky<=1; ++ky)
    for (index_t kz=-1; kz<=1; ++kz) {
        index_t dk = IDX3( (kx+nk[0])%nk[0], (ky+nk[1])%nk[1], (kz+nk[2])%nk[2], nk[1], nk[2] );
        index_t c_dk = k1p2( dk, central_coarse_idx, nk );
        if (c_dk != central_coarse_idx)
            cn_idx.push_back( c_dk );
    }
    std::sort( cn_idx.begin(), cn_idx.end() );
    auto end = std::unique( cn_idx.begin(), cn_idx.end() );
    cn_idx.erase( end, cn_idx.end() );
    cn_idx.insert( cn_idx.begin(), central_coarse_idx );
    vector<Vec3d> central_w_nghbrs;
    for (index_t i: cn_idx)
        central_w_nghbrs.push_back( coarse_pts[i] );

    vector<index_t> fine_mesh;
    vector<index_t> fine_rev;
    vector<double> fine_weights;
    generate_refinement( fine_pts, central_w_nghbrs, cn_idx[0], nk, nkf, fine_mesh, fine_weights, fine_rev );

    // now allocate the structures
    intls->grid_kfmesh = (index_t*)malloc(sizeof(index_t) * fine_mesh.size()*intls->grid_nk);
    intls->grid_nkf = fine_mesh.size();
    intls->grid_kfrev = (index_t*)malloc(sizeof(index_t) * fine_mesh.size());
    intls->grid_weights = (double*)malloc(sizeof(double) * fine_mesh.size());

    // and copy the stuff over
    memcpy( intls->grid_kfrev, fine_rev.data(), sizeof(index_t)*fine_rev.size() );
    memcpy( intls->grid_weights, fine_weights.data(), sizeof(index_t)*fine_weights.size() );
    // and set the kfzero point
    for (index_t i=0; i<(index_t)fine_rev.size(); ++i)
        if (fine_rev[i] == i)
            intls->grid_kfzero = i;

    // only the reordering map needs extra care
    index_t counter = 0;
    for (index_t c=0; c<intls->grid_nk; ++c)
    for (index_t m: fine_mesh)
        intls->grid_kfmesh[counter++] = k1p2(m, kidxc2f(c, nk, nkf), nkcf);

    intls->grid_k1m2 = (index_t*)malloc(sizeof(index_t)*intls->grid_nk*intls->grid_nk);
    intls->grid_k1p2 = (index_t*)malloc(sizeof(index_t)*intls->grid_nk*intls->grid_nk);
    #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t k1=0; k1<intls->grid_nk; ++k1)
    for (index_t k2=0; k2<intls->grid_nk; ++k2) {
        intls->grid_k1m2[IDX2(k1,k2,intls->grid_nk)] = k1m2(k1, k2, model->nk);
        intls->grid_k1p2[IDX2(k1,k2,intls->grid_nk)] = k1p2(k1, k2, model->nk);
    }

    index_t nb = model->n_orb * model->n_spin;
    intls->grid_greens_p_buf = (complex128_t*)calloc(intls->grid_nk*intls->grid_nkf*nb*nb, sizeof(complex128_t));
    intls->grid_greens_m_buf = (complex128_t*)calloc(intls->grid_nk*intls->grid_nkf*nb*nb, sizeof(complex128_t));

    grid::VertexMemory* V = new grid::VertexMemory(intls->grid_nk, intls->grid_nb,
            intls->grid_k1m2, intls->grid_k1p2, model->SU2 > 0);

    // DEVELOPERS
    //
    // example for ffill/vfill usage
    if (model->ffill != NULL) {
        mpi_vrb_printf("using full vertex generator\n");
        complex128_t* vbuf = V->vertex_buffers[2];
        vector<index_t> qlist = V->qshare();

        const index_t nk = intls->grid_nk;
        const index_t no4 = POW2(nb*nb);
        #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
        for (uindex_t k1=0; k1<qlist.size(); ++k1)
        for (index_t k2=0; k2<nk; ++k2)
        for (index_t k3=0; k3<nk; ++k3)
            (*(model->ffill))( model, qlist[k1], k2, k3, vbuf + IDX3(k1, k2, k3, nk, nk)*no4 );

        V->overwrite_transform( 0, 1, "D_to_V" );
        (*V)( vbuf, V->vertex_buffers[0], "V_to_D", false, 1.0 );
        V->overwrite_transform( 0, 1, "D_to_P" );
    } else {
        mpi_vrb_printf("using channel vertex generators\n");
        memset( (void*)V->vertex_buffers[0], 0, sizeof(complex128_t)*V->qshare().size()*POW2(intls->grid_nk)*POW4(ns*no) );
        complex128_t* qbuf_in = (complex128_t*)calloc(intls->grid_nk * POW2(ns*ns*no), sizeof(complex128_t));
        for (char chan : {'C', 'P', 'D'}) {
            memset( (void*)qbuf_in, 0, sizeof(complex128_t)*intls->grid_nk * POW2(ns*ns*no) );
            if ((*(model->vfill))( model, chan, qbuf_in ))
                V->init_from_qbuf( qbuf_in, chan, ns );
        }
        free(qbuf_in);
    }

    // DEVELOPERS
    //
    // example for exchange symmetry enforcment
    if (intls->enforce_exchange) {
        mpi_vrb_printf( "enforcing exchange symmetry of initial vertex\n" );
        V->exchange_symmetry();
    }
    intls->grid_vertex = (void*)V;
    intls->grid_vertex_destructor = &VertexMemory_destructor;
}
