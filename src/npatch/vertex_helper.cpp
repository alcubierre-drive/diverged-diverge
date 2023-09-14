/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "../diverge_Eigen3.hpp"
#include "../misc/mpi_functions.h"
#include <algorithm>
#include <vector>
#include <utility>

using std::vector;

#include "vertex.h"
#include "vertex_helper.h"

void vhelper_vertex_sort_in_q( complex128_t* vertex, qkkp_t* vindex, index_t np,
        index_t nb, index_t** pqcounts, index_t* pnq ) {

    index_t np3 = np*np*np;
    index_t no4 = nb*nb*nb*nb;
    index_t size = np3 * no4;

    typedef std::pair<complex128_t,qkkp_t> v_element_t;
    vector<v_element_t> vsort(size);

    #pragma omp parallel for schedule(static) collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t p3=0; p3<np3; ++p3) for (index_t o4=0; o4<no4; ++o4)
        vsort[p3*no4+o4] = { vertex[p3*no4+o4], vindex[p3] };

    std::stable_sort( vsort.begin(), vsort.end(),
            [&]( const v_element_t& va, const v_element_t& vb )->bool{
                return va.second.kp < vb.second.kp;
            } );
    std::stable_sort( vsort.begin(), vsort.end(),
            [&]( const v_element_t& va, const v_element_t& vb )->bool{
                return va.second.k < vb.second.k;
            } );
    std::stable_sort( vsort.begin(), vsort.end(),
            [&]( const v_element_t& va, const v_element_t& vb )->bool{
                return va.second.q < vb.second.q;
            } );

    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t p3=0; p3<np3; ++p3) {
        for (index_t o4=0; o4<no4; ++o4) {
            vertex[p3*no4+o4] = vsort[p3*no4+o4].first;
        }
        vindex[p3] = vsort[p3*no4].second;
    }

    index_t* qcounts = (index_t*)calloc(np*np, sizeof(index_t));
    qcounts[0] = 1;
    index_t nq = 0;
    for (index_t p3=1; p3<np3; ++p3) {
        if (vindex[p3-1].q != vindex[p3].q)
            nq++;
        qcounts[nq]++;
    }
    nq++;

    *pqcounts = qcounts;
    *pnq = nq;
}

npatch_qvert_mat_t* vhelper_q_vertex_matrix_from_vertex( const complex128_t* q_vertex,
        const qkkp_t* q_vindex, const index_t* q_counts, index_t nq,
        index_t nb ) {

    vector<vector<index_t>> npatch_qvert_mat_descr_k_kp;
    vector<vector<complex128_t>> npatch_qvert_mat;

    index_t* q_displs = (index_t*)calloc(nq, sizeof(index_t));
    for (index_t q=0; q<nq; ++q) if (q>0) q_displs[q] = q_displs[q-1] + q_counts[q-1];

    npatch_qvert_mat_t* result = (npatch_qvert_mat_t*)malloc(sizeof(npatch_qvert_mat_t)*nq);

    for (index_t q=0; q<nq; ++q) {
        vector<index_t> k_kp_vector;
        k_kp_vector.reserve( q_counts[q] );

        typedef vector<qkkp_t> qkkp_vec_t;
        typedef qkkp_vec_t::iterator qkkp_iter_t;
        qkkp_vec_t qvi(q_counts[q]);
        qkkp_iter_t qvi_end;

        memcpy( qvi.data(), q_vindex + q_displs[q], sizeof(qkkp_t)*q_counts[q] );
        std::sort( qvi.begin(), qvi.end(), [](const qkkp_t& a, const qkkp_t& b)->bool{ return a.k > b.k; } );
        qvi_end = std::unique( qvi.begin(), qvi.end(),
                [](const qkkp_t& a, const qkkp_t& b)->bool{ return a.k == b.k; } );
        for (qkkp_iter_t it = qvi.begin(); it != qvi_end; ++it)
            k_kp_vector.push_back( it->k );

        memcpy( qvi.data(), q_vindex + q_displs[q], sizeof(qkkp_t)*q_counts[q] );
        std::sort( qvi.begin(), qvi.end(), [](const qkkp_t& a, const qkkp_t& b)->bool{ return a.kp > b.kp; } );
        qvi_end = std::unique( qvi.begin(), qvi.end(),
                [](const qkkp_t& a, const qkkp_t& b)->bool{ return a.kp == b.kp; } );
        for (qkkp_iter_t it = qvi.begin(); it != qvi_end; ++it)
            k_kp_vector.push_back( it->kp );

        std::sort( k_kp_vector.begin(), k_kp_vector.end() );
        auto k_kp_end = std::unique( k_kp_vector.begin(), k_kp_vector.end() );
        k_kp_vector.erase( k_kp_end, k_kp_vector.end() );

        npatch_qvert_mat_descr_k_kp.push_back( k_kp_vector );
        index_t nkkp = k_kp_vector.size();
        index_t matrix_size = nkkp * nkkp * nb*nb*nb*nb;
        vector<complex128_t> matrix(matrix_size, 0.0);
        for (index_t qq=0; qq<q_counts[q]; ++qq) {
            const qkkp_t* qvi = q_vindex + q_displs[q] + qq;
            auto pki = std::find( k_kp_vector.begin(), k_kp_vector.end(), qvi->k );
            auto pkpi = std::find( k_kp_vector.begin(), k_kp_vector.end(), qvi->kp );
            index_t ki = std::distance( k_kp_vector.begin(), pki );
            index_t kpi = std::distance( k_kp_vector.begin(), pkpi );
            for (index_t o12=0; o12<nb*nb; ++o12)
            for (index_t o34=0; o34<nb*nb; ++o34)
                matrix[(ki*nb*nb+o12) * (nkkp*nb*nb) + kpi*nb*nb+o34] =
                    q_vertex[(q_displs[q]+qq)*nb*nb*nb*nb+o12*nb*nb+o34];
        }
        npatch_qvert_mat.push_back( matrix );

        result[q].data = (complex128_t*)malloc(sizeof(complex128_t)*matrix.size());
        memcpy( result[q].data, matrix.data(), sizeof(complex128_t)*matrix.size() );
        result[q].k_kp_descr = (index_t*)malloc(sizeof(index_t)*nkkp);
        memcpy( result[q].k_kp_descr, k_kp_vector.data(), sizeof(index_t)*nkkp );
        result[q].n_k_kp = nkkp;
        result[q].nb = nb;
        result[q].q = q_vindex[q_displs[q]].q;
        result[q].eigen = false;
        result[q].values = NULL;
        result[q].nv = -1;
    }

    free(q_displs);

    return result;
}

complex128_t* vhelper_q_vertex_matrices_to_buffer( npatch_qvert_mat_t* qvm, index_t nq, index_t Nmax ) {
    complex128_t* result = (complex128_t*)calloc(Nmax*Nmax*nq, sizeof(complex128_t));
    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t q=0; q<nq; ++q) {
        index_t N_orig = qvm[q].nb * qvm[q].nb * qvm[q].n_k_kp;
        Map<MatXcd> data_orig( qvm[q].data, N_orig, N_orig );
        Map<MatXcd> data_new( (result + q*Nmax*Nmax), Nmax, Nmax );
        if (data_orig.size() <= data_new.size())
            data_new.block(0,0,data_orig.rows(),data_orig.cols()) = data_orig;
        else
            data_new = data_orig.block(0,0,data_new.rows(),data_new.cols());
    }
    return result;
}

static inline double q_vertex_norm( const complex128_t* data, index_t matrix_size ) {
    return (Map<MatXcd>( const_cast<complex128_t*>(data), matrix_size, matrix_size )).norm() / (double)matrix_size;
}

void vhelper_q_vertex_sort_max( npatch_qvert_mat_t* qvm, index_t* pnq, double max_rel ) {
    typedef std::pair<double,index_t> norm_t;
    vector<norm_t> norms(*pnq);
    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t q=0; q<*pnq; ++q) {
        norm_t elem;
        elem.first = q_vertex_norm( qvm[q].data, qvm[q].n_k_kp*qvm[q].nb*qvm[q].nb );
        elem.second = q;
        norms[q] = elem;
    }
    std::sort( norms.begin(), norms.end(), []( const norm_t& A, const norm_t& B )->bool{
            return A.first > B.first;
        } );
    index_t qmax = 0;
    for (index_t q=0; q<*pnq; ++q)
        if (norms[q].first >= norms[0].first*max_rel)
            qmax++;

    vector<npatch_qvert_mat_t> buf(*pnq);
    for (index_t q=0; q<*pnq; ++q)
        buf[q] = qvm[norms[q].second];
    for (index_t q=0; q<*pnq; ++q)
        qvm[q] = buf[q];
    *pnq = qmax;
}
