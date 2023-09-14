/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "kmesh_to_bands.h"
#include "bitary.h"
#include "mpi_functions.h"
#include "../diverge_Eigen3.hpp"
#include "../diverge_model.h"
#include "../diverge_internals_struct.h"
#include "../diverge_momentum_gen.h"
#include <algorithm>
#include <vector>

using std::vector;

static void find_bandstructure_segment_indices( const double start[3],
        const double stop[3], const double* kfmesh, index_t nktot, index_t**
        indices, index_t* n_indices );
static index_t* diverge_kmesh_to_bands_both( diverge_model_t* m, index_t**
        indices, index_t* n_indices, bool fine );

index_t* diverge_kmesh_to_bands( diverge_model_t* m, index_t** indices, index_t* n_indices ) {
    return diverge_kmesh_to_bands_both( m, indices, n_indices, true );
}
index_t* diverge_kmesh_to_bands_crs( diverge_model_t* m, index_t** indices, index_t* n_indices ) {
    return diverge_kmesh_to_bands_both( m, indices, n_indices, false );
}

static void find_bandstructure_segment_indices( const double start_[3], const
        double stop_[3], const double* kfmesh, index_t nktot, index_t** indices,
        index_t* n_indices ) {

    Map<const Vec3d> start( start_ );
    Map<const Vec3d> stop( stop_ );

    // the line is given by L(t) = A + t * B
    const Vec3d A = start,
                B = stop - start;

    Mat3d Bmat = Mat3d::Zero();
    Bmat.col(0) = B;
    Eigen::HouseholderQR<Mat3d> Bqr(Bmat);
    Mat3d BQ = Bqr.householderQ();
    Mat3d BR = Bqr.matrixQR().triangularView<Eigen::Upper>();
    BQ *= BR(0,0); // directly get the norm right

    Mat3d BQI = BQ.inverse();

    cbyte_t* reached = bitary_alloc( nktot );

    index_t n_reached = 0;
    #pragma omp parallel for reduction(+:n_reached) num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<nktot; ++i) {
        Map<const Vec3d> K( kfmesh + 3*i );
        Vec3d Bb = BQ * (K-A);
        // check whether (K-A) is proportional to B, i.e. orthogonal to both QR
        // vectors of Bmat
        bool ortho_1 = (std::abs(Bb(1)) < DIVERGE_EPS_MESH);
        bool ortho_2 = (std::abs(Bb(2)) < DIVERGE_EPS_MESH);
        double tval = (BQI * (K-A))(0);

        if (ortho_1 && ortho_2 && tval > (-DIVERGE_EPS_MESH) && tval < (1+DIVERGE_EPS_MESH)) {
            bitary_set( reached, i );
            ++n_reached;
        }
    }

    vector<index_t> k_indices(n_reached);
    vector<double> tvals(n_reached);
    vector<index_t> t_indices(n_reached);
    index_t tidx = 0;

    #pragma omp parallel for schedule(dynamic) num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<nktot; ++i) {
        if (bitary_get( reached, i )) {
            Map<const Vec3d> K( kfmesh + 3*i );
            #pragma omp critical
            {
                tvals[tidx] = (BQI * (K-A))(0);
                k_indices[tidx] = i;
                t_indices[tidx] = tidx;
                tidx++;
            }
        }
    }

    free( reached );

    std::sort( t_indices.begin(), t_indices.end(), [&tvals](index_t i, index_t j){ return tvals[i]<tvals[j]; } );
    *indices = (index_t*)malloc(sizeof(index_t)*n_reached);
    *n_indices = n_reached;

    for (index_t i=0; i<n_reached; ++i)
        (*indices)[i] = k_indices[t_indices[i]];
}

static index_t* diverge_kmesh_to_bands_both( diverge_model_t* m, index_t**
        indices, index_t* n_indices, bool fine ) {
    if (m->n_ibz_path <= 1) {
        *n_indices = -1;
        *indices = NULL;
        mpi_err_printf( "cannot find IBZ path indices if n_ibz_path <= 1\n" );
        return NULL;
    }
    index_t *result = NULL, sresult = 0;
    double gvec[3][3];
    diverge_model_generate_mom_basis( m->lattice, gvec );
    Map<Mat3d> Gvec(gvec[0]);
    index_t* n_per_segment = (index_t*)malloc(sizeof(index_t)*m->n_ibz_path-1);
    for (index_t segment=0; segment<m->n_ibz_path-1; ++segment) {
        Map<Vec3d> start_crystal( m->ibz_path[segment] ),
                   stop_crystal( m->ibz_path[segment+1] );
        Vec3d start = Gvec * start_crystal,
              stop = Gvec * stop_crystal;

        if ((start-stop).norm() < DIVERGE_EPS_MESH)
            mpi_err_printf( "each segment must cover nonzero distance\n" );

        index_t *sres, nsres;

        if (fine)
            find_bandstructure_segment_indices( start.data(), stop.data(),
                m->internals->kfmesh, kdimtot(m->nk, m->nkf), &sres, &nsres );
        else
            find_bandstructure_segment_indices( start.data(), stop.data(),
                m->internals->kmesh, kdim(m->nk), &sres, &nsres );

        result = (index_t*)realloc( result, sizeof(index_t) * (sresult + nsres) );
        memcpy( result + sresult, sres, sizeof(index_t)*nsres );
        sresult += nsres;
        n_per_segment[segment] = nsres;
        free( sres );
    }
    *indices = result;
    *n_indices = sresult;
    return n_per_segment;
}

