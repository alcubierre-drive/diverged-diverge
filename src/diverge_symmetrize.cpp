/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_model.h"
#include "diverge_symmetrize.h"
#include "diverge_internals_struct.h"
#include "diverge_momentum_gen.h"
#include "diverge_Eigen3.hpp"
#include "misc/mpi_functions.h"
#include "misc/merge_orb_w_rs_symm.h"
#include <vector>
#include <cassert>

#ifndef SYMNORM_SQUARED_POSITION
#define SYMNORM_SQUARED_POSITION 1e-6
#endif
#ifndef SYMNORM_ASSERT_POSITION_BEYOND
#define SYMNORM_ASSERT_POSITION_BEYOND 1e-4
#endif
#ifndef SYMNORM_ORBITAL_OVERLAP
#define SYMNORM_ORBITAL_OVERLAP 1e-5
#endif
#ifndef SYMNORM_POS_EQUAL
#define SYMNORM_POS_EQUAL 1e-8
#endif
#ifndef SYMNOORM_MIN_ORBITAL_WEIGHT
#define SYMNOORM_MIN_ORBITAL_WEIGHT 0.99999
#endif

typedef Eigen::Matrix<double,3,3,Eigen::RowMajor> CMat3d;
typedef Matrix<index_t,3,1> Vec3idx;
using std::vector;

static inline bool map_back_to_BZ(Vec3d* k_point, index_t* idx, const diverge_model_t* model, bool fine) {
    Vec3idx nk = Map<const Vec3idx>( model->nk );
    if (fine) nk.array() *= Map<const Vec3idx>( model->nkf ).array();

    double* mesh = fine ? model->internals->kfmesh : model->internals->kmesh;
    Map<const CMat3d> UC(model->lattice[0]);

    Vec3d a = UC * (*k_point) / (2.*M_PI);
    for (int i=0; i<3; ++i)
        a[i] += labs(lround(a[i])) + 3;
    Vec3d ai = (a.array() * nk.array().cast<double>()).round().matrix();

    *idx = IDX3( lround(ai(0))%nk(0), lround(ai(1))%nk(1), lround(ai(2))%nk(2), nk(1), nk(2));
    *k_point = Map<Vec3d>( mesh+3*(*idx) );
    return (*idx < (nk(0)*nk(1)*nk(2)));
}

// allocates *ptr using realloc, i.e., must be free'd by user with free.
static void syms_serialize( const diverge_model_t* model, char** ptr, uindex_t* nbytes );
// allocates internals_t* using malloc/calloc, must be free'd by user with free
static internals_t* syms_deserialize( const diverge_model_t* model, char* ptr );
// checks whether two internals_t objects are equal in their symmetry parts
static int syms_equal( const diverge_model_t* model, const internals_t* ii_B );

void diverge_generate_symm_maps( diverge_model_t* model ) {
    if (model->internals->symm_orb_off &&
        model->internals->symm_orb_len &&
        model->internals->symm_beyond_UC &&
        model->internals->symm_map_mom_fine &&
        model->internals->symm_map_mom_crs &&
        model->internals->symm_map_orb &&
        model->internals->symm_pref &&
        model->n_sym > 0) return;
    mpi_vrb_printf("generate symmetries with %d elements\n", model->n_sym)

    index_t nk = kdim(model->nk);
    index_t nktot = kdimtot(model->nk, model->nkf);
    index_t n_os = model->n_orb * model->n_spin;
    index_t n_spin = model->n_spin;
    index_t n_orb = model->n_orb;
    double* kmesh = model->internals->kmesh;
    double* kfmesh = model->internals->kfmesh;
    index_t n_sym = model->n_sym;

    model->internals->symm_orb_off = (index_t*)malloc( n_os * n_sym * sizeof(index_t) );
    model->internals->symm_orb_len = (index_t*)malloc( n_os * n_sym * sizeof(index_t) );

    model->internals->symm_map_mom_fine = (index_t*)malloc( nktot * n_sym * sizeof(index_t) );
    model->internals->symm_map_mom_crs = (index_t*)malloc( nk * n_sym * sizeof(index_t) );
    model->internals->symm_beyond_UC = (double*)malloc( n_orb * n_sym * 3 * sizeof(double) );

    index_t* symm_orb_off = model->internals->symm_orb_off;
    index_t* symm_orb_len = model->internals->symm_orb_len;
    index_t* symm_map_mom_fine = model->internals->symm_map_mom_fine;
    index_t* symm_map_mom_crs = model->internals->symm_map_mom_crs;
    double* symm_beyond_UC = model->internals->symm_beyond_UC;

    double* rs_symmetries = (double*)model->rs_symmetries;
    complex128_t* orb_symmetries = (complex128_t*)model->orb_symmetries;

    mpi_vrb_printf( "finding coarse IBZ…\n" );
    #pragma omp parallel for schedule(static) collapse(2) num_threads(diverge_omp_num_threads())
    for(index_t s = 0; s < n_sym; ++s)
    for(index_t k = 0; k < nk; ++k) {
        Map<CMat3d> sym(rs_symmetries+s*9);
        Map<Vec3d> kpnt(kmesh+k*3);
        Vec3d tmp =  sym * kpnt;

        index_t idx;
        bool found = map_back_to_BZ(&tmp, &idx, model, false);
        symm_map_mom_crs[s+n_sym*k] = idx;
        if (!found) mpi_err_printf("not all momenta mappable (%d->%d; accuracy issue?)\n", s, k);
    }

    mpi_vrb_printf( "finding fine IBZ…\n" );
    #pragma omp parallel for schedule(static) collapse(2) num_threads(diverge_omp_num_threads())
    for(index_t s = 0; s < n_sym; ++s)
    for(index_t k = 0; k < nktot; ++k) {
        Map<CMat3d> sym(rs_symmetries+s*9);
        Map<Vec3d> kpnt(kfmesh+k*3);
        Vec3d tmp =  sym * kpnt;
        index_t idx;
        bool found = map_back_to_BZ(&tmp, &idx, model, true);
        symm_map_mom_fine[s+n_sym*k] = idx;
        if (!found) mpi_err_printf("not all momenta mappable (%d->%d; accuracy issue?)\n", s, k);
    }

    mpi_vrb_printf( "finding orbital maps…\n" );
    vector<index_t> orbmaps_to_len; orbmaps_to_len.resize(n_os*n_sym);
    vector<index_t> orbmaps_to_off; orbmaps_to_off.resize(n_os*n_sym);
    vector<index_t> orbmaps_to;
    vector<complex128_t> orbmaps_to_prefac;
    vector<Vec3d> mapped_pos; mapped_pos.resize(n_orb*n_sym);

    int periodic_iter[3];
    periodic_iter[0] = model->internals->periodic_direction[0] ? 3:0;
    periodic_iter[1] = model->internals->periodic_direction[1] ? 3:0;
    periodic_iter[2] = model->internals->periodic_direction[2] ? 3:0;

    Map<Vec3d> UC1(model->lattice[0]);
    Map<Vec3d> UC2(model->lattice[1]);
    Map<Vec3d> UC3(model->lattice[2]);

    // TODO the symmetrizer fails if multiple orbitals are on the same position,
    // but the position is shifted by a lattice vector (i.e. +A1/2 and -A1/2)

    #ifndef ASAN_COMPILATION
    _Pragma("omp parallel for collapse(2) num_threads(diverge_omp_num_threads())")
    #endif
    for(index_t s = 0; s < n_sym; ++s)
    for(index_t o1 = 0; o1 < n_orb; ++o1) {
        Map<CMat3d> sym_rs(rs_symmetries+s*9);
        Map<Vec3d> from(model->positions[o1]);
        Map<Vec3d> bey_UC(symm_beyond_UC+3*(s+n_sym*o1));
        Vec3d to = sym_rs*from;
        index_t maps_to = -1;
        for(index_t o2 = 0; o2 < n_orb; ++o2)
        for(int in = -periodic_iter[0];in<=periodic_iter[0];in++)
        for(int im = -periodic_iter[1];im<=periodic_iter[1];im++)
        for(int iq = -periodic_iter[2];iq<=periodic_iter[2];iq++) {
            Map<Vec3d> to_trial(model->positions[o2]);
            Vec3d trial = to_trial+in*UC1+im*UC2+iq*UC3;
            if((trial-to).squaredNorm() < SYMNORM_SQUARED_POSITION) {
                maps_to = o2;
                mapped_pos[o1+n_orb*s] = to_trial;
                bey_UC = ( to - to_trial );
                assert((bey_UC-(in*UC1+im*UC2+iq*UC3)).norm() < SYMNORM_ASSERT_POSITION_BEYOND);
                goto symmetrizer_orb_found;
            }
        }
        if(maps_to == -1) {
            #pragma omp critical
            {
                mpi_wrn_printf("position %li doesn't map under symmetry %li!\n", o1, s);
                model->n_sym = 0;
            }
        }
        symmetrizer_orb_found:
        Map<Vec3d> from_in_UC(model->positions[maps_to]);
    }

    for(index_t s = 0; s < n_sym; ++s)
    for(index_t s1 = 0; s1 < n_spin; ++s1)
    for(index_t o1 = 0; o1 < n_orb; ++o1) {
        index_t len = 0;
        for(index_t s1t = 0; s1t < n_spin; ++s1t)
        for(index_t o1t = 0; o1t < n_orb; ++o1t) {
            Map<Vec3d> pos(model->positions[o1t]);
            complex128_t overlap = orb_symmetries[IDX5(s, s1t, o1t, s1, o1, n_spin,n_orb, n_spin,n_orb)];
            if(std::abs(overlap) > SYMNORM_ORBITAL_OVERLAP && (mapped_pos[o1+n_orb*s] - pos).norm() < SYMNORM_POS_EQUAL) {
                orbmaps_to.push_back(IDX2(s1t,o1t,n_orb));
                orbmaps_to_prefac.push_back(overlap);
                len+=1;
            }
        }
        if(len == 0) {
            mpi_wrn_printf("orbital %li doesn't map under symmetry %li!\n", o1, s);
            model->n_sym = 0;
        }
        orbmaps_to_len[IDX3(s,s1,o1,n_spin,n_orb)] = len;
    }
    orbmaps_to_off[0] = 0;
    for (index_t i = 1; i < n_os*n_sym; ++i) {
        orbmaps_to_off[i] = orbmaps_to_off[i-1] + orbmaps_to_len[i-1];
    }
    for (index_t i = 0; i < n_os*n_sym; ++i) {
        if(orbmaps_to_len[i]>1) {
            mpi_vrb_printf("Symmetries map one to many \n");
            i= n_os*n_sym;
        }
    }
    for(index_t o1 = 0; o1 < n_os*n_sym; ++o1) {
        if(orbmaps_to_len[o1] == 0) {
            mpi_err_printf("orbital maps not constructed correctly, there is an empty one \n");
            model->n_sym = 0;
        }
        double sum = 0.;
        for(index_t e = 0; e < orbmaps_to_len[o1]; ++e) {
            sum += std::abs(orbmaps_to_prefac[orbmaps_to_off[o1]+e]
                    *conj(orbmaps_to_prefac[orbmaps_to_off[o1]+e]));
        }
        if(sum < SYMNOORM_MIN_ORBITAL_WEIGHT) {
            mpi_err_printf("Orbitals do not sum up to one %1.3f  \n", sum);
            model->n_sym = 0;
        }
    }


    index_t total_length = 0;
    for(index_t s = 0; s < n_sym; ++s)
    for(index_t o1 = 0; o1 < n_os; ++o1) {
        symm_orb_off[s + n_sym * o1] = orbmaps_to_off[s * n_os + o1];
        symm_orb_len[s + n_sym * o1] = orbmaps_to_len[s * n_os + o1];
        total_length += orbmaps_to_len[s * n_os + o1];
    }
    model->internals->symm_map_orb = (index_t*)malloc( total_length * sizeof(index_t) );
    model->internals->symm_pref = (complex128_t*)malloc( total_length * sizeof(complex128_t) );

    index_t* symm_map_orb = model->internals->symm_map_orb;
    complex128_t* symm_pref = model->internals->symm_pref;

    memcpy(symm_map_orb, orbmaps_to.data(), total_length * sizeof(index_t));
    memcpy(symm_pref, orbmaps_to_prefac.data(), total_length * sizeof(complex128_t));

    mpi_vrb_printf( "merging realspace orbitals…\n" );
    merge_rs_orb(model);

    if (model->internals->symcheck_mpi) {
        // serialization and communication
        char* ptr = NULL;
        uindex_t nbytes = 0;
        syms_serialize( model, &ptr, &nbytes );
        vector<index_t> nbytes_ranks( diverge_mpi_comm_size(), 0 );
        nbytes_ranks[diverge_mpi_comm_rank()] = nbytes;
        diverge_mpi_allgather_index_inplace( nbytes_ranks.data(), 1 );
        vector<index_t> sendcounts( diverge_mpi_comm_size(), 0 ),
                        recvcounts( diverge_mpi_comm_size(), 0 ),
                        senddispls( diverge_mpi_comm_size(), 0 ),
                        recvdispls( diverge_mpi_comm_size(), 0 );
        int to = (diverge_mpi_comm_rank() + 1) % diverge_mpi_comm_size(),
            from = (diverge_mpi_comm_rank() - 1 + diverge_mpi_comm_size()) % diverge_mpi_comm_size();
        char* other = (char*)calloc( nbytes_ranks[from], 1 );
        sendcounts[to] = nbytes;
        recvcounts[from] = nbytes_ranks[from];
        diverge_mpi_alltoallv_bytes( ptr, sendcounts.data(), senddispls.data(), other,
                recvcounts.data(), recvdispls.data(), 1 );
        free( ptr );

        // get the symmstruct and check for error
        internals_t* ii_from = syms_deserialize( model, other );
        free( other );
        int error = syms_equal( model, ii_from );
        free( ii_from->symm_orb_off );
        free( ii_from->symm_orb_len );
        free( ii_from->symm_beyond_UC );
        free( ii_from->symm_map_mom_fine );
        free( ii_from->symm_map_mom_crs );
        free( ii_from->symm_map_orb );
        free( ii_from->symm_pref );
        free( ii_from );

        // print if we found an error
        vector<int> errors( diverge_mpi_comm_size() );
        diverge_mpi_allgather_int( &error, errors.data(), 1 );
        int total_error = 0;
        for (int e: errors) total_error += e;
        if (total_error) {
            mpi_err_printf_all( "rank %i got error %i from symm-compare with rank %i (round-robin)\n", diverge_mpi_comm_rank(), error, from );
        } else {
            mpi_log_printf( "symm-compare check successful\n" );
        }
    }
}

static void syms_serialize( const diverge_model_t* model, char** ptr_, uindex_t* nbytes_ ) {
    const internals_t* ii = model->internals;

    index_t n_sym = model->n_sym;
    index_t n_os = model->n_orb * model->n_spin;
    index_t nk = kdim(model->nk);
    index_t nktot = kdimtot(model->nk, model->nkf);
    index_t n_orb = model->n_orb;

    index_t total_length = 0;
    for(index_t s = 0; s < n_sym; ++s)
    for(index_t o = 0; o < n_os; ++o) {
        total_length += ii->symm_orb_len[s * n_os + o];
    }

    char* ptr = NULL;
    index_t nbytes = 0;

#define push_array( ary, sz ) { \
    ptr = (char*) realloc( ptr, nbytes + sz ); \
    memcpy( ptr + nbytes, ary, sz ); \
    nbytes += sz; \
}

    push_array( ii->symm_orb_off, n_os * n_sym * sizeof(index_t) );
    push_array( ii->symm_orb_len, n_os * n_sym * sizeof(index_t) );
    push_array( ii->symm_beyond_UC, n_orb * n_sym * 3 * sizeof(double) );
    push_array( ii->symm_map_mom_fine, nktot * n_sym * sizeof(index_t) );
    push_array( ii->symm_map_mom_crs, nk * n_sym * sizeof(index_t) );
    push_array( ii->symm_map_orb, total_length * sizeof(index_t) );
    push_array( ii->symm_pref, total_length * sizeof(complex128_t) );

    *ptr_ = ptr;
    *nbytes_ = nbytes;
}

static internals_t* syms_deserialize( const diverge_model_t* model, char* ptr ) {
    internals_t* ii = (internals_t*)calloc( 1, sizeof(internals_t) );

    index_t n_sym = model->n_sym;
    index_t n_os = model->n_orb * model->n_spin;
    index_t nk = kdim(model->nk);
    index_t nktot = kdimtot(model->nk, model->nkf);
    index_t n_orb = model->n_orb;

#define pull_array( ary, sz, type ) { \
    ary = (type *) malloc( sz ); \
    memcpy( ary, ptr, sz ); \
    ptr += sz; \
}

    pull_array( ii->symm_orb_off, n_os * n_sym * sizeof(index_t), index_t );
    pull_array( ii->symm_orb_len, n_os * n_sym * sizeof(index_t), index_t );
    pull_array( ii->symm_beyond_UC, n_orb * n_sym * 3 * sizeof(double), double );
    pull_array( ii->symm_map_mom_fine, nktot * n_sym * sizeof(index_t), index_t );
    pull_array( ii->symm_map_mom_crs, nk * n_sym * sizeof(index_t), index_t );

    index_t total_length = 0;
    for(index_t s = 0; s < n_sym; ++s)
    for(index_t o = 0; o < n_os; ++o) {
        total_length += ii->symm_orb_len[s * n_os + o];
    }

    pull_array( ii->symm_map_orb, total_length * sizeof(index_t), index_t );
    pull_array( ii->symm_pref, total_length * sizeof(complex128_t), complex128_t );

    return ii;
}

static int compare_ary_i( const index_t* A, const index_t* B, index_t len ) {
    int errors = 0;
    for (index_t i=0; i<len; ++i)
        errors += A[i] != B[i];
    return errors;
}

static int compare_ary_c( const complex128_t* A, const complex128_t* B, index_t len ) {
    int errors = 0;
    for (index_t i=0; i<len; ++i)
        errors += std::abs(A[i] - B[i]) > 1.e-5;
    return errors;
}

static int compare_ary_d( const double* A, const double* B, index_t len ) {
    int errors = 0;
    for (index_t i=0; i<len; ++i)
        errors += std::abs(A[i] - B[i]) > 1.e-5;
    return errors;
}

static int syms_equal( const diverge_model_t* model, const internals_t* ii_B ) {
    const internals_t* ii_A = model->internals;

    index_t n_sym = model->n_sym;
    index_t n_os = model->n_orb * model->n_spin;
    index_t nk = kdim(model->nk);
    index_t nktot = kdimtot(model->nk, model->nkf);
    index_t n_orb = model->n_orb;

    index_t total_lengths[2];

    for (int i=0; i<2; ++i) {
        const internals_t* ii = i == 0 ? ii_A : ii_B;
        index_t total_length = 0;
        for(index_t s = 0; s < n_sym; ++s)
        for(index_t o = 0; o < n_os; ++o) {
            total_length += ii->symm_orb_len[s * n_os + o];
        }
        total_lengths[i] = total_length;
    }

    int errors = total_lengths[0] != total_lengths[1];
    if (errors) return errors;

#define compare( ary, typ, len ) \
    errors += compare_ary_##typ( ii_A->ary, ii_B->ary, len );

    compare( symm_orb_off, i, n_os * n_sym );
    compare( symm_orb_len, i, n_os * n_sym );
    compare( symm_beyond_UC, d, n_orb * n_sym * 3 );
    compare( symm_map_mom_fine, i, nktot * n_sym );
    compare( symm_map_mom_crs, i, nk * n_sym );
    compare( symm_map_orb, i, total_lengths[0] );
    compare( symm_pref, c, total_lengths[0] );

    return errors;
}
