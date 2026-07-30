/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_model.h"
#include "diverge_symmetrize.h"
#include "diverge_internals_struct.h"
#include "diverge_momentum_gen.h"
#include "misc/mpi_functions.h"
#include "misc/merge_orb_w_rs_symm.h"
#include "misc/bitary.h"
#include "misc/gemm.h"

#define LINALG_NO_LAPACKE
#include "misc/linalg.h"
LINALG_IMPLEMENT

// extern double SYMNORM_TU_BOND_MAX;
// extern double SYMNORM_MATRIX_ZERO;
#include "tu/diverge_interface.h"

#include <assert.h>

static double SYMNORM_ASSERT_POSITION_BEYOND = 1.e-4;
static double SYMNORM_ORBITAL_OVERLAP = 1.e-5;
static double SYMNORM_POS_EQUAL = 1.e-8;
static double SYMNORM_MIN_ORBITAL_WEIGHT = 0.99999;

void diverge_generate_symm_maps_set_precision( double ASSERT_POSITION_BEYOND,
        double ORBITAL_OVERLAP, double POS_EQUAL, double MIN_ORBITAL_WEIGHT,
        double TU_BOND_MAX, double MATRIX_ZERO ) {
    #define SET_TO( THING ) { if (THING > 0) { \
        SYMNORM_##THING = THING; \
        mpi_log_printf( "SYMNORM_" #THING " = %.5e\n", THING ); \
    } }
    SET_TO( ASSERT_POSITION_BEYOND );
    SET_TO( ORBITAL_OVERLAP );
    SET_TO( POS_EQUAL );
    SET_TO( MIN_ORBITAL_WEIGHT );
    SET_TO( TU_BOND_MAX );
    SET_TO( MATRIX_ZERO );
}

void diverge_generate_symm_maps_set_precision_global( double factor ) {
    diverge_generate_symm_maps_set_precision( factor, factor, factor, 1.-factor,
            factor, factor );
}

diverge_generate_symm_maps_precision_t diverge_generate_symm_maps_precision_get( void ) {
    return (diverge_generate_symm_maps_precision_t){
        .ASSERT_POSITION_BEYOND = SYMNORM_ASSERT_POSITION_BEYOND,
        .ORBITAL_OVERLAP = SYMNORM_ORBITAL_OVERLAP,
        .POS_EQUAL = SYMNORM_POS_EQUAL,
        .MIN_ORBITAL_WEIGHT = SYMNORM_MIN_ORBITAL_WEIGHT,
        .BOND_EQUAL = SYMNORM_TU_BOND_MAX,
        .MATRIX_ZERO = SYMNORM_MATRIX_ZERO,
    };
}

void diverge_generate_symm_maps_precision_set( diverge_generate_symm_maps_precision_t p ) {
    diverge_generate_symm_maps_set_precision( p.ASSERT_POSITION_BEYOND,
            p.ORBITAL_OVERLAP, p.POS_EQUAL, p.MIN_ORBITAL_WEIGHT, p.BOND_EQUAL,
            p.MATRIX_ZERO );
}

static inline bool map_back_to_BZ(claV3d_t* k_point, index_t* idx, const diverge_model_t* model, bool fine) {
    index_t nk[3]; memcpy( nk, model->nk, sizeof(nk) );
    if (fine) { for (int d=0; d<3; ++d) nk[d] *= model->nkf[d]; }

    double* mesh = fine ? model->internals->kfmesh : model->internals->kmesh;
    claM3d_t UC = claM3d_map( (double*)model->lattice[0] );

    claV3d_t a = claV3d_scale( claM3d_matvec( UC, *k_point ), 1./(2.*M_PI) );
    for (int i=0; i<3; ++i)
        a.v[i] += labs(lround(a.v[i])) + 3;
    index_t ai[3];
    for (int d=0; d<3; ++d) ai[d] = lround(a.v[d] * nk[d]);

    *idx = IDX3( ai[0]%nk[0], ai[1]%nk[1], ai[2]%nk[2], nk[1], nk[2] );
    *k_point = claV3d_map( mesh+3*(*idx) );
    return (*idx < kdim(nk));
}

// allocates *ptr using realloc, i.e., must be free'd by user with free.
static void syms_serialize( const diverge_model_t* model, char** ptr, uindex_t* nbytes );
// allocates internals_t* using malloc/calloc, must be free'd by user with free
static internals_t* syms_deserialize( const diverge_model_t* model, char* ptr );
// checks whether two internals_t objects are equal in their symmetry parts
static int syms_equal( const diverge_model_t* model, const internals_t* ii_B );

static index_t check_orbital_equivalence( diverge_model_t* m ) {
    const int periodic_iter[3] = {m->internals->periodic_direction[0]?3:0,
                                  m->internals->periodic_direction[1]?3:0,
                                  m->internals->periodic_direction[2]?3:0};
    const claM3d_t lat = claM3d_map(m->lattice[0]);

    // iterate over all orbs and find whether some are at equivalent positions
    index_t eqpos = 0;
    #pragma omp parallel for num_threads(diverge_omp_num_threads()) reduction(+:eqpos)
    for (index_t o=0; o<m->n_orb; ++o) {
        claV3d_t vo = claV3d_map(m->positions[o]);
        for (index_t op=0; op<m->n_orb; ++op) {
            claV3d_t vop = claV3d_map(m->positions[o]);
            double dist = claV3d_norm(claV3d_sub(vo,vop));
            if (dist > SYMNORM_POS_EQUAL) {
                for (int x=-periodic_iter[0]; x<=periodic_iter[0]; ++x)
                for (int y=-periodic_iter[0]; y<=periodic_iter[0]; ++y)
                for (int z=-periodic_iter[0]; z<=periodic_iter[0]; ++z) {
                    claV3d_t L = claM3d_vecmat( (claV3d_t){{x,y,z}}, lat );
                    double Ldist = claV3d_norm(claV3d_sub(vo, claV3d_add(L, vop)));
                    dist = MIN(dist, Ldist);
                }
                if (dist < SYMNORM_POS_EQUAL) {
                    eqpos++;
                    break;
                }
            }
        }
    }

    return eqpos;
}

static inline int approx_id( const double* A, index_t sz ) {
    double maxdiff = 0.0;
    for (index_t i=0; i<sz; ++i)
    for (index_t j=0; j<sz; ++j) {
        double diff = fabs(A[IDX2(i,j,sz)] - ((i==j)?1.0:0.0));
        maxdiff = MAX(diff,maxdiff);
    }
    // printf( "%.5e\n", maxdiff );
    return maxdiff < 1.e-3;
}
static inline int capprox_id( const complex128_t* A, index_t sz ) {
    double maxdiff = 0.0;
    for (index_t i=0; i<sz; ++i)
    for (index_t j=0; j<sz; ++j) {
        double diff = cabs( A[IDX2(i,j,sz)] - (((i==j)?1.0:0.0)) );
        maxdiff = MAX(diff,maxdiff);
    }
    // printf( "%.5e\n", maxdiff );
    return maxdiff < 1.e-3;
}

static diverge_ibz_t* find_ibz( diverge_model_t* m, int coarse ) {
    const index_t n_sym = m->n_sym;
    const index_t nk = coarse ? kdim(m->nk) : kdimtot(m->nk,m->nkf);

    const double* rs_sym = m->rs_symmetries[0][0];
    const complex128_t* orb_sym = m->orb_symmetries;
    index_t n_sorb = m->n_orb * m->n_spin;

    const index_t* mom_maps_to = coarse ? m->internals->symm_map_mom_crs :
                                          m->internals->symm_map_mom_fine;
    index_t* symm_inv = calloc(n_sym, sizeof*symm_inv);
    index_t errors = 0;
    index_t symm_identity = -1;
    for (index_t s=0; s<n_sym; ++s) {
        if (approx_id(rs_sym+s*POW2(3),3) && capprox_id(orb_sym+s*POW2(n_sorb),n_sorb))
            symm_identity = s;

        symm_inv[s] = -1;
        for (index_t t=0; t<n_sym; ++t) {
            complex128_t o_st[POW2(n_sorb)];
            single_square_gemm(orb_sym+s*POW2(n_sorb),0, orb_sym+t*POW2(n_sorb),0, o_st, n_sorb);
            claM3d_t r_st = claM3d_matmul(claM3d_map(rs_sym+s*POW2(3)), claM3d_map(rs_sym+t*POW2(3)));
            if (approx_id(r_st.m[0],3) && capprox_id(o_st,n_sorb)) {
                symm_inv[s] = t;
                break;
            }
        }
        if (symm_inv[s] == -1) {
            mpi_err_printf( "inverse for symmetry %li missing; ibz mapping won't work.\n", s );
            errors++;
        }
    }
    if (symm_identity == -1) {
        mpi_err_printf( "identity missing\n" );
        errors++;
    }

    if (errors) {
        free(symm_inv);
        return NULL;
    }

    cbytes_t* is_mapped_to_ibz = bitary_alloc(nk);
    diverge_ibz_t* ibz = calloc(1, sizeof*ibz);
    ibz->kmaps_to = calloc(nk, sizeof*ibz->kmaps_to);
    ibz->which_sym = calloc(nk, sizeof*ibz->which_sym*2);

    for (index_t k=0; k<nk; ++k) {
        if (!bitary_get(is_mapped_to_ibz, k)) {
            bitary_set(is_mapped_to_ibz, k);
            ibz->which_sym[k*2+0] = symm_identity;
            ibz->which_sym[k*2+1] = symm_identity;
            ibz->kmaps_to[k] = ibz->nkibz;
            for (index_t s=0; s<n_sym; ++s) {
                index_t idx = mom_maps_to[s+n_sym*k];
                if (!bitary_get(is_mapped_to_ibz, idx)) {
                    bitary_set(is_mapped_to_ibz, idx);
                    ibz->which_sym[idx*2+0] = s;
                    ibz->which_sym[idx*2+1] = symm_inv[s];
                    ibz->kmaps_to[idx] = ibz->nkibz;
                }
            }
            ibz->nkibz++;
        }
    }
    index_t nk_mapped = bitary_count_parallel(is_mapped_to_ibz, nk, diverge_omp_num_threads());
    if (nk_mapped != nk) {
        mpi_err_printf( "not all points mapped to ibz (%li vs. %li); errors?\n", nk_mapped, nk );
    }
    free(is_mapped_to_ibz);

    ibz->bz_map_off = calloc(ibz->nkibz, sizeof*ibz->bz_map_off);
    ibz->bz_map_len = calloc(ibz->nkibz, sizeof*ibz->bz_map_len);
    ibz->bz_map = calloc(nk, sizeof*ibz->bz_map*2);
    // count
    for (index_t k=0; k<nk; ++k) ibz->bz_map_len[ibz->kmaps_to[k]]++;
    // displ
    for (index_t i=1; i<ibz->nkibz; ++i)
        ibz->bz_map_off[i] = ibz->bz_map_len[i-1]+ibz->bz_map_off[i-1];
    // map (modifies displ)
    for (index_t k=0; k<nk; ++k) {
        claV2ll_t bz_map = {{ibz->kmaps_to[k], ibz->which_sym[IDX2(k,1,2)]}};
        ibz->bz_map[IDX2(ibz->bz_map_off[bz_map.v[0]],0,2)] = bz_map.v[0];
        ibz->bz_map[IDX2(ibz->bz_map_off[bz_map.v[0]],1,2)] = bz_map.v[1];
        ibz->bz_map_off[bz_map.v[0]]++;
    }
    // redo displ
    for (index_t i=1; i<ibz->nkibz; ++i)
        ibz->bz_map_off[i] = ibz->bz_map_len[i-1]+ibz->bz_map_off[i-1];

    ibz->nk = nk;
    return ibz;
}

void _diverge_ibz_free( diverge_ibz_t* ibz ) {
    if (ibz) {
        free(ibz->kmaps_to);
        free(ibz->which_sym);
        free(ibz->bz_map_off);
        free(ibz->bz_map_len);
        free(ibz->bz_map);
        free(ibz);
    }
}

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

    model->internals->symm_orb_off = calloc( n_os * n_sym, sizeof(index_t) );
    model->internals->symm_orb_len = calloc( n_os * n_sym, sizeof(index_t) );
    model->internals->symm_map_mom_fine = calloc( nktot * n_sym, sizeof(index_t) );
    model->internals->symm_map_mom_crs = calloc( nk * n_sym, sizeof(index_t) );
    model->internals->symm_beyond_UC = calloc( n_orb * n_sym, 3 * sizeof(double) );

    index_t* symm_orb_off = model->internals->symm_orb_off;
    index_t* symm_orb_len = model->internals->symm_orb_len;
    index_t* symm_map_mom_fine = model->internals->symm_map_mom_fine;
    index_t* symm_map_mom_crs = model->internals->symm_map_mom_crs;
    double* symm_beyond_UC = model->internals->symm_beyond_UC;

    double* rs_symmetries = model->rs_symmetries[0][0];
    complex128_t* orb_symmetries = model->orb_symmetries;

    mpi_vrb_printf( "finding coarse IBZ…\n" );
    #pragma omp parallel for schedule(static) collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t s=0; s<n_sym; ++s)
    for (index_t k=0; k<nk; ++k) {
        claM3d_t sym = claM3d_map( rs_symmetries + s*3*3 );
        claV3d_t kpnt = claV3d_map( kmesh + k*3 );
        claV3d_t tmp = claM3d_matvec( sym, kpnt );
        index_t idx = 0; bool found = map_back_to_BZ(&tmp, &idx, model, false);
        symm_map_mom_crs[s+n_sym*k] = idx;
        if (!found) {
            #pragma omp critical
            mpi_err_printf("not all momenta mappable (%d->%d; accuracy issue?)\n", s, k);
        }
    }

    mpi_vrb_printf( "finding fine IBZ…\n" );
    #pragma omp parallel for schedule(static) collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t s=0; s<n_sym; ++s)
    for (index_t k=0; k<nktot; ++k) {
        claM3d_t sym = claM3d_map( rs_symmetries + s*3*3 );
        claV3d_t kpnt = claV3d_map( kfmesh + k*3 );
        claV3d_t tmp = claM3d_matvec( sym, kpnt );
        index_t idx = 0; bool found = map_back_to_BZ(&tmp, &idx, model, true);
        symm_map_mom_fine[s+n_sym*k] = idx;
        if (!found) {
            #pragma omp critical
            mpi_err_printf("not all momenta mappable (%d->%d; accuracy issue?)\n", s, k);
        }
    }

    mpi_vrb_printf( "finding orbital maps…\n" );
    index_t* orbmaps_to_len = calloc(n_os * n_sym, sizeof(index_t));
    index_t* orbmaps_to_off = calloc(n_os * n_sym, sizeof(index_t));
    VECTOR_INIT( index_t, orbmaps_to )
    VECTOR_INIT( complex128_t, orbmaps_to_prefac )
    claV3d_t* mapped_pos = calloc(n_orb*n_sym, sizeof(claV3d_t));

    int periodic_iter[3];
    periodic_iter[0] = model->internals->periodic_direction[0] ? 3:0;
    periodic_iter[1] = model->internals->periodic_direction[1] ? 3:0;
    periodic_iter[2] = model->internals->periodic_direction[2] ? 3:0;

    claV3d_t UC1 = claV3d_map(model->lattice[0]);
    claV3d_t UC2 = claV3d_map(model->lattice[1]);
    claV3d_t UC3 = claV3d_map(model->lattice[2]);

    index_t eqpos = check_orbital_equivalence( model );
    if (eqpos)
        mpi_err_printf( "%li orbitals share *periodic* position, but not *absolute*\n", eqpos );

    #ifndef ASAN_COMPILATION
    _Pragma("omp parallel for collapse(2) num_threads(diverge_omp_num_threads())")
    #endif
    for (index_t s=0; s<n_sym; ++s)
    for (index_t o1=0; o1<n_orb; ++o1) {
        claM3d_t sym_rs = claM3d_map(rs_symmetries + s*3*3);
        claV3d_t from = claV3d_map(model->positions[o1]),
                 bey_UC = claV3d_map(symm_beyond_UC+3*(s+n_sym*o1));
        claV3d_t to = claM3d_matvec(sym_rs, from);
        index_t maps_to = -1;
        for (index_t o2=0; o2<n_orb; ++o2)
        for (int in=-periodic_iter[0]; in<=periodic_iter[0]; in++)
        for (int im=-periodic_iter[1]; im<=periodic_iter[1]; im++)
        for (int iq=-periodic_iter[2]; iq<=periodic_iter[2]; iq++) {
            claV3d_t to_trial = claV3d_map(model->positions[o2]);
            claV3d_t UC123 = claV3d_add( claV3d_add(
                                claV3d_scale(UC1,in),
                                claV3d_scale(UC2,im) ),
                                claV3d_scale(UC3,iq) );
            claV3d_t trial = claV3d_add(to_trial, UC123);
            if (claV3d_norm(claV3d_sub(trial,to)) < SYMNORM_POS_EQUAL) {
                maps_to = o2;
                mapped_pos[o1+n_orb*s] = to_trial;
                bey_UC = claV3d_sub( to, to_trial );
                claV3d_rmap( bey_UC, symm_beyond_UC + 3*(s+n_sym*o1) );
                assert( claV3d_norm(claV3d_sub(bey_UC, UC123)) < SYMNORM_ASSERT_POSITION_BEYOND );
                goto symmetrizer_orb_found;
            }
        }
        if (maps_to == -1) {
            #pragma omp critical
            {
                mpi_wrn_printf("position %li doesn't map under symmetry %li!\n", o1, s);
                model->n_sym = 0;
            }
        }
        symmetrizer_orb_found:
        {}
    }

    // generate orbmaps, try to find each orbital that has overlap with the
    // current one *and* maps to its position
    for (index_t s=0; s<n_sym; ++s)
    for (index_t s1=0; s1<n_spin; ++s1)
    for (index_t o1=0; o1<n_orb; ++o1) {
        index_t len = 0;
        for (index_t s1t=0; s1t<n_spin; ++s1t)
        for (index_t o1t=0; o1t<n_orb; ++o1t) {
            claV3d_t pos = claV3d_map(model->positions[o1t]);
            complex128_t overlap = orb_symmetries[IDX5(s, s1t, o1t, s1, o1, n_spin,n_orb, n_spin,n_orb)];
            if ((cabs(overlap) > SYMNORM_ORBITAL_OVERLAP) &&
                (claV3d_norm(claV3d_sub(mapped_pos[o1+n_orb*s], pos)) < SYMNORM_POS_EQUAL)) {
                VECTOR_PUSH_BACK( orbmaps_to, IDX2(s1t,o1t,n_orb) );
                VECTOR_PUSH_BACK( orbmaps_to_prefac, overlap );
                len++;
            }
        }
        if (len == 0) {
            mpi_wrn_printf("orbital %li doesn't map under symmetry %li!\n", o1, s);
            model->n_sym = 0;
        }
        orbmaps_to_len[IDX3(s,s1,o1,n_spin,n_orb)] = len;
    }
    // offsets
    orbmaps_to_off[0] = 0;
    for (index_t i=1; i<n_os*n_sym; ++i) {
        orbmaps_to_off[i] = orbmaps_to_off[i-1] + orbmaps_to_len[i-1];
    }
    // verbose info
    for (index_t i=0; i<n_os*n_sym; ++i) {
        if (orbmaps_to_len[i]>1) {
            mpi_vrb_printf("Symmetries map one to many\n");
            break;
        }
    }
    // and checking
    for (index_t o1=0; o1<n_os*n_sym; ++o1) {
        if (orbmaps_to_len[o1] == 0) {
            mpi_err_printf("orbital maps not constructed correctly, there is an empty one\n");
            model->n_sym = 0;
        }
        double sum = 0.0;
        for (index_t e=0; e<orbmaps_to_len[o1]; ++e)
            sum += cabs(POW2(orbmaps_to_prefac[orbmaps_to_off[o1]+e]));
        /* printf( "o1=%li, len=%li, off=%li, sum=%.5e\n", o1, orbmaps_to_len[o1], orbmaps_to_off[o1], sum ); */
        if (sum < SYMNORM_MIN_ORBITAL_WEIGHT) {
            mpi_err_printf("Orbitals do not sum up to one %1.3f\n", sum);
            model->n_sym = 0;
        }
    }

    // calc total length and copy data over
    index_t total_length = 0;
    for (index_t s=0; s<n_sym; ++s)
    for (index_t o1=0; o1<n_os; ++o1) {
        symm_orb_off[IDX2(o1,s,n_sym)] = orbmaps_to_off[IDX2(s,o1,n_os)];
        symm_orb_len[IDX2(o1,s,n_sym)] = orbmaps_to_len[IDX2(s,o1,n_os)];
        total_length += orbmaps_to_len[IDX2(s,o1,n_os)];
    }
    model->internals->symm_map_orb = calloc( total_length, sizeof(index_t) );
    model->internals->symm_pref = calloc( total_length, sizeof(complex128_t) );

    index_t* symm_map_orb = model->internals->symm_map_orb;
    complex128_t* symm_pref = model->internals->symm_pref;

    memcpy(symm_map_orb, orbmaps_to, total_length * sizeof(index_t));
    memcpy(symm_pref, orbmaps_to_prefac, total_length * sizeof(complex128_t));

    // free dynamic arrays
    free( orbmaps_to_len );
    free( orbmaps_to_off );
    free( orbmaps_to );
    free( orbmaps_to_prefac );
    free( mapped_pos );

    mpi_vrb_printf( "merging realspace orbitals…\n" );
    merge_rs_orb(model);

    model->internals->ibz_crs = find_ibz(model, 1);
    model->internals->ibz_fine = find_ibz(model, 0);

    if (model->internals->symcheck_mpi) {
        // serialization and communication
        char* ptr = NULL;
        uindex_t nbytes = 0;
        syms_serialize( model, &ptr, &nbytes );
        index_t* nbytes_ranks = calloc( diverge_mpi_comm_size(), sizeof(index_t) );
        nbytes_ranks[diverge_mpi_comm_rank()] = nbytes;
        diverge_mpi_allgather_index_inplace( nbytes_ranks, 1 );
        index_t *sendcounts = calloc( diverge_mpi_comm_size(), sizeof(index_t) ),
                *recvcounts = calloc( diverge_mpi_comm_size(), sizeof(index_t) ),
                *senddispls = calloc( diverge_mpi_comm_size(), sizeof(index_t) ),
                *recvdispls = calloc( diverge_mpi_comm_size(), sizeof(index_t) );
        int to = (diverge_mpi_comm_rank() + 1) % diverge_mpi_comm_size(),
            from = (diverge_mpi_comm_rank() - 1 + diverge_mpi_comm_size()) % diverge_mpi_comm_size();
        char* other = (char*)calloc( nbytes_ranks[from], 1 );
        sendcounts[to] = nbytes;
        recvcounts[from] = nbytes_ranks[from];
        diverge_mpi_alltoallv_bytes( ptr, sendcounts, senddispls, other, recvcounts, recvdispls, 1 );
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
        int* errors = calloc( diverge_mpi_comm_size(), sizeof(int) );
        diverge_mpi_allgather_int( &error, errors, 1 );
        int total_error = 0;
        for (int ie=0; ie<diverge_mpi_comm_size(); ++ie) total_error += errors[ie];
        free(errors);
        if (total_error) {
            mpi_err_printf_all( "rank %i got error %i from symm-compare with rank %i (round-robin)\n", diverge_mpi_comm_rank(), error, from );
        } else {
            mpi_log_printf( "symm-compare check successful\n" );
        }

        free(nbytes_ranks);
        free(sendcounts);
        free(senddispls);
        free(recvcounts);
        free(recvdispls);
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
        ptr = realloc( ptr, nbytes + sz ); \
        if (sz > 0) { memcpy( ptr + nbytes, ary, sz ); } \
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
        if (sz > 0) { memcpy( ary, ptr, sz ); } \
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
        errors += cabs(A[i] - B[i]) > 1.e-5;
    return errors;
}

static int compare_ary_d( const double* A, const double* B, index_t len ) {
    int errors = 0;
    for (index_t i=0; i<len; ++i)
        errors += fabs(A[i] - B[i]) > 1.e-5;
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
