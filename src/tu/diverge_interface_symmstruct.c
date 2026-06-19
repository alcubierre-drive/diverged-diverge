/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#define LINALG_NO_LAPACKE
#include "../misc/linalg.h"
LINALG_IMPLEMENT

double SYMNORM_TU_BOND_MAX = 1.e-5;
double SYMNORM_MATRIX_ZERO = 1.e-7;

#include "diverge_interface_symmstruct.h"
#include "../diverge_internals_struct.h"
#include "../diverge_momentum_gen.h"
#include "../diverge_symmetrize.h"
#include "../misc/bitary.h"
#include "../misc/invert.h"
#include "../misc/gemm.h"
#include "../misc/mpi_functions.h"

// this must follow the LINALG include to pre-define the types
#include "diverge_interface.h"

static void symmstruct_serialize( const tu_symmstruct_t* symm, char** ptr, uindex_t* nbytes );
static tu_symmstruct_t* symmstruct_deserialize( const char* ptr );
static int symmstruct_equal( const tu_symmstruct_t* A, const tu_symmstruct_t* B );

void tu_symmstruct_delete( tu_symmstruct_t* symm ) {
    free( symm->inv_symmetry );
    free( symm->mi_prefac_in_out );
    free( symm->mi_prefac_in_in );
    free( symm->mi_maps_to );
    free( symm->mi_map_len );
    free( symm->mi_map_off );
    free( symm->beyond_UC );
    free( symm->kmaps_to );
    free( symm->idx_ibz_in_fullmesh );
    free( symm->which_symmetry_used );
    free( symm->prefactor_symmP );
    free( symm->prefactor_symmCD );
    free( symm->idx_map_symm );
    free( symm->o2m_map_len );
    free( symm->o2m_map_off );
    free( symm->from_ibz_2_bz );
    free( symm->from_ibz_2_bz_off );
    free( symm->from_ibz_2_bz_len );
    free( symm );
}

static inline int approx_id( const double* A, index_t sz );
static inline int capprox_id( const complex128_t* A, index_t sz );

static bool init_inv_symmerty_map( diverge_model_t* model ) {
    tu_data_t* tu_data = model->internals->tu_data;
    tu_symmstruct_t* symm = tu_data->symm;
    complex128_t* orb_symmetries = model->orb_symmetries;
    double* rs_symmetries = (double*)model->rs_symmetries;
    index_t n_orb = model->n_orb;
    index_t n_spin = model->n_spin;
    index_t n_sym = model->n_sym;
    symm->inv_symmetry = calloc(n_sym, sizeof(index_t));
    index_t mi = n_orb * n_spin;

    VECTOR_INIT( index_t, missing_symmetries );
    bool added_symm = false;
    index_t which_is_identity = -1;
    for (index_t sym = 0; sym < n_sym; ++sym) {
        if (approx_id(rs_symmetries+IDX2(sym,0,9), 3) && capprox_id(orb_symmetries+IDX2(sym,0,mi*mi), mi))
            which_is_identity = sym;
        bool found = false;
        index_t symt = 0;
        while (symt < n_sym && !found) {
            claM3d_t rt = claM3d_matmul(claM3d_map( rs_symmetries+IDX2(sym,0,9) ),
                                        claM3d_map( rs_symmetries+IDX2(symt,0,9) ));
            complex128_t ot[mi*mi];
            single_square_gemm( orb_symmetries+IDX2(sym,0,mi*mi),0, orb_symmetries+IDX2(symt,0,mi*mi),0, ot, mi );

            if (approx_id(rt.m[0], 3) && capprox_id(ot, mi)) {
                symm->inv_symmetry[sym] = symt;
                found = true;
            }
            symt++;
        }
        if (!found) VECTOR_PUSH_BACK( missing_symmetries, sym );
    }

    if (missing_symmetries_sz != 0) {
        added_symm = true;
        index_t n_symold = n_sym;

        // change n_sym and realloc orbital symmetries
        n_sym = (model->n_sym += missing_symmetries_sz);

        orb_symmetries = model->orb_symmetries = realloc(orb_symmetries, sizeof(complex128_t)*n_sym*POW2(mi));
        symm->inv_symmetry = realloc(symm->inv_symmetry, sizeof(index_t)*n_sym);

        for (index_t i=0; i<missing_symmetries_sz; ++i) {
            index_t sym = missing_symmetries[i];
            mpi_wrn_printf("adding inverse for symmetry #%li\n", sym);
            matrix_invert( rs_symmetries+IDX2(sym,0,9), 3, rs_symmetries+IDX2(n_symold+i,0,9) );
            cmatrix_invert( orb_symmetries+IDX2(sym,0,mi*mi), mi, orb_symmetries+IDX2(n_symold+i,0,mi*mi) );
            symm->inv_symmetry[sym] = n_symold + i;
            symm->inv_symmetry[n_symold + i] = sym;
        }
        mpi_wrn_printf("increased number of symmetries from %li to %li\n", n_symold, n_sym);
    }
    VECTOR_FREE( missing_symmetries );

    if (which_is_identity == -1) {
        mpi_wrn_printf("identity missing, will be added now\n");
        added_symm = true;

        n_sym++;
        orb_symmetries = model->orb_symmetries = realloc(orb_symmetries, sizeof(complex128_t)*n_sym*POW2(mi));
        symm->inv_symmetry = realloc(symm->inv_symmetry, sizeof(index_t)*n_sym);

        // set last (new) element to identity
        for (index_t i=0; i<mi; ++i) for (index_t j=0; j<mi; ++j)
            orb_symmetries[IDX3(n_sym-1,i,j,mi,mi)] = (i==j) ? 1.0 : 0.0;
        for (int i=0; i<3; ++i) for (int j=0; j<3; ++j)
            rs_symmetries[IDX3(n_sym-1,i,j,3,3)] = (i==j) ? 1.0 : 0.0;

        symm->inv_symmetry[n_sym-1] = n_sym-1;
    }
    symm->n_sym = n_sym;
    return added_symm;
}

static void init_ibz( diverge_model_t* model ) {
    tu_data_t* tu_data = model->internals->tu_data;
    tu_symmstruct_t* symm = tu_data->symm;
    index_t n_sym = model->n_sym;
    index_t nk = kdim(model->nk);

    index_t* mom_maps_to = model->internals->symm_map_mom_crs;
    cbytes_t* is_mapped_to_ibz = bitary_alloc(nk);

    complex128_t* orb_symmetries = model->orb_symmetries;
    double* rs_symmetries = (double*)model->rs_symmetries;
    index_t n_orb = model->n_orb;
    index_t n_spin = model->n_spin;
    index_t mi = n_orb * n_spin;
    index_t which_is_identity = -1;
    for (index_t sym=0; sym<n_sym; ++sym) {
        if (approx_id(rs_symmetries+IDX2(sym,0,9),3) && capprox_id(orb_symmetries+IDX2(sym,0,mi*mi),mi)) {
            which_is_identity = sym;
            break;
        }
    }
    if (which_is_identity < 0)
        mpi_err_printf("identity not in symmetries, init_ibz will fail\n");

    index_t current_ibz_idx = 0;
    symm->kmaps_to = calloc(nk, sizeof(index_t));
    symm->which_symmetry_used = calloc(nk, sizeof(tu_whichsym_t));
    symm->nk = nk;

    VECTOR_INIT( index_t, idx_ibz_in_fullmesh );
    for (index_t ii=0; ii<nk; ++ii) {
        if (!bitary_get( is_mapped_to_ibz, ii )) {
            VECTOR_PUSH_BACK( idx_ibz_in_fullmesh, ii );
            bitary_set( is_mapped_to_ibz, ii );
            symm->which_symmetry_used[ii] = (tu_whichsym_t){which_is_identity, which_is_identity};
            symm->kmaps_to[ii] = current_ibz_idx;
            for (index_t s=0; s<n_sym; ++s) {
                index_t idx = mom_maps_to[s+n_sym*ii];
                if (!bitary_get( is_mapped_to_ibz, idx )) {
                    bitary_set( is_mapped_to_ibz, idx );
                    symm->which_symmetry_used[idx] = (tu_whichsym_t){s, symm->inv_symmetry[s]};
                    symm->kmaps_to[idx] = current_ibz_idx;
                }
            }
            ++current_ibz_idx;
        }
    }
    // Test that all k-points have an IBZ map
    for (index_t i=0; i<nk; ++i) {
        if (!bitary_get( is_mapped_to_ibz, i ))
            mpi_wrn_printf("No IBZ-BZ-map at idx: %li\n", i);
    }
    free(is_mapped_to_ibz);

    VECTOR_SHRINK_TO_FIT( idx_ibz_in_fullmesh );
    symm->idx_ibz_in_fullmesh = idx_ibz_in_fullmesh;
    tu_data->nkibz = symm->nkibz = idx_ibz_in_fullmesh_sz;
    mpi_vrb_printf("#k in ibz: %li\n", tu_data->nkibz);

    // only necessary for MPI
    // gives for each IBZ point which points it corresponds to beyond the
    // IBZ, again in a linearized fashion
    symm->from_ibz_2_bz_off = calloc(tu_data->nkibz, sizeof(index_t));
    symm->from_ibz_2_bz_len = calloc(tu_data->nkibz, sizeof(index_t));

    // generate counts
    for (index_t i=0; i<nk; ++i)
        symm->from_ibz_2_bz_len[symm->kmaps_to[i]]++;

    // generate displs from counts
    for (index_t i=1; i<tu_data->nkibz; ++i)
        symm->from_ibz_2_bz_off[i] = symm->from_ibz_2_bz_off[i-1] + symm->from_ibz_2_bz_len[i-1];

    // allocate memory
    symm->from_ibz_2_bz = calloc(symm->from_ibz_2_bz_off[tu_data->nkibz-1] +
                                 symm->from_ibz_2_bz_len[tu_data->nkibz-1], sizeof(index_t));

    // reset counts
    memset( symm->from_ibz_2_bz_len, 0, sizeof(index_t)*tu_data->nkibz );

    // and push elements
    for (index_t i=0; i<nk; ++i)
        symm->from_ibz_2_bz[symm->from_ibz_2_bz_off[symm->kmaps_to[i]] +
                            symm->from_ibz_2_bz_len[symm->kmaps_to[i]]++] = i;

    // the MPI stuff
    init_MPI_dist( model );
    symm->my_nk_in_bz = 0;
    symm->my_nk_in_bz_off = symm->from_ibz_2_bz_off[tu_data->my_nk_off];
    for (index_t ibz=0; ibz<tu_data->my_nk; ++ibz)
        symm->my_nk_in_bz += symm->from_ibz_2_bz_len[ibz+tu_data->my_nk_off];
}


static void init_map_for_one_leg( diverge_model_t* model ) {
    tu_data_t* tu_data = model->internals->tu_data;
    tu_symmstruct_t* symm = tu_data->symm;
    index_t* bond_sizes = tu_data->bond_sizes;
    index_t* bond_offsets = tu_data->bond_offsets;
    tu_formfactor_t* tu_ff = model->tu_ff;

    double* rs_symmetries = (double*)model->rs_symmetries;
    index_t n_orb = model->n_orb;
    index_t n_spin = model->n_spin;
    index_t n_sym = model->n_sym;
    index_t n_orbff = tu_data->n_orbff;

    index_t mi_cnt = symm->mi_cnt = tu_data->n_orbff*POW2(n_spin)*n_sym;
    symm->mi_map_len = calloc( mi_cnt, sizeof(index_t) );
    symm->mi_map_off = calloc( mi_cnt, sizeof(index_t) );
    index_t res = tu_data->n_orbff*POW2(n_spin)*n_sym*2;
    VECTOR_RESERVE( symm->mi_prefac_in_out, res );
    VECTOR_RESERVE( symm->mi_prefac_in_in, res );
    VECTOR_RESERVE( symm->mi_maps_to, res );
    VECTOR_RESERVE( symm->beyond_UC, res );
    index_t* orbmaps_to_len = model->internals->symm_orb_len;
    index_t* orbmaps_to_off = model->internals->symm_orb_off;
    index_t* orbmaps_to = model->internals->symm_map_orb;
    complex128_t* orbmaps_to_prefac = model->internals->symm_pref;
    double* beyond_UC_in = model->internals->symm_beyond_UC;
    int periodic_iter[3];
    periodic_iter[0] = model->internals->periodic_direction[0] ? 3:0;
    periodic_iter[1] = model->internals->periodic_direction[1] ? 3:0;
    periodic_iter[2] = model->internals->periodic_direction[2] ? 3:0;

    claV3d_t UC1 = claV3d_map(model->lattice[0]),
             UC2 = claV3d_map(model->lattice[1]),
             UC3 = claV3d_map(model->lattice[2]);

    for (index_t s=0; s<n_sym; ++s)
    for (index_t s2=0; s2<n_spin; ++s2)
    for (index_t s1=0; s1<n_spin; ++s1)
    for (index_t mo1=0; mo1<n_orbff; ++mo1) {
        index_t o1 = tu_data->mi_to_ofrom[mo1];
        index_t o2 = tu_data->mi_to_oto[mo1];
        index_t so1 = IDX2(s1,o1,n_orb);
        index_t so2 = IDX2(s2,o2,n_orb);
        claM3d_t sym_rs = claM3d_map(rs_symmetries+IDX2(s,0,9));
        claV3d_t po1 = claV3d_map(model->positions[o1]),
                 po2 = claV3d_map(model->positions[o2]);
        claV3d_t _from = claV3d_add(claV3d_add(claV3d_add(
                        claV3d_scale(UC1,tu_data->mi_to_R[3*mo1+0]),
                        claV3d_scale(UC2,tu_data->mi_to_R[3*mo1+1])),
                        claV3d_scale(UC3,tu_data->mi_to_R[3*mo1+2])),
                        claV3d_sub(po1, po2));
        claV3d_t from = claM3d_matvec(sym_rs, _from);

        index_t found = 0;
        for (index_t id1=0; id1<orbmaps_to_len[so1*n_sym+s]; ++id1)
        for (index_t id2=0; id2<orbmaps_to_len[so2*n_sym+s]; ++id2) {
            index_t symap_so1 = orbmaps_to[id1+orbmaps_to_off[so1*n_sym+s]];
            complex128_t maps_to_pref_o1 =
                    orbmaps_to_prefac[id1+orbmaps_to_off[so1*n_sym+s]];
            index_t symap_so2 = orbmaps_to[id2+orbmaps_to_off[so2*n_sym+s]];
            complex128_t maps_to_pref_o2 =
                    orbmaps_to_prefac[id2+orbmaps_to_off[so2*n_sym+s]];
            index_t symap_s1 = symap_so1/n_orb;
            index_t symap_s2 = symap_so2/n_orb;
            index_t symap_o1 = symap_so1%n_orb;
            index_t symap_o2 = symap_so2%n_orb;
            for (index_t mb1=0; mb1<bond_sizes[symap_o1]; ++mb1)
            if (tu_ff[bond_offsets[symap_o1]+mb1].oto==symap_o2) {
                claV3d_t mpo1 = claV3d_map(model->positions[symap_o1]),
                         mpo2 = claV3d_map(model->positions[symap_o2]);
                index_t mi_map = tu_data->ob_to_orbff[mb1 + tu_data->n_bonds * symap_o1];
                claV3d_t ffc = claV3d_add(claV3d_add(claV3d_add(
                            claV3d_scale(UC1,tu_data->mi_to_R[3*mi_map+0]),
                            claV3d_scale(UC2,tu_data->mi_to_R[3*mi_map+1])),
                            claV3d_scale(UC3,tu_data->mi_to_R[3*mi_map+2])),
                            claV3d_sub(mpo1, mpo2));
                bool mapped_to = false;
                for (int in=-periodic_iter[0]; in<=periodic_iter[0]; in++)
                for (int im=-periodic_iter[1]; im<=periodic_iter[1]; im++)
                for (int iq=-periodic_iter[2]; iq<=periodic_iter[2]; iq++) {
                    claV3d_t trial = claV3d_sub(ffc, from);
                    trial = claV3d_add(trial,claV3d_scale(UC1,model->nk[0]*in));
                    trial = claV3d_add(trial,claV3d_scale(UC2,model->nk[1]*im));
                    trial = claV3d_add(trial,claV3d_scale(UC3,model->nk[2]*iq));
                    if (claV3d_norm(trial) < SYMNORM_TU_BOND_MAX)
                        mapped_to = true;
                }

                if (mapped_to) {
                    found++;
                    VECTOR_PUSH_BACK( symm->mi_maps_to, mi_map+tu_data->n_orbff*(symap_s1+n_spin*symap_s2) );
                    VECTOR_PUSH_BACK( symm->mi_prefac_in_in, maps_to_pref_o1*maps_to_pref_o2 );
                    VECTOR_PUSH_BACK( symm->mi_prefac_in_out, maps_to_pref_o1*conj(maps_to_pref_o2) );
                    for (index_t d=0; d<3; ++d)
                        VECTOR_PUSH_BACK( symm->beyond_UC, beyond_UC_in[IDX2(s+n_sym*o2,d,3)] );
                }
            }
        }
        index_t mi_idx = mo1+tu_data->n_orbff*(s1 + n_spin*(s2+n_spin*s));
        symm->mi_map_len[mi_idx] = found;
        if (found == 0)
            mpi_wrn_printf("bond not symmapped: midx%d, s%d\n", mo1, s);
    }

    symm->mi_map_off[0] = 0;
    for (index_t i=1; i<mi_cnt; ++i) {
        symm->mi_map_off[i] = symm->mi_map_len[i-1] + symm->mi_map_off[i-1];
    }
    VECTOR_SHRINK_TO_FIT( symm->mi_prefac_in_in );
    VECTOR_SHRINK_TO_FIT( symm->mi_prefac_in_out );
    VECTOR_SHRINK_TO_FIT( symm->mi_maps_to );
}

static void init_symmetry_maps_o2m(diverge_model_t* model, const double* beyond_UC) {
    mpi_vrb_printf("Generate one-to-many maps\n");
    tu_data_t* tu_data = model->internals->tu_data;
    tu_symmstruct_t* symm = tu_data->symm;
    index_t n_spin = model->n_spin;
    index_t my_nk_in_bz = symm->my_nk_in_bz;
    index_t my_nk_in_bz_off = symm->my_nk_in_bz_off;
    index_t n_orbff = tu_data->n_orbff;
    double* rs_symmetries = (double*)model->rs_symmetries;

    VECTOR_RESERVE( symm->prefactor_symmP, my_nk_in_bz*(n_orbff)*POW2(n_spin) );
    VECTOR_RESERVE( symm->prefactor_symmCD, my_nk_in_bz*(n_orbff)*POW2(n_spin) );
    VECTOR_RESERVE( symm->idx_map_symm, my_nk_in_bz*(n_orbff)*POW2(n_spin) );

    symm->o2m_map_sz = POW2(n_spin)*n_orbff*my_nk_in_bz;
    symm->o2m_map_off = calloc(POW2(n_spin)*n_orbff*my_nk_in_bz, sizeof(index_t));
    symm->o2m_map_len = calloc(POW2(n_spin)*n_orbff*my_nk_in_bz, sizeof(index_t));
    for (index_t s2=0; s2<n_spin;++s2)
    for (index_t s1=0; s1<n_spin;++s1)
    for (index_t mi1=0; mi1<n_orbff;++mi1)
    for (index_t q=0; q<my_nk_in_bz; ++q) {
        const index_t q_in_full = symm->from_ibz_2_bz[q + my_nk_in_bz_off];
        const index_t which_sym = symm->which_symmetry_used[q_in_full].isym;
        /* assert(symm->kmaps_to[q_in_full] - tu_data->my_nk_off < tu_data->my_nk); */
        claM3d_t sym_rs = claM3d_map(rs_symmetries+symm->which_symmetry_used[q_in_full].sym*9);

        index_t idx = mi1 + n_orbff*(s1 + n_spin*(s2+n_spin*which_sym));
        claV3d_t kp = claV3d_map(model->internals->kmesh+3*q_in_full);
        const complex128_t pref_sym = cexp( -I*claV3d_dot(kp,claM3d_matvec(sym_rs,claV3d_map(beyond_UC+3*idx))) );

        for (index_t to=0; to<symm->mi_map_len[idx]; ++to) {
            VECTOR_PUSH_BACK( symm->idx_map_symm, symm->mi_maps_to[to + symm->mi_map_off[idx]] );

            complex128_t pref_elem = symm->mi_prefac_in_in[to + symm->mi_map_off[idx]];
            VECTOR_PUSH_BACK( symm->prefactor_symmP, pref_elem*pref_sym );

            pref_elem = symm->mi_prefac_in_out[to + symm->mi_map_off[idx]];
            VECTOR_PUSH_BACK( symm->prefactor_symmCD, pref_elem*pref_sym );
        }
        symm->o2m_map_len[IDX4(s2,s1,mi1,q,n_spin,n_orbff,my_nk_in_bz)]
            = symm->mi_map_len[idx];
    }
    symm->o2m_map_off[0] = 0;
    for (index_t i=1; i<POW2(n_spin)*n_orbff*my_nk_in_bz; ++i)
        symm->o2m_map_off[i] = symm->o2m_map_off[i-1]+symm->o2m_map_len[i-1];
    VECTOR_SHRINK_TO_FIT( symm->prefactor_symmCD );
    VECTOR_SHRINK_TO_FIT( symm->prefactor_symmP );
    VECTOR_SHRINK_TO_FIT( symm->idx_map_symm );
}

void tu_init_symmetries( diverge_model_t* model ) {
    tu_data_t* tu_data = model->internals->tu_data;
    tu_symmstruct_t* symm = tu_data->symm;
    if (model->n_sym == 0) {
        symm->use_symmetries = false;
        tu_data->nkibz = tu_data->nk;
        init_MPI_dist( model );
        symm->my_nk_in_bz = tu_data->my_nk;
        symm->my_nk_in_bz_off = tu_data->my_nk_off;
        symm->from_ibz_2_bz = calloc(tu_data->nk, sizeof(index_t));
        symm->idx_ibz_in_fullmesh = calloc(tu_data->nk, sizeof(index_t));
        for (index_t i=0; i<tu_data->nk; ++i) symm->from_ibz_2_bz[i] = i;
        for (index_t i=0; i<tu_data->nk; ++i) symm->idx_ibz_in_fullmesh[i] = i;
        return;
    }
    symm->use_symmetries = true;
    bool added_symm = init_inv_symmerty_map( model );
    if (added_symm) {
        free(model->internals->symm_map_mom_crs);
        free(model->internals->symm_map_mom_fine);
        free(model->internals->symm_beyond_UC);
        free(model->internals->symm_orb_off);
        free(model->internals->symm_orb_len);
        free(model->internals->symm_map_orb);
        free(model->internals->symm_pref);

        model->internals->symm_map_mom_crs = NULL;
        model->internals->symm_map_mom_fine = NULL;
        model->internals->symm_beyond_UC = NULL;
        model->internals->symm_orb_off = NULL;
        model->internals->symm_orb_len = NULL;
        model->internals->symm_map_orb = NULL;
        model->internals->symm_pref = NULL;
    }
    diverge_generate_symm_maps( model );
    init_ibz( model );
    init_map_for_one_leg( model );
    init_symmetry_maps_o2m( model, symm->beyond_UC );

    if (model->internals->tu_symcheck_mpi) {
        // serialization and communication
        char* ptr = NULL;
        uindex_t nbytes = 0;
        symmstruct_serialize( symm, &ptr, &nbytes );
        index_t* nbytes_ranks = calloc(diverge_mpi_comm_size(), sizeof*nbytes_ranks);
        nbytes_ranks[diverge_mpi_comm_rank()] = nbytes;
        diverge_mpi_allgather_index_inplace( nbytes_ranks, 1 );
        index_t *sendcounts = calloc(diverge_mpi_comm_size(),sizeof*sendcounts),
                *recvcounts = calloc(diverge_mpi_comm_size(),sizeof*recvcounts),
                *senddispls = calloc(diverge_mpi_comm_size(),sizeof*senddispls),
                *recvdispls = calloc(diverge_mpi_comm_size(),sizeof*recvdispls);
        int to = (diverge_mpi_comm_rank() + 1) % diverge_mpi_comm_size(),
            from = (diverge_mpi_comm_rank() - 1 + diverge_mpi_comm_size()) % diverge_mpi_comm_size();
        char* other = calloc( nbytes_ranks[from], 1 );
        sendcounts[to] = nbytes;
        recvcounts[from] = nbytes_ranks[from];
        diverge_mpi_alltoallv_bytes( ptr, sendcounts, senddispls, other, recvcounts, recvdispls, 1 );
        free( sendcounts ); free( recvcounts ); free( senddispls ); free( recvdispls );
        free( nbytes_ranks );
        free( ptr );

        // get the symmstruct and check for error
        tu_symmstruct_t* symm_from = symmstruct_deserialize( other );
        free( other );
        int error = symmstruct_equal( symm, symm_from );

        tu_symmstruct_delete( symm_from );

        // print if we found an error
        int* errors = calloc( diverge_mpi_comm_size(), sizeof*errors );
        diverge_mpi_allgather_int( &error, errors, 1 );
        int total_error = 0;
        for (int i=0; i<diverge_mpi_comm_size(); ++i) total_error += errors[i];
        free( errors );
        if (total_error) {
            mpi_err_printf_all( "rank %i: error %i for TU symm-cmp (round-robin from %i)\n",
                    diverge_mpi_comm_rank(), error, from );
        } else {
            mpi_log_printf( "TU symm-compare check successful\n" );
        }
    }
}

char* _push_value( char* ptr, uindex_t* p_sz, const void* pval, size_t pwidth ) {
    uindex_t sz = *p_sz;
    ptr = (char*)realloc(ptr, sz + pwidth);
    memcpy( ptr + sz, pval, pwidth );
    *p_sz += pwidth;
    return ptr;
}

char* _push_vector( char* ptr, uindex_t* p_sz, const void* pvec, uindex_t nel, size_t pwidth ) {
    uindex_t sz = *p_sz;
    uindex_t add = pwidth * nel + sizeof(nel);
    ptr = (char*)realloc(ptr, sz + add);

    memcpy( ptr + sz, &nel, sizeof(nel) );
    memcpy( ptr + sz + sizeof(nel), pvec, pwidth * nel );

    *p_sz += add;
    return ptr;
}

#define push_value( ptr, psz, val ) _push_value( ptr, psz, &val, sizeof(val) )
#define push_vector( ptr, psz, vec, nel ) _push_vector( ptr, psz, vec, nel, sizeof(*vec) )

#define PUSH_VALUE( val ) { ptr = push_value( ptr, &size, val ); }
#define PUSH_VECTOR( vec, sz ) { ptr = push_vector( ptr, &size, vec, sz ); }

void symmstruct_serialize( const tu_symmstruct_t* symm, char** ptr_, uindex_t* nbytes_ ) {
    char* ptr = NULL;
    uindex_t size = 0;
    PUSH_VALUE( symm->use_symmetries );
    PUSH_VECTOR( symm->inv_symmetry, symm->n_sym );
    PUSH_VECTOR( symm->mi_prefac_in_out, symm->mi_prefac_in_out_sz );
    PUSH_VECTOR( symm->mi_prefac_in_in, symm->mi_prefac_in_in_sz );
    PUSH_VECTOR( symm->mi_maps_to, symm->mi_maps_to_sz );
    PUSH_VECTOR( symm->mi_map_len, symm->mi_cnt );
    PUSH_VECTOR( symm->mi_map_off, symm->mi_cnt );
    PUSH_VECTOR( symm->beyond_UC, symm->beyond_UC_sz );
    PUSH_VECTOR( symm->kmaps_to, symm->nk );
    PUSH_VECTOR( symm->idx_ibz_in_fullmesh, symm->nkibz );
    PUSH_VECTOR( symm->which_symmetry_used, symm->nk );
    PUSH_VECTOR( symm->prefactor_symmP, symm->prefactor_symmP_sz );
    PUSH_VECTOR( symm->prefactor_symmCD, symm->prefactor_symmCD_sz );
    PUSH_VECTOR( symm->idx_map_symm, symm->idx_map_symm_sz );
    PUSH_VECTOR( symm->o2m_map_len, symm->o2m_map_sz );
    PUSH_VECTOR( symm->o2m_map_off, symm->o2m_map_sz );
    PUSH_VECTOR( symm->from_ibz_2_bz,
            symm->from_ibz_2_bz_off[symm->nkibz-1]+symm->from_ibz_2_bz_len[symm->nkibz-1] );
    PUSH_VECTOR( symm->from_ibz_2_bz_off, symm->nkibz );
    PUSH_VECTOR( symm->from_ibz_2_bz_len, symm->nkibz );
    PUSH_VALUE( symm->my_nk_in_bz );
    PUSH_VALUE( symm->my_nk_in_bz_off );
    *ptr_ = ptr;
    *nbytes_ = size;
}

void _pop_value( const char* ptr, uindex_t* p_off, char* pval, size_t pwidth ) {
    memcpy( pval, ptr + *p_off, pwidth );
    *p_off += pwidth;
}

#define pop_value( type, target, ptr, p_off ) { \
    type _tmp; \
    _pop_value( ptr, p_off, (char*)&_tmp, sizeof(type) ); \
    target = _tmp; \
}

void _pop_vector( const char* ptr, uindex_t* p_off, void** pvec, uindex_t* nel, size_t pwidth ) {
    pop_value( uindex_t, *nel, ptr, p_off );
    *pvec = malloc( pwidth * (*nel) );
    memcpy( *pvec, ptr + *p_off, pwidth * (*nel) );
    *p_off += pwidth * (*nel);
}

#define pop_vector( type, target, target_sz, ptr, p_off ) { \
    type* _tmp; \
    uindex_t _tmp_sz; \
    _pop_vector( ptr, p_off, (void**)&_tmp, &_tmp_sz, sizeof(*_tmp) ); \
    target = _tmp; \
    target_sz = _tmp_sz; \
}

#define POP_VECTOR( type, t, t_sz ) pop_vector( type, t, t_sz, ptr, &off )
#define POP_VALUE( type, t ) pop_value( type, t, ptr, &off )

tu_symmstruct_t* symmstruct_deserialize( const char* ptr ) {
    tu_symmstruct_t* symm = calloc(1, sizeof*symm);
    uindex_t off = 0;
    POP_VALUE( bool, symm->use_symmetries );
    POP_VECTOR( index_t, symm->inv_symmetry, symm->n_sym );
    POP_VECTOR( complex128_t, symm->mi_prefac_in_out, symm->mi_prefac_in_out_sz );
    POP_VECTOR( complex128_t, symm->mi_prefac_in_in, symm->mi_prefac_in_in_sz );
    POP_VECTOR( index_t, symm->mi_maps_to, symm->mi_maps_to_sz );
    POP_VECTOR( index_t, symm->mi_map_len, symm->mi_cnt );
    POP_VECTOR( index_t, symm->mi_map_off, symm->mi_cnt );
    POP_VECTOR( double, symm->beyond_UC, symm->beyond_UC_sz );
    POP_VECTOR( index_t, symm->kmaps_to, symm->nk );
    POP_VECTOR( index_t, symm->idx_ibz_in_fullmesh, symm->nkibz );
    POP_VECTOR( tu_whichsym_t, symm->which_symmetry_used, symm->nk );
    POP_VECTOR( complex128_t, symm->prefactor_symmP, symm->prefactor_symmP_sz );
    POP_VECTOR( complex128_t, symm->prefactor_symmCD, symm->prefactor_symmCD_sz );
    POP_VECTOR( index_t, symm->idx_map_symm, symm->idx_map_symm_sz );
    POP_VECTOR( index_t, symm->o2m_map_len, symm->o2m_map_sz );
    POP_VECTOR( index_t, symm->o2m_map_off, symm->o2m_map_sz );
    index_t i2b = -1;
    POP_VECTOR( index_t, symm->from_ibz_2_bz, i2b );
    POP_VECTOR( index_t, symm->from_ibz_2_bz_off, symm->nkibz );
    POP_VECTOR( index_t, symm->from_ibz_2_bz_len, symm->nkibz );
    if (i2b != symm->from_ibz_2_bz_off[symm->nkibz-1]+symm->from_ibz_2_bz_len[symm->nkibz-1]) {
        mpi_err_printf( "something went wrong, not the right lengths\n" );
    }
    POP_VALUE( index_t, symm->my_nk_in_bz );
    POP_VALUE( index_t, symm->my_nk_in_bz_off );
    return symm;
}

static inline int vector_cmp_bytes( const char* A, const char* B, index_t sz ) {
    int errors = 0;
    for (index_t i=0; i<sz; ++i) if (A[i] != B[i]) { errors++; break; }
    return errors;
}

static inline int vector_cmp_doubles( const double* A, const double* B, index_t sz ) {
    int errors = 0;
    for (index_t i=0; i<sz; ++i) if (fabs(A[i]-B[i]) > 1e-5) { errors++; break; }
    return errors;
}

#define vector_cmp_i( A, B, len ) vector_cmp_bytes( (const char*)A, (const char*)B, len*sizeof(*A) )
#define vector_cmp_f( A, B, len ) vector_cmp_doubles( (const double*)A, (const double*)B, \
        len*sizeof(*A)/sizeof(double) )

int symmstruct_equal( const tu_symmstruct_t* A, const tu_symmstruct_t* B ) {
    int errors = 0;
    errors += A->use_symmetries != B->use_symmetries;
    errors += vector_cmp_i( A->inv_symmetry, B->inv_symmetry, A->n_sym );

    errors += vector_cmp_f( A->mi_prefac_in_out, B->mi_prefac_in_out, A->mi_prefac_in_out_sz );
    errors += vector_cmp_f( A->mi_prefac_in_in, B->mi_prefac_in_in, A->mi_prefac_in_in_sz );
    errors += vector_cmp_i( A->mi_maps_to, B->mi_maps_to, A->mi_maps_to_sz );
    errors += vector_cmp_i( A->mi_map_len, B->mi_map_len, A->mi_cnt );
    errors += vector_cmp_i( A->mi_map_off, B->mi_map_off, A->mi_cnt );
    errors += vector_cmp_f( A->beyond_UC, B->beyond_UC, A->beyond_UC_sz );
    errors += vector_cmp_i( A->kmaps_to, B->kmaps_to, A->nk );
    errors += vector_cmp_i( A->idx_ibz_in_fullmesh, B->idx_ibz_in_fullmesh, A->nkibz );
    errors += vector_cmp_i( A->which_symmetry_used, B->which_symmetry_used, A->nk );

    /* those are MPI specific (due to my_nk)
    errors += A->prefactor_symmP != B->prefactor_symmP;
    errors += A->prefactor_symmCD != B->prefactor_symmCD;
    errors += A->idx_map_symm != B->idx_map_symm;
    errors += A->o2m_map_len != B->o2m_map_len;
    errors += A->o2m_map_off != B->o2m_map_off;
    */

    errors += vector_cmp_i( A->from_ibz_2_bz_off, B->from_ibz_2_bz_off, A->nkibz );
    errors += vector_cmp_i( A->from_ibz_2_bz_len, B->from_ibz_2_bz_len, A->nkibz );
    errors += vector_cmp_i( A->from_ibz_2_bz, B->from_ibz_2_bz,
            A->from_ibz_2_bz_off[A->nkibz-1]+A->from_ibz_2_bz_len[A->nkibz-1] );

    return errors;
}

static inline int approx_id( const double* A, index_t sz ) {
    double maxdiff = 0.0;
    for (index_t i=0; i<sz; ++i)
    for (index_t j=0; j<sz; ++j) {
        double diff = fabs(A[IDX2(i,j,sz)] - ((i==j)?1.0:0.0));
        maxdiff = MAX(diff,maxdiff);
    }
    return maxdiff < SYMNORM_MATRIX_ZERO;
}

static inline int capprox_id( const complex128_t* A, index_t sz ) {
    double maxdiff = 0.0;
    for (index_t i=0; i<sz; ++i)
    for (index_t j=0; j<sz; ++j) {
        double diff = cabs( A[IDX2(i,j,sz)] - (((i==j)?1.0:0.0)) );
        maxdiff = MAX(diff,maxdiff);
    }
    return maxdiff < SYMNORM_MATRIX_ZERO;
}

