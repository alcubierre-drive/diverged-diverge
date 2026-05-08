/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_model_internals.h"

#include "diverge_internals_struct.h"
#include "diverge_momentum_gen.h"
#include "diverge_symmetrize.h"
#include "misc/batched_eigen.h"
#include "misc/batched_svd.h"
#include "misc/mpi_functions.h"
#include "diverge_flow_type.h"
#include "misc/shared_mem.h"
#include "misc/qnan.h"

#include <stdarg.h>
#include <limits.h>

static double SYMMETRIZE_HAMILTONIAN_ERROR = DIVERGE_EPS_MESH;
double diverge_model_internals_common_set_symmbound( double bound ) {
    double old_bound = SYMMETRIZE_HAMILTONIAN_ERROR;
    SYMMETRIZE_HAMILTONIAN_ERROR = bound;
    return old_bound;
}

double diverge_max_dist_iobi( void ) {
    return qnan_gen( IOBI_FORMFACTOR_MAGIC_NUMBER );
}

void diverge_model_internals_any( diverge_model_t* model, const char* mode, ... ) {
    if (!model->internals->has_common_internals)
        diverge_model_internals_common( model );
    diverge_flow_type t = diverge_flow_type_fromstr( mode );
    mpi_log_printf("internals mode: %s\n", mode);
    va_list ap;
    va_start(ap, mode);

    index_t np = 0;
    double ffdist = 0;
    typedef rs_hopping_t* rs_hopping_t_ptr;
    rs_hopping_t_ptr phop = NULL;
    index_t nphop = 0;
    typedef ph_vertex_t* ph_vertex_t_ptr;
    ph_vertex_t_ptr pvert = NULL;
    index_t npvert = 0;

    switch (t) {
        case diverge_grid:
            diverge_model_internals_grid( model );
            break;
        case diverge_patch:
            np = va_arg(ap, index_t);
            diverge_model_internals_patch( model, np );
            break;
        case diverge_tu:
            ffdist = va_arg(ap, double);
            diverge_model_internals_tu( model, ffdist );
            break;
        case diverge_tu_phonons:
            ffdist = va_arg(ap, double);
            phop = va_arg(ap, rs_hopping_t_ptr);
            nphop = va_arg(ap, index_t);
            pvert = va_arg(ap, ph_vertex_t_ptr);
            npvert = va_arg(ap, index_t);
            diverge_model_internals_tu_phonons( model, ffdist, phop, nphop, pvert, npvert );
            break;
        case diverge_flow_type_invalid:
        default:
            mpi_err_printf("invalid flow type '%s' (use tu, tu_phonons, grid, patch)\n", mode);
            goto EndFunc;
            break;
    }
EndFunc:
    va_end(ap);
}

static index_t find_zero_array_elements( void* array_, index_t sz, index_t num ) {
    unsigned char* array = array_;
    unsigned char* testblock = calloc(1, sz);
    index_t num_nonzero = 0;
    for (index_t i=0; i<num; ++i)
        if (memcmp( testblock, array+i*sz, sz ) == 0)
            num_nonzero++;
    free(testblock);
    return num_nonzero;
}

typedef enum {
    default_spin_index_zerolength,
    default_spin_index_all,
    default_spin_index_all_mix_zero,
    default_spin_index_mixed,
    default_spin_index_none,
} default_spin_index_t;

static default_spin_index_t check_model_default_spin_index( const diverge_model_t* model ) {
    index_t result = 0;
    index_t* count_zero_length_C = calloc(POW2(model->n_orb), sizeof(index_t));
    index_t* count_zero_length_D = calloc(POW2(model->n_orb), sizeof(index_t));
    for (index_t idx=0; idx<model->n_vert; ++idx) {
        const rs_vertex_t* vert = model->vert + idx;
        result += vert->s1 == -1;
        if (vert->s1 == -1) {
            if (vert->chan == 'C' || vert->chan == 'D') {
                index_t* count = vert->chan == 'C' ? count_zero_length_C : count_zero_length_D;
                if (vert->R[0] == 0 && vert->R[1] == 0 && vert->R[2] == 0) {
                    count[ IDX2( vert->o1, vert->o2, model->n_orb ) ]++;
                }
            }
        }
    }
    int array_equal = 0;
    for (index_t i=0; i<POW2(model->n_orb); ++i)
        array_equal += count_zero_length_D[i] != count_zero_length_C[i];

    free(count_zero_length_C);
    free(count_zero_length_D);

    if (result == 0)
        return default_spin_index_zerolength;
    if (model->n_vert == result) {
        if (!array_equal)
            return default_spin_index_all_mix_zero;
        return default_spin_index_all;
    }
    if (result != 0)
        return default_spin_index_mixed;
    return default_spin_index_none;
}

void diverge_model_internals_common( diverge_model_t* model ) {
    internals_t* intls = model->internals;

    if (intls->has_common_internals)
        return;
    else
        intls->has_common_internals = 1;

    intls->dim = 0;
    for (int d=0; d<3; ++d) {
        intls->periodic_direction[d] = model->nk[d] != 0;
        intls->dim += intls->periodic_direction[d];
        if (model->nk[d] == 0) model->nk[d] = 1;
        if (model->nkf[d] == 0) model->nkf[d] = 1;
    }
    mpi_log_printf("model is %iD\n", intls->dim);

    index_t nk = kdim(model->nk),
            nktot = kdimtot(model->nk, model->nkf),
            nb = model->n_spin * model->n_orb;

    intls->enforce_exchange = (model->SU2 == 0);

    if (nk < diverge_mpi_comm_size() * diverge_omp_num_threads()) {
        int nthr_new = nk / diverge_mpi_comm_size();
        mpi_wrn_printf("using more ranks×threads (%i×%i) than coarse kpts (%li).\n",
                diverge_mpi_comm_size(), diverge_omp_num_threads(), nk);
        mpi_wrn_printf("please run divERGe with sensible OMP_NUM_THREADS (%i) exported\n", nthr_new);
        mpi_wrn_printf("trying to continue anyways…\n");
    }

    mpi_vrb_printf("allocating/generating meshes\n");
    intls->kmesh = shared_malloc(sizeof(double)*nk*3);
    intls->kfmesh = shared_malloc(sizeof(double)*nktot*3);
    diverge_lattice_t* rlattice = shared_malloc(sizeof *rlattice);
    if (shared_exclusive_enter(intls->kmesh)) {
        *rlattice = diverge_model_generate_meshes( intls->kmesh, intls->kfmesh,
                model->nk, model->nkf, model->lattice );
    }
    shared_exclusive_wait(intls->kmesh);
    intls->rlattice = *rlattice;
    shared_free(rlattice);

    if (intls->common_internals_mesh_only) {
        mpi_wrn_printf( "early termination of common_internals, only kmesh\n" );
        return;
    }

    mpi_vrb_printf("allocating internal common buffers\n");
    intls->ham = shared_calloc(nktot*nb*nb,sizeof(complex128_t));
    intls->U = shared_calloc(nktot*nb*nb,sizeof(complex128_t));
    intls->E = shared_calloc(nktot*nb,sizeof(double));

    // the GF does not default to shared, as writing GF generators for
    // node-shared mem is a bit complicated (see diverge_hack.cpp)
    intls->greens = calloc(nktot*nb*nb*2,sizeof(gf_complex_t));

    // we check here whether to symmetrize the Hamiltonian
    int symmetrize_hamiltonian = model->n_sym < 0;
    model->n_sym = labs(model->n_sym);

    if (model->n_sym > 0) {
        // this step must be taken out of the shmem region because the GF is
        // symmetrized as well, and the structure won't be initialized on all
        // ranks that way.
        diverge_generate_symm_maps( model );
    }

    mpi_vrb_printf("filling hamiltonian buffer\n");
    if (shared_exclusive_enter(intls->ham))
        (*(model->hfill))( model, intls->ham );
    shared_exclusive_wait(intls->ham);

    if (symmetrize_hamiltonian) {
        mpi_vrb_printf( "symmetrizing Hamiltonain before diagonalization\n" );
        // copyin
        memcpy( intls->greens, intls->ham, sizeof(complex128_t)*nktot*nb*nb );
        // symm
        double err = diverge_symmetrize_2pt_fine( model, (complex128_t*)intls->greens, NULL );
        if (err > SYMMETRIZE_HAMILTONIAN_ERROR)
            mpi_wrn_printf("resymmetrize hamiltonian->large error: %.5e\n", err);
        // copyout
        if (shared_exclusive_enter( intls->ham ))
            memcpy( intls->ham, intls->greens, sizeof(complex128_t) * nktot*nb*nb );
        shared_exclusive_wait( intls->ham );
        memset( intls->greens, 0, sizeof(complex128_t)*nktot*nb*nb );
    }

    if (intls->ignore_eigensolver_in_common) {
        mpi_vrb_printf("ignoring hamiltonian diag\n");
    } else {
        mpi_vrb_printf("diagonalizing hamiltonian\n");
        if (shared_exclusive_enter(intls->U)) {
            memcpy( intls->U, intls->ham, sizeof(complex128_t)*nktot*nb*nb );
            if (!intls->use_batched_svd_algo_in_common)
                batched_eigen_r( NULL, -2, intls->U, intls->E, nb, nktot );
            else {
                mpi_vrb_printf( "doing eigensolution via the SVD module\n" );
                int selector_flag = INT_MIN;
                if (intls->use_batched_svd_algo_in_common == -1) {
                    mpi_vrb_printf( "defaulting to CPU implementation\n" );
                    selector_flag += 1;
                }
                batched_svd( NULL, selector_flag, intls->ham, intls->U, NULL, intls->E, nb, nktot );
            }
        }
        shared_exclusive_wait( intls->U );
    }
}

void diverge_model_internals_reset( diverge_model_t* model ) {
    if (model->internals == NULL) {
        mpi_err_printf("cannot reset internals, ptr==NULL\n");
        return;
    }
    shared_free(model->internals->kmesh);
    shared_free(model->internals->kfmesh);
    shared_free(model->internals->ham);
    shared_free(model->internals->E);
    shared_free(model->internals->U);

    if (model->internals->greens_shared)
        shared_free(model->internals->greens);
    else
        free(model->internals->greens);

    free(model->internals->grid_kfmesh);
    free(model->internals->grid_weights);
    free(model->internals->grid_kfrev);
    free(model->internals->grid_k1m2);
    free(model->internals->grid_k1p2);
    free(model->internals->grid_greens_p_buf);
    free(model->internals->grid_greens_m_buf);
    if ((model->internals->grid_vertex != NULL) &&
        (model->internals->grid_vertex_destructor != NULL))
        (*(model->internals->grid_vertex_destructor))( model->internals->grid_vertex );
    if ((model->internals->patch_vertex != NULL) &&
        (model->internals->patch_vertex_destructor != NULL))
        (*(model->internals->patch_vertex_destructor))( model->internals->patch_vertex );
    if ((model->internals->tu_data != NULL) &&
        (model->internals->tu_data_destructor != NULL))
        (*(model->internals->tu_data_destructor))( model->internals->tu_data );
    if ((model->internals->tu_data_phonons != NULL) &&
        (model->internals->tu_data_phonons_destructor != NULL))
        (*(model->internals->tu_data_phonons_destructor))( model->internals->tu_data_phonons );
    if ((model->internals->model_phonons != NULL) &&
        (model->internals->model_phonons_destructor != NULL))
        (*(model->internals->model_phonons_destructor))( model->internals->model_phonons );
    free(model->internals->model_phonons_imag_freq);

    free(model->internals->patch_kmesh);
    free(model->internals->patch_pmesh);
    free(model->internals->patch_k3map);

    free(model->internals->symm_map_mom_fine);
    free(model->internals->symm_map_mom_crs);
    free(model->internals->symm_beyond_UC);
    free(model->internals->symm_map_orb);
    free(model->internals->symm_orb_off);
    free(model->internals->symm_orb_len);
    free(model->internals->symm_pref);

    int dim = model->internals->dim;
    for (int d=dim; d<3; ++d) model->nk[d] = model->nkf[d] = 0;

    memset(model->internals, 0, sizeof(internals_t));
    model->internals->has_common_internals = 0;
}

#define verbose_assert( expr, ... ) \
    if (!(expr)) { \
        error++; \
        mpi_err_printf( __VA_ARGS__ ); \
    }
#define verbose_wassert( expr, ... ) \
    if (!(expr)) { \
        mpi_wrn_printf( __VA_ARGS__ ); \
    }
int diverge_model_validate( diverge_model_t* model ) {
    int error = 0;

    bool dim0c = model->nk[0] == 0 && model->nk[1] == 0 && model->nk[2] == 0,
         dim1c = model->nk[0]  > 0 && model->nk[1] == 0 && model->nk[2] == 0,
         dim2c = model->nk[0]  > 0 && model->nk[1]  > 0 && model->nk[2] == 0,
         dim3c = model->nk[0]  > 0 && model->nk[1]  > 0 && model->nk[2]  > 0;
    bool dim0f = model->nkf[0] == 0 && model->nkf[1] == 0 && model->nkf[2] == 0,
         dim1f = model->nkf[0]  > 0 && model->nkf[1] == 0 && model->nkf[2] == 0,
         dim2f = model->nkf[0]  > 0 && model->nkf[1]  > 0 && model->nkf[2] == 0,
         dim3f = model->nkf[0]  > 0 && model->nkf[1]  > 0 && model->nkf[2]  > 0;

    int dim_c = dim0c ? 0 : dim1c ? 1 : dim2c ? 2 : dim3c ? 3 : -1;
    int dim_f = dim0f ? 0 : dim1f ? 1 : dim2f ? 2 : dim3f ? 3 : -1;
    verbose_assert( dim_c != -1 && dim_f != -1, "invalid dimensions\n" )
    verbose_assert( dim_c == dim_f, "fine mesh dim (%i) doesnt match coarse mesh dim (%i)\n", dim_f, dim_c )
    verbose_assert( dim0c ^ dim1c ^ dim2c ^ dim3c, "coarse mesh must be either 0, 1, 2, or 3 dimensional\n" )

    verbose_assert( model->n_orb > 0 && model->n_orb < MAX_N_ORBS, "invalid n_orb (%li)\n", model->n_orb )
    verbose_assert( labs(model->n_sym) < MAX_N_SYM, "too many symmetries given (%li)\n", model->n_sym )
    verbose_assert( (model->n_sym == 0) || (model->orb_symmetries != NULL), "orbital symmetries must be allocated (found NULL)\n" )
    verbose_assert( model->n_hop >= 0, "invalid n_hop (%li)\n", model->n_hop )
    verbose_assert( (model->n_hop == 0) || (model->hop != NULL), "realspace hoppings must be allocated (found NULL)\n" )
    verbose_assert( (model->n_vert == 0) || (model->vert != NULL), "realspace vertex must be allocated (found NULL)\n" )

    if (model->n_ibz_path > 0) {
        int ibz_path_outside_1st_pz = 0;
        for (index_t i=0; i<model->n_ibz_path; ++i)
        for (short d=0; d<3; ++d)
            ibz_path_outside_1st_pz += (model->ibz_path[i][d] > (1.0 + DIVERGE_EPS_MESH)) ||
                                       (model->ibz_path[i][d] < (0.0 - DIVERGE_EPS_MESH)) ;
        verbose_assert( ibz_path_outside_1st_pz == 0, "ibz_path not in 1st primitive zone (0<=ibz_path<=1; found %i violations)\n",
                ibz_path_outside_1st_pz )
    }

    verbose_assert( (model->SU2>0 && model->n_spin == 1) ^ ((model->SU2<=0) && model->n_spin >=1),
            "either (SU2>0 && n_spin==1) or (n_spin>=1), have SU2=%i and n_spin=%li\n", model->SU2, model->n_spin )

    switch (check_model_default_spin_index( model )) {
        case default_spin_index_all_mix_zero:
            if (model->SU2 == 0) {
                mpi_wrn_printf("number of D and C channel on-site vertex elements is equal,\n");
                mpi_wrn_printf("model->SU2==0 (i.e. enforced), and all elements carry auto SU2. INTENDED?\n");
            }
            break;
        case default_spin_index_all:
            if (model->SU2 < 0) {
                mpi_wrn_printf("found default spin config in *all* interaction components,\n");
                mpi_wrn_printf("but model->SU2<0 (i.e. do not enforce exchange symmetry). INTENDED?\n");
            }
            break;
        case default_spin_index_mixed:
            if (model->SU2 == 0) {
                mpi_wrn_printf("found default spin config only in *some* interaction components,\n");
                mpi_wrn_printf("and model->SU2==0 (i.e. enforce exchange symmetry). INTENDED?\n");
            }
            break;
        case default_spin_index_none:
            if (model->SU2 == 0) {
                mpi_wrn_printf("found default spin config in *no* interaction component,\n");
                mpi_wrn_printf("but model->SU2==0 (i.e. enforce exchange symmetry). INTENDED?\n");
            }
            break;
        case default_spin_index_zerolength:
        default:
            break;
    }

    index_t num_zero = 0;

    num_zero = find_zero_array_elements( model->hop, sizeof(rs_hopping_t), model->n_hop );
    verbose_wassert( num_zero == 0, "found %li hopping elements that are all zero bytes\n", num_zero );
    num_zero = find_zero_array_elements( model->vert, sizeof(rs_vertex_t), model->n_vert );
    verbose_wassert( num_zero == 0, "found %li vertex elements that are all zero bytes\n", num_zero );

    num_zero = find_zero_array_elements( model->rs_symmetries, sizeof(double)*9, model->n_sym );
    verbose_wassert( num_zero == 0, "found %li rs symmetry elements that are all zero bytes\n", num_zero );
    num_zero = find_zero_array_elements( model->orb_symmetries, sizeof(complex128_t)
            *POW2(model->n_orb*model->n_spin), model->n_sym );
    verbose_wassert( num_zero == 0, "found %li orbital symmetry elements that are all zero bytes\n", num_zero );

    num_zero = find_zero_array_elements( model->positions, sizeof(double)*3, model->n_orb );
    if (num_zero == model->n_orb && model->n_orb > 1)
        mpi_wrn_printf( "found only zero positions and n_orb > 1. INTENDED?\n" );

    num_zero = find_zero_array_elements( model->lattice, sizeof(double)*3, 3 );
    if (num_zero != 0) mpi_wrn_printf( "found %li zero lattice vectors. INTENDED?\n", num_zero );

    verbose_assert( model->n_vert >= 0, "invalid n_vert (%li)\n", model->n_vert )
    verbose_assert( model->gfill != NULL, "must provide gfill (found NULL)\n" )
    verbose_assert( model->vfill != NULL || model->ffill != NULL,
            "must provide either vfill or ffill (found NULL)\n" )
    if (model->data != NULL)
        verbose_assert( model->nbytes_data > 0, "additional data set, nbytes_data>0 required, "
                "found nbytes_data=%li\n", model->nbytes_data )
    if (error)
        mpi_err_printf( "encountered %i errors…\n", error );
    return error;
}

double*       diverge_model_internals_get_E(      const diverge_model_t* model ) {
    return model->internals->has_common_internals ? model->internals->E : NULL;
}
complex128_t* diverge_model_internals_get_U(      const diverge_model_t* model ) {
    return model->internals->has_common_internals ? model->internals->U : NULL;
}
complex128_t* diverge_model_internals_get_H(      const diverge_model_t* model ) {
    return model->internals->has_common_internals ? model->internals->ham : NULL;
}
double*       diverge_model_internals_get_kmesh(  const diverge_model_t* model ) {
    return model->internals->has_common_internals ? model->internals->kmesh : NULL;
}
double*       diverge_model_internals_get_kfmesh( const diverge_model_t* model ) {
    return model->internals->has_common_internals ? model->internals->kfmesh : NULL;
}
gf_complex_t* diverge_model_internals_get_greens( const diverge_model_t* model ) {
    return model->internals->has_common_internals ? model->internals->greens : NULL;
}
index_t       diverge_model_internals_get_dim( const diverge_model_t* model ) {
    return model->internals->dim;
}
diverge_lattice_t diverge_model_internals_get_rlattice( const diverge_model_t* model ) {
    return model->internals->rlattice;
}
