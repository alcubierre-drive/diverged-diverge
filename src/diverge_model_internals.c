/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_model_internals.h"

#include "diverge_internals_struct.h"
#include "diverge_momentum_gen.h"
#include "diverge_symmetrize.h"
#include "misc/batched_eigen.h"
#include "misc/mpi_functions.h"
#include "misc/merge_orb_w_rs_symm.h"
#include "diverge_symmetrize.h"
#include <omp.h>


typedef enum {
    default_spin_index_zerolength,
    default_spin_index_all,
    default_spin_index_all_mix_zero,
    default_spin_index_mixed,
    default_spin_index_none,
} default_spin_index_t;

static default_spin_index_t check_model_default_spin_index( const diverge_model_t* model ) {
    index_t result = 0;
    index_t* count_zero_length_C = (index_t*)calloc(POW2(model->n_orb), sizeof(index_t));
    index_t* count_zero_length_D = (index_t*)calloc(POW2(model->n_orb), sizeof(index_t));
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
    mpi_log_printf("model is %i dimensional \n", intls->dim);

    index_t nk = kdim(model->nk),
            nktot = kdimtot(model->nk, model->nkf),
            nb = model->n_spin * model->n_orb;

    intls->enforce_exchange = model->SU2 == 0;

    if (nk < diverge_mpi_comm_size() * diverge_omp_num_threads()) {
        mpi_wrn_printf("using more ranks×threads (%i×%i) than coarse kpts (%li).\n",
                diverge_mpi_comm_size(), diverge_omp_num_threads(), nk);
        mpi_wrn_printf("please run divERGe with sensible OMP_NUM_THREADS exported\n");
        mpi_wrn_printf("trying to continue anyways…\n");
    }

    mpi_vrb_printf("allocating/generating meshes\n");
    intls->kmesh = (double*)malloc(sizeof(double)*nk*3);
    intls->kfmesh = (double*)malloc(sizeof(double)*nktot*3);
    diverge_model_generate_meshes( intls->kmesh, intls->kfmesh, model->nk,
            model->nkf, model->lattice );

    mpi_vrb_printf("allocating internal common buffers\n");
    intls->ham = (complex128_t*)calloc(nktot*nb*nb,sizeof(complex128_t));
    intls->U = (complex128_t*)calloc(nktot*nb*nb,sizeof(complex128_t));
    intls->E = (double*)calloc(nktot*nb,sizeof(double));
    intls->greens = (gf_complex_t*)calloc(nktot*nb*nb*2,sizeof(gf_complex_t));

    mpi_vrb_printf("filling hamiltonian buffer\n");
    (*(model->hfill))( model, intls->ham );
    if (model->n_sym < 0) {
        model->n_sym *= -1;
        double err = diverge_symmetrize_2pt_fine( model, intls->ham, intls->U );
        if (err > DIVERGE_EPS_MESH) {
            mpi_wrn_printf("resymmetrize hamiltonian->large error: %.5e\n", err);
        }
    }

    mpi_vrb_printf("diagonalizing hamiltonian\n");
    memcpy( intls->U, intls->ham, sizeof(complex128_t)*nktot*nb*nb );
    batched_eigen_r( NULL, -2, intls->U, intls->E, nb, nktot );

    merge_rs_orb(model);
}

void diverge_model_internals_reset( diverge_model_t* model ) {
    if (model->internals == NULL) {
        mpi_err_printf("cannot reset internals, ptr==NULL\n");
        return;
    }
    free(model->internals->kmesh);
    free(model->internals->kfmesh);
    free(model->internals->ham);
    free(model->internals->E);
    free(model->internals->U);
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
    if ((model->internals->grid_sym != NULL) &&
        (model->internals->grid_sym_destructor != NULL))
        (*(model->internals->grid_sym_destructor))( model->internals->grid_sym );
    if ((model->internals->patch_vertex != NULL) &&
        (model->internals->patch_vertex_destructor != NULL))
        (*(model->internals->patch_vertex_destructor))( model->internals->patch_vertex );
    if ((model->internals->tu_data != NULL) &&
        (model->internals->tu_data_destructor != NULL))
        (*(model->internals->tu_data_destructor))( model->internals->tu_data );

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

    memset(model->internals, 0, sizeof(internals_t));
    model->internals->has_common_internals = 0;
}

#define verbose_assert( expr, ... ) \
    if (!(expr)) { \
        error++; \
        mpi_err_printf( __VA_ARGS__ ); \
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
    verbose_assert( model->n_sym >= 0 && model->n_sym < MAX_N_SYM, "invalid n_sym (%li)\n", model->n_sym )
    verbose_assert( (model->n_sym == 0) || (model->orb_symmetries != NULL), "orbital symmetries must be allocated (found NULL)\n" )
    verbose_assert( model->n_hop >= 0, "invalid n_hop (%li)\n", model->n_hop )
    verbose_assert( (model->n_hop == 0) || (model->hop != NULL), "realspace hoppings must be allocated (found NULL)\n" )
    verbose_assert( (model->n_vert == 0) || (model->vert != NULL), "realspace vertex must be allocated (found NULL)\n" )

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

    verbose_assert( model->n_vert >= 0, "invalid n_vert (%li)\n", model->n_vert )
    verbose_assert( model->gfill != NULL, "must provide gfill (found NULL)\n" )
    verbose_assert( model->vfill != NULL || model->ffill != NULL, "must provide either vfill or ffill (found NULL)\n" )
    if (model->data != NULL)
        verbose_assert( model->nbytes_data > 0, "additional data set, nbytes_data>0 required, found nbytes_data=%li\n", model->nbytes_data )
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
