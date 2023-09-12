#pragma once

#include "diverge_model.h"

// the interface with energies and bands is chosen because we want to make it
// possible to simulate models with a correlated subspace, i.e. more bands in
// the dispersion than in the vertex. generating a patching for these models
// thus requires the energy array E as a separate piece of information, as well
// as the number of bands nb.

#ifdef __cplusplus
#include <vector>
using std::vector;
// Function: diverge_patching_find_fs_pts
// uses internal energy array if NULL is provided for E
vector<index_t> diverge_patching_find_fs_pts( diverge_model_t* mod, const double* E, index_t nb, index_t np_ibz, index_t np_search );
extern "C" {
#endif // __cplusplus

// Function: diverge_patching_find_fs_pts_C
// C interface to the above function. must free the pts memory with free(*pts)
void diverge_patching_find_fs_pts_C( diverge_model_t* mod, double* E, index_t nb,
        index_t np_ibz, index_t np_search, index_t** pts, index_t* npts );

// Function: diverge_patching_from_indices
// allocates everything inside the mom_patching_t* struct. the mom_patching_t*
// internals as well as the struct itself are freed by the code itself when they
// are attached to a diverge_model_t* instance.
mom_patching_t* diverge_patching_from_indices( diverge_model_t* mod, const index_t* patches, index_t npatches );

// Function: diverge_patching_autofine
// alpha - (|E-E_min|/|E_max-E_min|)^alpha is used to calculate group index g
// beta - (1-g/#g)^beta is used to calculate number of clusters in group g
// gamma - if gamma < beta * #g use all points in cluster
//
// uses internal energy array if NULL is provided for E
void diverge_patching_autofine( diverge_model_t* mod, mom_patching_t* patch,
        const double* E, index_t nb, index_t ngroups, double alpha, double beta,
        double gamma );

// Function: diverge_patching_symmetrize_refinement
// according to the realspace symmetries defined in the model the refinement is
// symmetrized. useful if the refinement has been generated with the autofine
// method.
void diverge_patching_symmetrize_refinement( diverge_model_t* mod, mom_patching_t* patch );

// Function: diverge_patching_free
// frees all resources of a <mom_patching_t> structure. Usage is typically not
// needed, but for advanced patching calculations it may be advantageous.
void diverge_patching_free( mom_patching_t* patching );


// typical calling of the above functions in the model would look like
//
// index_t *patches, npatches;
// diverge_patching_find_fs_pts_C( mod, E, nb, np_ibz, np_ibz*10, &patches, &npatches );
// mom_patching_t* pat = diverge_patching_from_indices( mod, patches, npatches );
// free(patches);
// diverge_patching_autofine( mod, pat, E, nb, 10, 0.5, 4.0, 0.4 );
// diverge_patching_symmetrize_refinement( mod, pat );
// mod->patching = pat;
//
// ... (some further diverge code)

#ifdef __cplusplus
}
#endif
