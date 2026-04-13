/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_patching.h"
#include "diverge_patching_helpers.h"
#include "diverge_momentum_gen.h"

#include "diverge_model.h"
#include "diverge_internals_struct.h"

#include "misc/mpi_functions.h"
#include "misc/bitary.h"
#include "misc/mergesort.h"

#define LINALG_NO_LAPACKE
#include "misc/linalg.h"
LINALG_IMPLEMENT

#define SKIP_CLAV2D_T
#include "misc/kmeans_lloyd.h"

#include "diverge_flow_step.h"
#include "diverge_flow_step_internal.h"
#include "npatch/vertex.h"

// convenience functions to re-normalize and mpi-distribute the patching struct
static void diverge_patching_weights_equalize( mom_patching_t* patch );
static void diverge_patching_mpi_equalize( mom_patching_t* patch );

// helper structures
typedef struct { const double* min; const double* max; } minmax_t;
typedef struct { index_t group; index_t ppcoord; } group_t;
typedef struct { index_t p; index_t howmany; index_t* multi; } pgen_info_t;
typedef struct { index_t idx; double wght; } patch_helper_t;

// vectors (s.t. we can make vectors of vectors)
typedef struct { VECTOR_DECL( index_t, v ) } vec_index_t;
typedef struct { VECTOR_DECL( double, v ) } vec_double;


// helper functions
static inline index_t unique_idx( index_t* arr, index_t size );
static inline index_t unique_patch_helper( patch_helper_t* arr, index_t size );
static inline void limit_to( index_t max, index_t* idx );
static minmax_t minmax_element( const double* ary, index_t num );

static inline index_t kpt_to_idx( claV3d_t kpt, claM3d_t latt, const index_t nk[3] );
static inline index_t find_closest_np_search_idx( claV2d_t mean_vec, const claV2d_t* kmeans_points, index_t N );
static index_t* find_ibz_pts( diverge_model_t* mod, index_t* nibz );
static inline claV3d_t kpt_from_mesh( const double* mesh, index_t k ) { return claV3d_map( mesh+3*k ); }
static inline claV2d_t to_basis( claV2d_t k, complex128_t g1, complex128_t g2 );
static inline index_t closest_k_index( claV2d_t kpt, complex128_t g1, complex128_t g2, const index_t nk[3] );
static void* memcpy_and_advance( void* dest, const void* src, size_t size );
static void gen_serial_size_ary( const mom_patching_t* P, index_t mapsize, size_t ary[8] );
static void patching_serialize( mom_patching_t* P, void** pdata, size_t* psize );
static void* malloc_and_advance_src( void** dest, const void* src, size_t size );
static void patching_deserialize( mom_patching_t* P, void* data );
static inline index_t group_scaling_function( double E_min, double E_max, double E, int ngroups, double alpha );
static inline index_t num_clusters( index_t group, index_t group_count, index_t ngroups, double beta );
static inline group_t* gen_grouped_map( index_t* pmap, index_t pcnt, index_t p_xy,
        const index_t nk_flat[3], const index_t* groups );
static inline index_t* gen_group_counts( const group_t* grouped_map, index_t pcnt, index_t* ngroups );

// sorting algorithms for diverge_patching_find_fs_pts
#undef MERGESORT_COMP_R
#define MERGESORT_COMP_R( x, y, E ) (((double*)E)[x] < ((double*)E)[y] ? -1 : \
                                     ((double*)E)[x] > ((double*)E)[y] ? 1 : 0)
MERGESORT_IMPL_R( mergesort_idx, index_t )
MERGESORT_IMPL( mergesort_idx, index_t )

#undef MERGESORT_COMP
#define MERGESORT_COMP(x,y) (x.idx < y.idx ? -1 : x.idx > y.idx ? 1 : 0)
MERGESORT_IMPL( mergesort_patch_helper, patch_helper_t )

#undef MERGESORT_COMP
#define MERGESORT_COMP(x,y) (x.group < y.group ? -1 : \
                             x.group > y.group ? 1 : 0)
MERGESORT_IMPL( mergesort_group, group_t )


// returns an array that is terminated with -1
index_t* diverge_patching_find_fs_pts( diverge_model_t* mod, const double* E, index_t nb,
        index_t np_ibz, index_t np_search ) {
    index_t mod_n_sym = MAX(mod->n_sym,1);
    index_t prev_mod_n_sym = mod->n_sym;
    if (mod_n_sym != mod->n_sym) {
        mod->n_sym = mod_n_sym;
        claM3d_rmap( claM3d_id(), mod->rs_symmetries[0][0] );
    }
    if (np_search >= kdim(mod->nk)/mod_n_sym) {
        np_search = kdim(mod->nk)/mod_n_sym;
        mpi_wrn_printf("must have np_search<=nk/n_sym. reset np_search=nk/n_sym\n");
    }
    if (np_ibz < 1) {
        np_ibz = 1;
        mpi_wrn_printf("must have np_ibz>=1. reset np_ibz=1\n");
    }
    if (np_search < np_ibz) {
        np_search = np_ibz;
        mpi_wrn_printf("must have np_search>=np_ibz. reset np_search=np_ibz\n");
    }
    mpi_vrb_printf("finding FS with np_ibz=%li and np_search=%li\n", np_ibz, np_search);

    // reset energy to automatic energy if not passed as parameter
    if (E == NULL) E = mod->internals->E;
    if (nb == -1) nb = mod->n_orb * mod->n_spin;

    // find IBZ
    index_t nk_ibz; index_t* ibz_to_bz = find_ibz_pts( mod, &nk_ibz );

    // flattened minimum energies on IBZ
    double* E_ibz = calloc(nk_ibz, sizeof*E_ibz);
    // ibz indices for each energy
    index_t* E_ibz_idx = calloc(nk_ibz, sizeof*E_ibz_idx);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t ibz=0; ibz<nk_ibz; ++ibz) {
        E_ibz[ibz] = fabs(E[IDX2(ibz_to_bz[ibz],0,nb)]); // find min in next line
        for (index_t b=1; b<nb; ++b) E_ibz[ibz] = MIN(E_ibz[ibz], fabs(E[IDX2(ibz_to_bz[ibz],b,nb)]));
        E_ibz_idx[ibz] = ibz;
    }

    // argsort the energies (therefore the incides above!)
    mergesort_idx_r( E_ibz_idx, nk_ibz, E_ibz, diverge_omp_num_threads(), 0 );

    index_t* k_idx_fs_in_ibz = calloc( np_ibz, sizeof*k_idx_fs_in_ibz );

    // FS algo
    index_t* np_search_vec = calloc(np_search, sizeof*np_search_vec);
    claV2d_t* kmeans_points = calloc(np_search, sizeof*kmeans_points);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t p=0; p<np_search; ++p) {
        np_search_vec[p] = ibz_to_bz[E_ibz_idx[p]];
        claV3d_t kpt = kpt_from_mesh( mod->internals->kmesh, np_search_vec[p] );
        kmeans_points[p] = (claV2d_t){{kpt.v[0], kpt.v[1]}};
    }
    dkm_result_t kmeans_result = kmeans_lloyd( kmeans_points, np_search, np_ibz, DIVERGE_EPS_MESH, -1 );
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t p=0; p<np_ibz; ++p)
        k_idx_fs_in_ibz[p] = np_search_vec[find_closest_np_search_idx(kmeans_result.points[p],
                kmeans_points, np_search)];

    dkm_result_free( kmeans_result );
    free( np_search_vec );
    free( kmeans_points );

    // setup for full BZ FS
    claM3d_t sym[mod_n_sym]; memcpy(sym, mod->rs_symmetries, sizeof(double)*9*mod_n_sym);
    const claM3d_t latt = claM3d_map(mod->lattice[0]);

    VECTOR_INIT_CAP( index_t, k_idx_fs, np_ibz * mod_n_sym );
    VECTOR_RESIZE( k_idx_fs, np_ibz * mod_n_sym );
    #pragma omp parallel for collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t s=0; s<mod_n_sym; ++s)
    for (index_t p=0; p<np_ibz; ++p) {
        index_t kk = k_idx_fs_in_ibz[p];
        k_idx_fs[IDX2(s,p,np_ibz)] = kpt_to_idx(
                claM3d_matvec(sym[s],kpt_from_mesh(mod->internals->kmesh,kk)), latt, mod->nk );
    }

    // sort and make unique
    mergesort_idx( k_idx_fs, k_idx_fs_sz, diverge_omp_num_threads(), 0 );
    k_idx_fs_sz = unique_idx( k_idx_fs, k_idx_fs_sz );
    VECTOR_PUSH_BACK( k_idx_fs, -1 )
    VECTOR_SHRINK_TO_FIT( k_idx_fs );

    mpi_vrb_printf("found %li patches\n", k_idx_fs_sz-1);
    if (mod_n_sym != prev_mod_n_sym) mod->n_sym = prev_mod_n_sym; // reset symmetries

    free( ibz_to_bz );
    free( E_ibz );
    free( E_ibz_idx );
    free( k_idx_fs_in_ibz );

    return k_idx_fs;
}

// C interface of previous versions
void diverge_patching_find_fs_pts_C( diverge_model_t* mod, const double* E, index_t nb,
        index_t np_ibz, index_t np_search, index_t** pts, index_t* npts ) {
    *pts = diverge_patching_find_fs_pts( mod, E, nb, np_ibz, np_search );
    *npts = 0;
    while ((*pts)[*npts] != -1) { (*npts)++; }
}

mom_patching_t* diverge_patching_from_indices( diverge_model_t* mod, const index_t* pp, index_t np ) {
    mpi_vrb_printf("generate patching struct from patch indices\n");

    const index_t nk = kdim(mod->nk);
    complex128_t* kf = malloc(sizeof(*kf)*nk);
    double* dkf = (double*)kf;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk; ++k)
        dkf[2*k+0] = mod->internals->kmesh[3*k+0],
        dkf[2*k+1] = mod->internals->kmesh[3*k+1];

    double kbasis[3][3];
    diverge_model_generate_mom_basis(mod->lattice[0], kbasis[0]);
    complex128_t g1, g2;
    memcpy(&g1, kbasis[0], sizeof(complex128_t));
    memcpy(&g2, kbasis[1], sizeof(complex128_t));

    pgen_info_t* kf_patch_info = malloc(sizeof(*kf_patch_info)*nk);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk; ++k) {
        index_t thispatch = -1;
        double min_norm = 2.*M_PI;
        for (index_t p=0; p<np; ++p) {
            double _min_norm = periodic_norm( kf[k] - kf[pp[p]], g1, g2 );
            if (_min_norm < min_norm) {
                min_norm = _min_norm;
                thispatch = p;
            }
        }
        index_t howmany_patches = 0;
        for (index_t p=0; p<np; ++p) {
            double norm = periodic_norm( kf[k] - kf[pp[p]], g1, g2 );
            if (fabs(norm-min_norm) < NPATCH_ALMOST_ZERO) howmany_patches++;
        }
        if (howmany_patches == 1) {
            pgen_info_t result = { thispatch, howmany_patches, NULL };
            kf_patch_info[k] = result;
        } else {
            index_t* multi = malloc(sizeof(*multi) * howmany_patches);
            index_t hcnt = 0;
            for (index_t p=0; p<np; ++p) {
                double norm = periodic_norm( kf[k] - kf[pp[p]], g1, g2 );
                if (fabs(norm-min_norm) < NPATCH_ALMOST_ZERO)
                    multi[hcnt++] = p;
            }
            pgen_info_t result = { thispatch, howmany_patches, multi };
            kf_patch_info[k] = result;
        }
    }

    index_t* nkf_per_patch = calloc(np, sizeof(*nkf_per_patch));
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk; ++k) {
        if (kf_patch_info[k].howmany > 1) {
            for (index_t pi=0; pi<kf_patch_info[k].howmany; ++pi)
                #pragma omp atomic
                nkf_per_patch[kf_patch_info[k].multi[pi]]++;
        } else {
            #pragma omp atomic
            nkf_per_patch[kf_patch_info[k].p]++;
        }
    }

    vec_index_t* kf_per_patch = calloc(np, sizeof*kf_per_patch);
    for (index_t p=0; p<np; ++p)
        VECTOR_RESIZE( kf_per_patch[p].v, nkf_per_patch[p] );

    memset(nkf_per_patch, 0, sizeof(index_t)*np);

    for (index_t k=0; k<nk; ++k) {
        if (kf_patch_info[k].howmany > 1) {
            for (index_t pi=0; pi<kf_patch_info[k].howmany; ++pi) {
                index_t p = kf_patch_info[k].multi[pi];
                kf_per_patch[p].v[nkf_per_patch[p]++] = k;
            }
        } else {
            index_t p = kf_patch_info[k].p;
            kf_per_patch[p].v[nkf_per_patch[p]++] = k;
        }
    }
    free(nkf_per_patch);

    mom_patching_t* patch = (mom_patching_t*)calloc(1, sizeof(mom_patching_t));
    patch->n_patches = np;
    patch->patches = (index_t*)malloc(sizeof(index_t)*np);
    memcpy( patch->patches, pp, sizeof(index_t)*np );

    patch->weights = (double*)calloc(np, sizeof(double));
    patch->p_count = (index_t*)calloc(np, sizeof(index_t));
    patch->p_displ = (index_t*)calloc(np, sizeof(index_t));

    // set count and displ
    for (index_t p=0; p<np; ++p) {
        patch->p_count[p] = kf_per_patch[p].v_sz;
        patch->p_displ[p] = p == 0 ? 0 : patch->p_displ[p-1] + patch->p_count[p-1];
    }
    index_t total_refined = patch->p_count[np-1] + patch->p_displ[np-1];

    patch->p_map = (index_t*)calloc(total_refined, sizeof(index_t));
    patch->p_weights = (double*)calloc(total_refined, sizeof(double));

    // save fine weights for each kpt so we can 'choose' later on
    double* fine_weights = (double*)malloc(sizeof(double)*nk);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk; ++k)
        fine_weights[k] = 1./(double)kf_patch_info[k].howmany;

    // info no longer needed
    for (index_t k=0; k<nk; ++k) {
        if (kf_patch_info[k].multi)
            free(kf_patch_info[k].multi);
    }
    free(kf_patch_info);

    #pragma omp parallel for schedule(dynamic) num_threads(diverge_omp_num_threads())
    for (index_t p=0; p<np; ++p) {
        index_t displ = patch->p_displ[p];
        double total_weight = 0.;
        for (index_t d=0; d<patch->p_count[p]; ++d) {
            patch->p_map[displ+d] = kf_per_patch[p].v[d];
            total_weight += fine_weights[patch->p_map[displ+d]];
        }
        patch->weights[p] = total_weight;
    }
    // find total weight and normalize
    double tw = 0.;
    for (index_t p=0; p<np; ++p) {
        tw += patch->weights[p];
    }
    for (index_t p=0; p<np; ++p) {
        patch->weights[p] = 1./tw;
    }

    #pragma omp parallel for schedule(dynamic) num_threads(diverge_omp_num_threads())
    for (index_t p=0; p<np; ++p) {
        index_t displ = patch->p_displ[p];
        for (index_t d=0; d<patch->p_count[p]; ++d) {
            index_t kfull = patch->p_map[displ+d];
            index_t kpatch = patch->patches[p];
            patch->p_map[displ+d] = k1m2( kfull, kpatch, mod->nk );
        }
    }

    free( kf );

    // update p_weights
    for (index_t p=0; p<patch->n_patches; ++p) {
        for (index_t d=patch->p_displ[p]; d<patch->p_displ[p]+patch->p_count[p]; ++d) {
            index_t pk = k1p2( patch->patches[p], patch->p_map[d], mod->nk );
            patch->p_weights[d] = fine_weights[pk];
        }
    }

    free( fine_weights );

    for (index_t p=0; p<np; ++p) VECTOR_FREE( kf_per_patch[p].v );
    free( kf_per_patch );

    return patch;
}

// alpha: scaling, beta: significance, gamma: rel_limit
void diverge_patching_autofine( diverge_model_t* mod, mom_patching_t* patch,
        const double* energies, index_t nb, index_t ngroups, double alpha, double beta,
        double gamma ) {

    // set energies to internal ptr if NULL
    if (energies == NULL) {
        energies = mod->internals->E;
        nb = mod->n_orb * mod->n_spin;
    }

    mpi_vrb_printf("starting autofine using the $k$-means Lloyd algorithm\n");

    // setting up the variables...
    index_t nk_flat[3]; memcpy(nk_flat, mod->nk, sizeof(index_t)*3);
    const index_t nk = kdim(mod->nk);
    complex128_t* kmesh = (complex128_t*)malloc(sizeof(complex128_t)*nk);

    double* dkmesh = (double*)kmesh;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk; ++k)
    for (index_t d=0; d<2; ++d) dkmesh[IDX2(k,d,2)] = mod->internals->kmesh[IDX2(k,d,3)];

    double kbasis[3][3];
    diverge_model_generate_mom_basis(mod->lattice[0], kbasis[0]);
    complex128_t g1, g2;
    memcpy(&g1, kbasis[0], sizeof(complex128_t));
    memcpy(&g2, kbasis[1], sizeof(complex128_t));

    const index_t np = patch->n_patches;

    // setting up the groups
    index_t* groups = calloc(nk, sizeof*groups);
    double* E = calloc(nk, sizeof*E);

    for (index_t k=0; k<nk; ++k) {
        index_t b=0; E[k] = fabs( energies[IDX2(k,b,nb)] );
        for (b=1; b<nb; ++b) E[k] = MIN(E[k], fabs( energies[IDX2(k,b,nb)] ));
    }

    minmax_t pE = minmax_element( E, nk );
    for (index_t k=0; k<nk; ++k) groups[k] = group_scaling_function( *pE.min, *pE.max, E[k], ngroups, alpha );
    free( E );

    vec_index_t* new_patches = calloc(np, sizeof*new_patches);
    vec_double* new_weights = calloc(np, sizeof*new_weights);
    double* timings = calloc(np, sizeof*timings);

    #pragma omp parallel for schedule(dynamic) num_threads(diverge_omp_num_threads())
    for (int p=0; p<np; ++p) {
        index_t pcoord = patch->patches[p];
        index_t* pmap = patch->p_map + patch->p_displ[p];
        index_t pcnt = patch->p_count[p];

        group_t* grouped_map = gen_grouped_map( pmap, pcnt, pcoord, nk_flat, groups );
        index_t ngroups_local = grouped_map[(pcnt-1)].group+1;
        index_t* group_counts = gen_group_counts( grouped_map, pcnt, &ngroups_local );

        VECTOR_INIT( index_t, patches );
        VECTOR_INIT( double, weights );

        group_t* group_ptr = grouped_map;
        for (index_t g=0; g<ngroups_local; ++g) {
            const group_t* c_group_ptr = group_ptr;

            claV2d_t* vectors = calloc(group_counts[g], sizeof*vectors);
            for (index_t i=0; i<group_counts[g]; ++i)
                vectors[i] = claV2d_map( (double*)(kmesh + (*(group_ptr++)).ppcoord) );

            index_t ncl = num_clusters(g, group_counts[g], ngroups, beta);

            if (ncl < gamma*group_counts[g]) {
                double tick = diverge_mpi_wtime();
                dkm_result_t cluster_result = kmeans_lloyd( vectors, group_counts[g], ncl, DIVERGE_EPS_MESH, -1 );
                double tock = diverge_mpi_wtime();
                timings[p] += tock - tick;

                index_t* cluster_patches = calloc(cluster_result.k, sizeof*cluster_patches);
                double* cluster_weights = calloc(cluster_result.k, sizeof*cluster_weights);

                for (index_t i=0; i<cluster_result.k; ++i) {
                    cluster_patches[i] = closest_k_index( cluster_result.points[i], g1, g2, nk_flat );
                    cluster_weights[cluster_result.clusters[i]] += 1.0;
                }

                VECTOR_RESERVE( patches, patches_sz + cluster_result.k );
                VECTOR_RESERVE( weights, weights_sz + cluster_result.k );
                for (index_t i=0; i<cluster_result.k; ++i) {
                    VECTOR_PUSH_BACK( patches, cluster_patches[i] );
                    VECTOR_PUSH_BACK( weights, cluster_weights[i] );
                }

                dkm_result_free( cluster_result );
                free( cluster_patches );
                free( cluster_weights );
            } else {
                VECTOR_RESERVE( patches, patches_sz + group_counts[g] );
                VECTOR_RESERVE( weights, weights_sz + group_counts[g] );
                for (index_t i=0; i<group_counts[g]; ++i) {
                    VECTOR_PUSH_BACK(patches, c_group_ptr[i].ppcoord);
                    VECTOR_PUSH_BACK(weights, 1.0);
                }
            }

            free( vectors );
        }

        free( grouped_map );
        free( group_counts );

        new_patches[p].v = patches;
        new_patches[p].v_sz = patches_sz;
        new_patches[p].v_cap = patches_cap;

        new_weights[p].v = weights;
        new_weights[p].v_sz = weights_sz;
        new_weights[p].v_cap = weights_cap;
    }

    free(kmesh);

    index_t total_size = 0;
    index_t added_self = 0;
    for (index_t pi=0; pi<np; ++pi) {
        // find patch in refinement
        int found = 0;
        for (index_t i=0; i<new_patches[pi].v_sz; ++i)
            found += (patch->patches[pi] == new_patches[pi].v[i]);

        if (!found) {
            added_self++;
            VECTOR_PUSH_BACK( new_patches[pi].v, patch->patches[pi] );
            VECTOR_PUSH_BACK( new_weights[pi].v, 1.0 );
        }
        total_size += new_patches[pi].v_sz;
    }
    if (added_self)
        mpi_vrb_printf("added patch itself for %li times\n", added_self);

    patch->p_weights = realloc(patch->p_weights, sizeof(double)*total_size);
    patch->p_map = realloc(patch->p_map, sizeof(index_t)*total_size);
    index_t* pmap = patch->p_map;

    index_t cntr = 0;
    for (index_t p=0; p<np; ++p) {
        index_t p_idx = patch->patches[p];
        for (index_t pi=0; pi<new_patches[p].v_sz; ++pi) {
            *(pmap++) = k1m2( new_patches[p].v[pi], p_idx, nk_flat );
            patch->p_weights[cntr++] = new_weights[p].v[pi];
        }
        patch->p_count[p] = new_patches[p].v_sz;
        patch->p_displ[p] = (p == 0) ? 0 : patch->p_count[p-1] + patch->p_displ[p-1];
    }

    free( groups );
    for (index_t p=0; p<np; ++p) VECTOR_FREE( new_patches[p].v );
    free( new_patches );
    for (index_t p=0; p<np; ++p) VECTOR_FREE( new_weights[p].v );
    free( new_weights );

    diverge_patching_weights_equalize( patch );
    diverge_patching_mpi_equalize( patch );

    minmax_t pt = minmax_element( timings, np );
    mpi_tim_printf( "dkm min/max time: %.1fs, %.1fs\n", *pt.min, *pt.max );
    free( timings );

    index_t ntot = patch->p_count[patch->n_patches-1] + patch->p_displ[patch->n_patches-1],
            ktot = kdim(mod->nk);
    mpi_vrb_printf( "autofine done. using %li/%li points (%.1f%%)\n", ntot, ktot, 100.0*ntot/ktot );
}

void diverge_patching_symmetrize_refinement( diverge_model_t* mod, mom_patching_t* patch ) {
    mpi_vrb_printf("starting refinement symmetrizer\n");

    const index_t np = patch->n_patches;
    index_t nk_flat[3]; memcpy(nk_flat, mod->nk, sizeof(index_t)*3);
    claM3d_t latt = claM3d_map(mod->lattice[0]);

    index_t n_sym = labs(mod->n_sym);
    if (n_sym == 0) {
        mpi_wrn_printf("symmetrizing refinement is useless with n_sym == 0\n");
        return;
    }

    claM3d_t syms[n_sym]; memcpy(syms, mod->rs_symmetries, sizeof(double)*9*n_sym);
    if (n_sym == 1) {
        if (claM3d_norm(claM3d_sub(syms[0],claM3d_id())) < NPATCH_ALMOST_ZERO)
            mpi_wrn_printf("found only *one* symmetry that is *not* the identity\n");
        mpi_wrn_printf("symmetrizing refinement is useless with n_sym == 1\n");
        return;
    }

    index_t idsym = -1;
    for (index_t ss=0; ss<n_sym; ++ss)
        if (claM3d_norm(claM3d_sub(syms[ss],claM3d_id())) < NPATCH_ALMOST_ZERO)
            idsym = ss;
    if (idsym == -1) {
        mpi_wrn_printf("could not find the identity\n");
    } else {
        // erase identity
        for (int ss=idsym; ss<n_sym-1; ++ss) syms[ss] = syms[ss+1];
        n_sym--;
    }

    // find patching 'ibz'
    VECTOR_INIT( index_t, ibz );
    index_t* bz_to_ibz = calloc(np, sizeof*bz_to_ibz);
    for (index_t p=0; p<np; ++p) bz_to_ibz[p] = -1;

    cbyte_t* reached = bitary_alloc( np );
    index_t* patches = patch->patches;
    // count how large the IBZ is
    index_t ibz_count = 0;
    for (index_t i=0; i<np; ++i) {
        if (!bitary_get(reached,i)) {
            bitary_set(reached,i);
            VECTOR_PUSH_BACK( ibz, i );
            bz_to_ibz[i] = ibz_count;
            for (index_t ss=0; ss<n_sym; ++ss) {
                index_t candidate_k = kpt_to_idx( claM3d_matvec(syms[ss],
                            kpt_from_mesh(mod->internals->kmesh,patches[i])), latt, nk_flat );
                // find
                index_t candidate = -1;
                for (index_t p=0; p<np; ++p) if ((patches[p] == candidate_k) && (p != i)) { candidate = p; break; }
                // and set if found
                if (candidate != -1) {
                    if (!bitary_get(reached,candidate)) {
                        bitary_set(reached,candidate);
                        bz_to_ibz[candidate] = ibz_count;
                    }
                }
            }
            ibz_count++;
        }
    }
    free(reached);
    // For points with higher symmetry, we need to construct which symmetries
    // are present at these points
    vec_index_t* bz_which_sym = calloc(np, sizeof*bz_which_sym);
    for (index_t i=0; i<ibz_count; ++i) {
        VECTOR_PUSH_BACK(bz_which_sym[ibz[i]].v, -1);
        for (index_t ss=0; ss<n_sym; ++ss) {
            index_t candidate_k = kpt_to_idx( claM3d_matvec(syms[ss],
                        kpt_from_mesh(mod->internals->kmesh,patches[ibz[i]])), latt, nk_flat );
            // find
            index_t candidate = -1;
            for (index_t p=0; p<np; ++p) if ((patches[p] == candidate_k) && (p != i)) { candidate = p; break; }
            if (candidate != -1)
                VECTOR_PUSH_BACK( bz_which_sym[candidate].v, ss );
        }
    }

    // need to update patch->count, patch->displ, patch->map, patch->p_weights
    vec_index_t* p_new_map = calloc(np, sizeof*p_new_map);
    vec_double* p_new_p_weights = calloc(np, sizeof*p_new_p_weights);

    for (index_t ppp=0; ppp<np; ++ppp) {
        index_t ibz_idx = bz_to_ibz[ppp];
        vec_index_t ibz_sym_vec = bz_which_sym[ppp];
        index_t ibz_pat = ibz[ibz_idx];
        index_t* ibz_refine = patch->p_map + patch->p_displ[ibz_pat];
        double* ibz_refine_p_weights = patch->p_weights + patch->p_displ[ibz_pat];
        index_t ibz_refine_sz = patch->p_count[ibz_pat];
        // if this BZ point has low symmetry (i.e. can only be generated from
        // the IBZ with a single symmetry operation) proceed in the simple
        // manner, else we need to construct the symmetrized refinement
        VECTOR_INIT_CAP( index_t, new_map, ibz_sym_vec.v_sz * ibz_refine_sz );
        VECTOR_INIT_CAP( double, new_p_weights, ibz_sym_vec.v_sz * ibz_refine_sz );
        for (index_t i=0; i<ibz_sym_vec.v_sz; ++i) {
            for (index_t refine=0; refine<ibz_refine_sz; ++refine) {
                VECTOR_PUSH_BACK( new_p_weights, ibz_refine_p_weights[refine] );
                if (ibz_sym_vec.v[i] == -1) {
                    VECTOR_PUSH_BACK( new_map, ibz_refine[refine] );
                } else {
                    VECTOR_PUSH_BACK( new_map, kpt_to_idx( claM3d_matvec(syms[ibz_sym_vec.v[i]],
                                    kpt_from_mesh(mod->internals->kmesh,ibz_refine[refine])), latt, nk_flat) );
                }
            }
        }
        if (ibz_sym_vec.v_sz > 1) {
            VECTOR_INIT_CAP( patch_helper_t, new_map_combined, new_map_sz );
            for (index_t i=0; i<new_map_sz; ++i)
                VECTOR_PUSH_BACK( new_map_combined, (patch_helper_t){ new_map[i], new_p_weights[i] } );
            mergesort_patch_helper( new_map_combined, new_map_combined_sz, diverge_omp_num_threads(), 0 );
            new_map_combined_sz = unique_patch_helper( new_map_combined, new_map_combined_sz );

            VECTOR_RESIZE( new_map, new_map_combined_sz );
            VECTOR_RESIZE( new_p_weights, new_map_combined_sz );
            for (index_t i=0; i<new_map_combined_sz; ++i) {
                new_map[i] = new_map_combined[i].idx;
                new_p_weights[i] = new_map_combined[i].wght;
            }
            VECTOR_FREE( new_map_combined );
        }

        // copy these vectors
        p_new_map[ppp].v = new_map;
        p_new_map[ppp].v_sz = new_map_sz;
        p_new_map[ppp].v_cap = new_map_cap;

        p_new_p_weights[ppp].v = new_p_weights;
        p_new_p_weights[ppp].v_sz = new_p_weights_sz;
        p_new_p_weights[ppp].v_cap = new_p_weights_cap;

    }


    // store total weights of each patching point so we can reset them
    // accordingly
    double* weight_per_patch = calloc(np, sizeof*weight_per_patch);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t pp=0; pp<np; ++pp)
        for (index_t c=0; c<patch->p_count[pp]; ++c)
            weight_per_patch[pp] += patch->p_weights[patch->p_displ[pp]+c];

    // copy everything over to the patching struct
    index_t total_size = 0;
    for (index_t pp=0; pp<np; ++pp) total_size += p_new_map[pp].v_sz;
    patch->p_map = realloc(patch->p_map, sizeof(index_t)*total_size);
    patch->p_weights = realloc(patch->p_weights, sizeof(double)*total_size);
    patch->p_displ[0] = 0;
    patch->p_count[0] = p_new_map[0].v_sz;
    for (index_t ppp=1; ppp<np; ++ppp) {
        patch->p_count[ppp] = p_new_map[ppp].v_sz;
        patch->p_displ[ppp] = patch->p_count[ppp-1] + patch->p_displ[ppp-1];
        memcpy(patch->p_map + patch->p_displ[ppp], p_new_map[ppp].v, sizeof(index_t)*patch->p_count[ppp]);
        memcpy(patch->p_weights + patch->p_displ[ppp], p_new_p_weights[ppp].v, sizeof(double)*patch->p_count[ppp]);
    }

    // generate the new weights of every patch
    double* new_weight_per_patch = calloc(np, sizeof*new_weight_per_patch);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t pp=0; pp<np; ++pp)
        for (index_t c=0; c<patch->p_count[pp]; ++c)
            new_weight_per_patch[pp] += patch->p_weights[patch->p_displ[pp]+c];

    // and finally update the weights accordingly
    for (index_t pp=0; pp<np; ++pp)
        for (index_t c=0; c<patch->p_count[pp]; ++c)
            patch->p_weights[patch->p_displ[pp]+c] *= weight_per_patch[pp] / new_weight_per_patch[pp];

    // clean up resources
    for (index_t p=0; p<np; ++p) VECTOR_FREE(bz_which_sym[p].v);
    free( bz_which_sym );

    VECTOR_FREE( ibz );
    free( bz_to_ibz );

    for (index_t p=0; p<np; ++p) VECTOR_FREE(p_new_p_weights[p].v);
    free( p_new_p_weights );

    for (index_t p=0; p<np; ++p) VECTOR_FREE(p_new_map[p].v);
    free( p_new_map );

    free( weight_per_patch );
    free( new_weight_per_patch );

    diverge_patching_weights_equalize( patch );
    diverge_patching_mpi_equalize( patch );
}

void diverge_patching_free( mom_patching_t* patching ) {
    if (patching) {
        free(patching->patches);
        free(patching->weights);
        free(patching->p_count);
        free(patching->p_displ);
        free(patching->p_map);
        free(patching->p_weights);
        free(patching);
    } else {
        mpi_err_printf( "cannot free empty patching struct\n" );
    }
}


// serialization routines for MPI equalized autofined patching structs.
static void* memcpy_and_advance( void* dest, const void* src, size_t size ) {
    memcpy( dest, src, size );
    return (char*)dest + size;
}

static void gen_serial_size_ary( const mom_patching_t* P, index_t mapsize, size_t ary[8] ) {
    int idx = 0;
    ary[idx++] = sizeof(mom_patching_t);
    ary[idx++] = sizeof(index_t); // mapsize
    ary[idx++] = sizeof(index_t) * P->n_patches;
    ary[idx++] = sizeof(index_t) * P->n_patches;
    ary[idx++] = sizeof(index_t) * P->n_patches;
    ary[idx++] = sizeof(index_t) * mapsize;
    ary[idx++] = sizeof(double) * P->n_patches;
    ary[idx++] = sizeof(double) * mapsize;
}

static void patching_serialize( mom_patching_t* P, void** pdata, size_t* psize ) {
    index_t mapsize = P->p_displ[P->n_patches-1] + P->p_count[P->n_patches-1];
    size_t sizes[8];
    gen_serial_size_ary( P, mapsize, sizes );
    size_t full_size = 0;
    for (int i=0; i<8; ++i) full_size += sizes[i];
    void* data = malloc(full_size);
    void* data_src = data;
    int idx = 0;
    data = memcpy_and_advance( data, P, sizes[idx++] );
    data = memcpy_and_advance( data, &mapsize, sizes[idx++] );
    data = memcpy_and_advance( data, P->patches, sizes[idx++] );
    data = memcpy_and_advance( data, P->p_count, sizes[idx++] );
    data = memcpy_and_advance( data, P->p_displ, sizes[idx++] );
    data = memcpy_and_advance( data, P->p_map, sizes[idx++] );
    data = memcpy_and_advance( data, P->weights, sizes[idx++] );
    data = memcpy_and_advance( data, P->p_weights, sizes[idx++] );
    *pdata = data_src;
    *psize = full_size;
}

static void* malloc_and_advance_src( void** dest, const void* src, size_t size ) {
    *dest = malloc(size);
    memcpy( *dest, src, size );
    return (char*)src + size;
}

static void patching_deserialize( mom_patching_t* P, void* data ) {
    free(P->patches);
    free(P->p_count);
    free(P->p_displ);
    free(P->p_map);
    free(P->weights);
    free(P->p_weights);

    memcpy(P, data, sizeof(mom_patching_t));
    index_t mapsize;
    memcpy(&mapsize, (char*)data + sizeof(mom_patching_t), sizeof(index_t));

    size_t sizes[8];
    gen_serial_size_ary( P, mapsize, sizes );
    int idx = 2; // start at second index, 1st is patching struct, 2nd is
                 // full_size
    data = malloc_and_advance_src( (void**)&P->patches, (char*)data + sizeof(mom_patching_t)+sizeof(index_t), sizes[idx++] );
    data = malloc_and_advance_src( (void**)&P->p_count, data, sizes[idx++] );
    data = malloc_and_advance_src( (void**)&P->p_displ, data, sizes[idx++] );
    data = malloc_and_advance_src( (void**)&P->p_map, data, sizes[idx++] );
    data = malloc_and_advance_src( (void**)&P->weights, data, sizes[idx++] );
    data = malloc_and_advance_src( (void**)&P->p_weights, data, sizes[idx++] );
}

static void diverge_patching_mpi_equalize( mom_patching_t* patch ) {
    void* packet;
    size_t packet_size;
    patching_serialize( patch, &packet, &packet_size );

    diverge_mpi_bcast_bytes( &packet_size, sizeof(size_t), 0 );
    packet = realloc( packet, packet_size );
    diverge_mpi_bcast_bytes( packet, packet_size, 0 );

    patching_deserialize( patch, packet );
    free(packet);
}

static void diverge_patching_weights_equalize( mom_patching_t* P ) {
    index_t np = P->n_patches,
            *cnt = P->p_count,
            *dis = P->p_displ;
    double *w_fine = P->p_weights,
           *w_coarse = P->weights;
    double global_sum = 0.0;
    #pragma omp parallel for reduction(+:global_sum) num_threads(diverge_omp_num_threads())
    for (index_t p=0; p<np; ++p) {
        double local_sum = 0.0;
        for (index_t c=0; c<cnt[p]; ++c)
            local_sum += w_fine[dis[p] + c];
        global_sum += local_sum * w_coarse[p];
    }
    for (index_t p=0; p<np; ++p)
        w_coarse[p] /= global_sum;
}

static inline void limit_to( index_t max, index_t* idx ) {
    if (*idx < 0) *idx = 0;
    if (*idx >= max) *idx = max-1;
}

static inline index_t group_scaling_function( double E_min, double E_max, double E, int ngroups, double alpha ) {
    double E_norm = fabs((E - E_min)/(E_max-E_min));
    double val = pow(E_norm,alpha);
    index_t group = ngroups * val;
    limit_to( ngroups, &group );
    return group;
}

static inline index_t num_clusters( index_t group, index_t group_count, index_t ngroups, double beta ) {
    double g_significance = pow( 1. - (double)group / (double)(ngroups-1), beta );
    double g_weight = (double)group_count;
    index_t result = g_weight * g_significance;
    limit_to( group_count, &result );
    if (result < 1) result = 1;
    return result;
}

static inline claV2d_t to_basis( claV2d_t k, complex128_t g1, complex128_t g2 ) {
    const double *pg1 = (const double*)&g1,
                 *pg2 = (const double*)&g2;
    double G[] = { pg1[0], pg2[0],
                   pg1[1], pg2[1] };
    double G_inv[] = { G[3], -G[1],
                       -G[2], G[0] };
    double G_inv_norm = 1./(G[0]*G[3]-G[1]*G[2]);
    for (short i=0; i<4; ++i) G_inv[i] *= G_inv_norm;
    return (claV2d_t){{ G_inv[0] * k.v[0] + G_inv[1] * k.v[1],
                        G_inv[2] * k.v[0] + G_inv[3] * k.v[1] }};
}

static inline index_t closest_k_index( claV2d_t kpt, complex128_t g1, complex128_t g2, const index_t nk[3] ) {
    claV2d_t kpt_N = to_basis( kpt, g1, g2 );
    index_t kx = kpt_N.v[0] * nk[0] + 0.5,
            ky = kpt_N.v[1] * nk[1] + 0.5;
    limit_to( nk[0], &kx ), limit_to( nk[1], &ky );
    return ky + kx*nk[1];
}

static inline group_t* gen_grouped_map( index_t* pmap, index_t pcnt, index_t p_xy,
        const index_t nk_flat[3], const index_t* groups ) {
    group_t* grouped_map = calloc( pcnt, sizeof*grouped_map );
    for (index_t pp=0; pp<pcnt; ++pp) {
        index_t pp_xy = pmap[pp];
        index_t ppcoord = k1p2( pp_xy, p_xy, nk_flat );
        grouped_map[pp] = (group_t){ groups[ppcoord], ppcoord };
    }
    mergesort_group( grouped_map, pcnt, 1, 0 );
    return grouped_map;
}

static inline index_t* gen_group_counts( const group_t* grouped_map, index_t pcnt, index_t* ngroups ) {
    index_t ngroups_local = grouped_map[pcnt-1].group+1;
    index_t* group_counts = calloc(ngroups_local, sizeof*group_counts);
    for (index_t i=0; i<pcnt; ++i)
        group_counts[grouped_map[i].group]++;
    *ngroups = ngroups_local;
    return group_counts;
}

static inline index_t unique_patch_helper( patch_helper_t* arr, index_t size ) {
    if (size == 0) return 0;
    index_t j = 0;
    for (index_t i=1; i<size; ++i) if (arr[i].idx != arr[j].idx) arr[++j] = arr[i];
    return j+1;
}

static inline index_t kpt_to_idx( claV3d_t kpt, claM3d_t latt, const index_t nk[3] ) {
    claV3d_t kpt_real = claV3d_scale(claM3d_matvec(latt, kpt), 1./(2.*M_PI));
    index_t ipt[3];
    for (int c=0; c<3; ++c) {
        while (kpt_real.v[c] < 0)
            kpt_real.v[c] += 1.0;
        kpt_real.v[c] = fmod(kpt_real.v[c], 1.0);
        if (kpt_real.v[c] > 1.0 - NPATCH_ALMOST_ZERO)
            kpt_real.v[c] -= 1.0;
        ipt[c] = lround(kpt_real.v[c] * nk[c]);
    }
    index_t result = IDX3( ipt[0], ipt[1], ipt[2], nk[1], nk[2] );
    if (result >= kdim(nk)) {
        result %= kdim(nk);
        mpi_wrn_printf("map kpt to index out of PZ, fix with %% to not get SIGSEGV\n");
    }
    return result;
}

static index_t* find_ibz_pts( diverge_model_t* m, index_t* nibz ) {
    const index_t nk = kdim(m->nk), ns = m->n_sym;

    VECTOR_INIT_CAP( index_t, ibz, nk/ns );
    const claM3d_t latt = claM3d_map(m->lattice[0]);
    claM3d_t syms[ns]; memcpy(syms, m->rs_symmetries, sizeof(double)*9*ns);
    const double* kmesh = m->internals->kmesh;
    cbyte_t* reached = bitary_alloc(nk);
    for (index_t k=0; k<nk; ++k) {
        if (!bitary_get(reached, k)) {
            VECTOR_PUSH_BACK( ibz, k );
            for (index_t s=0; s<ns; ++s)
                bitary_set( reached, kpt_to_idx(claM3d_matvec(syms[s],kpt_from_mesh(kmesh,k)), latt, m->nk) );
        }
    }
    free(reached);
    *nibz = ibz_sz;
    return ibz;
}

static inline index_t find_closest_np_search_idx( claV2d_t mean_vec, const claV2d_t* kmeans_points, index_t N ) {
    if (N < 1) return 0;
    double norm2 = claV2d_norm2(claV2d_sub(mean_vec, kmeans_points[0]));
    index_t p_min = 1;
    for (index_t p=1; p<N; ++p) {
        double new_norm2 = claV2d_norm2(claV2d_sub(mean_vec, kmeans_points[p]));
        if (new_norm2 < norm2) {
            norm2 = new_norm2;
            p_min = p;
        }
    }
    return p_min;
}

static inline index_t unique_idx( index_t* arr, index_t size ) {
    if (size == 0) return 0;
    index_t j = 0;
    for (index_t i=1; i<size; ++i) if (arr[i] != arr[j]) arr[++j] = arr[i];
    return j+1;
}

static minmax_t minmax_element( const double* ary, index_t num ) {
    minmax_t result = {NULL, NULL};
    if (num <= 0) return result;
    double min = ary[0], max = ary[0];
    index_t imin = 0, imax = 0;
    for (index_t i=0; i<num; ++i) {
        if (ary[i] < min) min = ary[(imin = i)];
        if (ary[i] > max) max = ary[(imax = i)];
    }
    result.min = ary + imin;
    result.max = ary + imax;
    return result;
}

npatch_qvert_mat_t* diverge_patching_vertex( diverge_flow_step_t* s, char chan, index_t* nvq ) {
    if (!s || diverge_flow_step_type(s) != diverge_patch || !nvq) return NULL;
    return npatch_qvert_mat_from_vertex( s->mod->internals->patch_vertex,
            s->patch_loop, toupper(chan), islower(chan), 0/*dont sort*/, nvq,
            nvq/*dont sort*/ );
}
void diverge_patching_vertex_free( npatch_qvert_mat_t* vq, index_t nvq ) {
    if (vq) npatch_qvert_mat_free(vq, nvq);
}
