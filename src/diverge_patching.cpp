/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_patching.h"
#include "diverge_patching_helpers.h"
#include "diverge_momentum_gen.h"

#include "diverge_model.h"
#include "diverge_internals_struct.h"

#include "diverge_Eigen3.hpp"

#include "misc/mpi_functions.h"
#include "misc/kmeans_lloyd.hpp"
#include "misc/bitary.h"
#include "misc/cmath_helpers.h"

#include <array>

using std::array;
using std::tuple;
using std::get;
typedef array<double,2> dvec2_t;
typedef Eigen::Matrix<double,3,3,Eigen::RowMajor> CMat3d;

static vector<Mat3d> rs_syms( diverge_model_t* mod );
static Mat3d latt_matrix( diverge_model_t* mod );
static inline index_t kpt_to_idx( const Vec3d& kpt, const Mat3d& latt, const index_t nk[3] );
static inline Vec3d kpt_from_mesh( const double* kmesh, index_t k );

struct KmeshWrapper {
    KmeshWrapper( const double* ptr ) : ptr(ptr) {}
    const double *ptr;
    Vec3d operator[]( index_t k ) const { return Map<Vec3d>(const_cast<double*>(ptr) + 3*k); }
};

static vector<index_t> find_ibz_pts( diverge_model_t* mod );

// convenience functions to re-normalize and mpi-distribute the patching struct
static void diverge_patching_weights_equalize( mom_patching_t* patch );
static void diverge_patching_mpi_equalize( mom_patching_t* patch );


vector<index_t> diverge_patching_find_fs_pts( diverge_model_t* mod, const double* E, index_t nb, index_t np_ibz, index_t np_search ) {
    index_t mod_n_sym = MAX(mod->n_sym,1);
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

    index_t nk = kdim(mod->nk);
    // reset energy to automatic energy if not passed as parameter
    if (E == NULL)
        E = mod->internals->E;

    // find IBZ
    vector<index_t> ibz_to_bz = find_ibz_pts( mod );
    index_t nk_ibz = ibz_to_bz.size();

    // map BZ to IBZ
    vector<index_t> bz_to_ibz( nk, -1 );
    for (index_t ii=0; ii<nk_ibz; ++ii)
        bz_to_ibz[ibz_to_bz[ii]] = ii;

    // flattened minimum energies on IBZ
    vector<double> E_ibz(nk_ibz);
    // ibz indices for each energy
    vector<index_t> E_ibz_idx(nk_ibz);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t ibz=0; ibz<nk_ibz; ++ibz) {
        E_ibz[ibz] = Map<VecXd>( const_cast<double*>(E) + nb*ibz_to_bz[ibz], nb ).array().abs().minCoeff();
        E_ibz_idx[ibz] = ibz;
    }

    // argsort the energies (therefore the incides above!)
    std::sort( E_ibz_idx.begin(), E_ibz_idx.end(), [&](const index_t& i, const index_t& j)->bool{
        return E_ibz[i] < E_ibz[j];
    } );

    vector<index_t> k_idx_fs_in_ibz;
    // FS algo
    {
        vector<index_t> np_search_vec(np_search);
        vector<array<double,2>> kmeans_points(np_search);
        for (index_t p=0; p<np_search; ++p) {
            np_search_vec[p] = ibz_to_bz[E_ibz_idx[p]];
            Vec3d kpt = kpt_from_mesh( mod->internals->kmesh, np_search_vec[p] );
            kmeans_points[p] = { kpt(0), kpt(1) };
        }
        auto kmeans_result = dkm::kmeans_lloyd( kmeans_points, np_ibz );
        auto kmeans_means = std::get<0>(kmeans_result);
        auto kmeans_cluster = std::get<1>(kmeans_result);

        auto find_closest_np_search_idx = [&](index_t cluster_idx)->index_t{
            Vec2d mean_vec = Map<Vec2d>(kmeans_means[cluster_idx].data());
            double norm = 10.0;
            index_t p_min = -1;
            for (index_t p=0; p<np_search; ++p) {
                Vec2d np_vec = Map<Vec2d>(kmeans_points[p].data());
                double new_norm = (np_vec - mean_vec).norm();
                if (new_norm < norm) {
                    norm = new_norm;
                    p_min = p;
                }
            }
            return p_min;
        };

        for (index_t p=0; p<np_ibz; ++p)
            k_idx_fs_in_ibz.push_back(np_search_vec[find_closest_np_search_idx(p)]);
    }

    // setup for full BZ FS
    vector<Mat3d> sym = rs_syms(mod);
    Mat3d latt = latt_matrix(mod);

    vector<index_t> k_idx_fs;
    k_idx_fs.reserve( k_idx_fs_in_ibz.size() * sym.size() );
    for (uindex_t s=0; s<sym.size(); ++s)
        for (index_t kk: k_idx_fs_in_ibz)
            k_idx_fs.push_back( kpt_to_idx( sym[s] * kpt_from_mesh( mod->internals->kmesh, kk ), latt, mod->nk ) );

    // sort and make unique
    std::sort( k_idx_fs.begin(), k_idx_fs.end() );
    auto k_idx_fs_end = std::unique( k_idx_fs.begin(), k_idx_fs.end() );
    k_idx_fs.erase( k_idx_fs_end, k_idx_fs.end() );

    mpi_vrb_printf("found %lu patches\n", k_idx_fs.size());
    return k_idx_fs;
}

// C interface
void diverge_patching_find_fs_pts_C( diverge_model_t* mod, double* E, index_t nb,
        index_t np_ibz, index_t np_search, index_t** pts, index_t* npts ) {

    vector<index_t> xpts = diverge_patching_find_fs_pts( mod, E, nb, np_ibz, np_search );
    index_t* result = (index_t*)malloc(sizeof(index_t)*xpts.size());
    memcpy(result, xpts.data(), sizeof(index_t)*xpts.size());
    *pts = result;
    *npts = xpts.size();

}

struct pgen_info_t {
    index_t p;
    index_t howmany;
    index_t* multi;
};
typedef struct pgen_info_t pgen_info_t;

mom_patching_t* diverge_patching_from_indices( diverge_model_t* mod, const index_t* pp, index_t np ) {

    mpi_vrb_printf("generate patching struct from patch indices\n");

    index_t nk_flat[3];
    memcpy(nk_flat, mod->nk, sizeof(index_t)*3);
    const index_t nk = kdim(mod->nk);
    complex128_t* kf = (complex128_t*)malloc(sizeof(complex128_t)*nk);
    kpoint_union_t* kum = (kpoint_union_t*)mod->internals->kmesh;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk; ++k)
        kf[k] = kum[k].k;

    double kbasis[3][3];
    diverge_model_generate_mom_basis(mod->lattice, kbasis);
    complex128_t g1, g2;
    memcpy(&g1, kbasis[0], sizeof(complex128_t));
    memcpy(&g2, kbasis[1], sizeof(complex128_t));

    pgen_info_t* kf_patch_info = (pgen_info_t*)malloc(sizeof(pgen_info_t)*nk);
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
            index_t* multi = (index_t*)malloc(sizeof(index_t) * howmany_patches);
            index_t hcnt = 0;
            for (index_t p=0; p<np; ++p) {
                double norm = periodic_norm( kf[k] - kf[pp[p]], g1, g2 );
                if (fabs(norm-min_norm) < NPATCH_ALMOST_ZERO) {
                    multi[hcnt++] = p;
                }
            }
            pgen_info_t result = { thispatch, howmany_patches, multi };
            kf_patch_info[k] = result;
        }
    }

    index_t* nkf_per_patch = (index_t*)calloc(np, sizeof(index_t));
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

    vector<vector<index_t>> kf_per_patch(np);
    for (index_t p=0; p<np; ++p)
        kf_per_patch[p].resize( nkf_per_patch[p] );

    memset(nkf_per_patch, 0, sizeof(index_t)*np);

    for (index_t k=0; k<nk; ++k) {
        if (kf_patch_info[k].howmany > 1) {
            for (index_t pi=0; pi<kf_patch_info[k].howmany; ++pi) {
                index_t p = kf_patch_info[k].multi[pi];
                kf_per_patch[p][nkf_per_patch[p]++] = k;
            }
        } else {
            index_t p = kf_patch_info[k].p;
            kf_per_patch[p][nkf_per_patch[p]++] = k;
        }
    }
    free(nkf_per_patch);

    mom_patching_t* patch = (mom_patching_t*)calloc(sizeof(mom_patching_t), 1);
    patch->n_patches = np;
    patch->patches = (index_t*)malloc(sizeof(index_t)*np);
    memcpy( patch->patches, pp, sizeof(index_t)*np );

    patch->weights = (double*)calloc(sizeof(double), np);
    patch->p_count = (index_t*)calloc(sizeof(index_t), np);
    patch->p_displ = (index_t*)calloc(sizeof(index_t), np);

    // set count and displ
    for (index_t p=0; p<np; ++p) {
        patch->p_count[p] = kf_per_patch[p].size();
        patch->p_displ[p] = p == 0 ? 0 : patch->p_displ[p-1] + patch->p_count[p-1];
    }
    index_t total_refined = patch->p_count[np-1] + patch->p_displ[np-1];

    patch->p_map = (index_t*)calloc(sizeof(index_t), total_refined);
    patch->p_weights = (double*)calloc(sizeof(double), total_refined);

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
            patch->p_map[displ+d] = kf_per_patch[p][d];
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
    return patch;
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

void diverge_patching_mpi_equalize( mom_patching_t* patch ) {
    void* packet;
    size_t packet_size;
    patching_serialize( patch, &packet, &packet_size );

    diverge_mpi_bcast_bytes( &packet_size, sizeof(size_t), 0 );
    packet = realloc( packet, packet_size );
    diverge_mpi_bcast_bytes( packet, packet_size, 0 );

    patching_deserialize( patch, packet );
    free(packet);
}

void diverge_patching_weights_equalize( mom_patching_t* P ) {
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

static inline dvec2_t dvec2_from_complex128_t( const complex128_t& A ) {
    const double* A_ptr = (const double*)&A;
    return {*A_ptr, *(A_ptr+1)};
}

static inline void limit_to( index_t max, index_t* idx ) {
    if (*idx < 0) *idx = 0;
    if (*idx >= max) *idx = max-1;
}

static inline index_t group_scaling_function( double E_min, double E_max, double E, int ngroups, double alpha ) {
    double E_norm = std::abs((E - E_min)/(E_max-E_min));
    double val = cmath_pow(E_norm,alpha);
    index_t group = ngroups * val;
    limit_to( ngroups, &group );
    return group;
}

static inline index_t num_clusters( index_t group, index_t group_count, index_t ngroups, double beta ) {
    double g_significance = cmath_pow( 1. - (double)group / (double)(ngroups-1), beta );
    double g_weight = (double)group_count;
    index_t result = g_weight * g_significance;
    limit_to( group_count, &result );
    if (result < 1) result = 1;
    return result;
}

static inline dvec2_t to_basis( const dvec2_t& k, const complex128_t& g1, const complex128_t& g2 ) {
    const double *pg1 = (const double*)&g1,
                 *pg2 = (const double*)&g2;
    double G[] = { pg1[0], pg2[0],
                   pg1[1], pg2[1] };
    double G_inv[] = { G[3], -G[1],
                       -G[2], G[0] };
    double G_inv_norm = 1./(G[0]*G[3]-G[1]*G[2]);
    for (short i=0; i<4; ++i) G_inv[i] *= G_inv_norm;
    return { G_inv[0] * k[0] + G_inv[1] * k[1],
             G_inv[2] * k[0] + G_inv[3] * k[1] };
}

static inline index_t closest_k_index( const dvec2_t& kpt, complex128_t g1, complex128_t g2, const index_t nk_flat[3] ) {
    dvec2_t kpt_N = to_basis( kpt, g1, g2 );
    index_t kx = kpt_N[0] * nk_flat[0] + 0.5,
            ky = kpt_N[1] * nk_flat[1] + 0.5;
    limit_to( nk_flat[0], &kx );
    limit_to( nk_flat[1], &ky );
    return ky + kx*nk_flat[1];
}

static inline vector<array<index_t,2>> gen_grouped_map( index_t* pmap, index_t pcnt, index_t p_xy, const index_t nk_flat[3], const vector<index_t>& groups ) {
    vector<array<index_t,2>> grouped_map(pcnt);
    for (index_t pp=0; pp<pcnt; ++pp) {
        index_t pp_xy = pmap[pp];
        index_t ppcoord = k1p2( pp_xy, p_xy, nk_flat );
        grouped_map.at(pp) = array<index_t,2> { groups.at(ppcoord), ppcoord };
    }
    std::sort( grouped_map.begin(), grouped_map.end(),
            []( const array<index_t,2>& a, const array<index_t,2>& b ){ return a[0] < b[0]; } );
    return grouped_map;
}

static inline vector<index_t> gen_group_counts( const vector<array<index_t,2>>& grouped_map ) {
    index_t ngroups_local = grouped_map[grouped_map.size()-1][0]+1;
    vector<index_t> group_counts(ngroups_local,0);
    for (const array<index_t,2>& gm: grouped_map) {
        group_counts.at(gm[0])++;
    }
    return group_counts;
}

// alpha: scaling
// beta: significance
// gamma: rel_limit
void diverge_patching_autofine( diverge_model_t* mod, mom_patching_t* patch,
        const double* energies, index_t nb, index_t ngroups, double alpha, double beta,
        double gamma ) {

    // set energies to internal ptr if NULL
    if (energies == NULL)
        energies = mod->internals->E;

    mpi_vrb_printf("starting autofine using the $k$-means Lloyd algorithm\n");

    typedef vector<dvec2_t> points_t;
    typedef vector<uint32_t> indices_t;
    typedef tuple<points_t, indices_t> dkm_result_t;

    // setting up the variables...
    index_t nk_flat[3];
    memcpy(nk_flat, mod->nk, sizeof(index_t)*3);
    const index_t nk = kdim(mod->nk);
    complex128_t* kmesh = (complex128_t*)malloc(sizeof(complex128_t)*nk);
    kpoint_union_t* kum = (kpoint_union_t*)mod->internals->kmesh;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<nk; ++k)
        kmesh[k] = kum[k].k;

    double kbasis[3][3];
    diverge_model_generate_mom_basis(mod->lattice, kbasis);
    complex128_t g1, g2;
    memcpy(&g1, kbasis[0], sizeof(complex128_t));
    memcpy(&g2, kbasis[1], sizeof(complex128_t));

    const index_t np = patch->n_patches;

    // setting up the groups
    vector<index_t> groups(nk);
    vector<double> E(nk);
    for (index_t k=0; k<nk; ++k) {
        index_t b=0;
        E[k] = std::abs( energies[k*nb + b] );
        for (b=1; b<nb; ++b)
            E[k] = std::min(E[k], std::abs( energies[k*nb + b] ));
    }
    const auto [pE_min, pE_max] = std::minmax_element( E.begin(), E.end() );
    for (index_t k=0; k<nk; ++k)
        groups[k] = group_scaling_function( *pE_min, *pE_max, E[k], ngroups, alpha );
    E.clear();

    vector<vector<index_t>> new_patches(np);
    vector<vector<double>> new_weights(np);
    vector<double> timings(np, 0.0);
    #pragma omp parallel for schedule(dynamic) num_threads(diverge_omp_num_threads())
    for (int p=0; p<np; ++p) {
        index_t pcoord = patch->patches[p];
        index_t* pmap = patch->p_map + patch->p_displ[p];
        index_t pcnt = patch->p_count[p];

        vector<array<index_t,2>> grouped_map = gen_grouped_map( pmap, pcnt, pcoord, nk_flat, groups );
        index_t ngroups_local = grouped_map.at(pcnt-1)[0]+1;
        vector<index_t> group_counts = gen_group_counts( grouped_map );

        vector<index_t> patches;
        vector<double> weights;
        array<index_t,2>* group_ptr = grouped_map.data();
        for (index_t g=0; g<ngroups_local; ++g) {
            const array<index_t,2>* c_group_ptr = group_ptr;

            points_t vectors(group_counts.at(g));
            for (index_t i=0; i<group_counts.at(g); ++i) {
                vectors.at(i) = dvec2_from_complex128_t( kmesh[ (*(group_ptr++))[1] ] );
            }

            index_t ncl = num_clusters(g, group_counts.at(g), ngroups, beta);

            if (ncl < gamma*group_counts.at(g)) {

                double tick = diverge_mpi_wtime();
                dkm_result_t cluster_result = dkm::kmeans_lloyd( vectors, ncl );
                double tock = diverge_mpi_wtime();
                timings[p] += tock - tick;
                points_t cluster_points = get<0>(cluster_result);
                indices_t cluster_idxs = get<1>(cluster_result);
                vector<index_t> cluster_patches(cluster_points.size());
                vector<double> cluster_weights(cluster_points.size());
                for (unsigned i=0; i<cluster_points.size(); ++i)
                    cluster_patches.at(i) = closest_k_index(cluster_points.at(i), g1, g2, nk_flat);
                for (unsigned i=0; i<cluster_idxs.size(); ++i) {
                    uint32_t cl = cluster_idxs[i];
                    double w = 1.0;
                    cluster_weights.at(cl) += w;
                }

                patches.reserve(patches.size() + cluster_patches.size());
                weights.reserve(weights.size() + cluster_weights.size());
                for (index_t cp: cluster_patches)
                    patches.push_back(cp);
                for (double cw: cluster_weights)
                    weights.push_back(cw);
            } else {
                patches.reserve(patches.size() + group_counts.at(g));
                weights.reserve(weights.size() + group_counts.at(g));
                for (index_t i=0; i<group_counts.at(g); ++i) {
                    patches.push_back(c_group_ptr[i][1]);
                    weights.push_back(1);
                }
            }
        }

        new_patches.at(p) = patches;
        new_weights.at(p) = weights;
    }
    free(kmesh);

    index_t total_size = 0;
    index_t added_self = 0;
    for (uindex_t pi=0; pi<new_patches.size(); ++pi) {
        auto fres = std::find( new_patches[pi].begin(), new_patches[pi].end(), patch->patches[pi] );
        uindex_t idx = std::distance( new_patches[pi].begin(), fres );
        if (idx == new_patches[pi].size()) {
            added_self++;
            new_patches[pi].push_back(patch->patches[pi]);
            new_weights[pi].push_back(1.0);
        }
        total_size += new_patches[pi].size();
    }
    if (added_self)
        mpi_vrb_printf("added patch itself for %li times\n", added_self);

    patch->p_weights = (double*)realloc(patch->p_weights, sizeof(double)*total_size);
    patch->p_map = (index_t*)realloc(patch->p_map, sizeof(index_t)*total_size);
    index_t* pmap = patch->p_map;

    index_t cntr = 0;
    for (index_t p=0; p<np; ++p) {
        index_t p_idx = patch->patches[p];
        for (unsigned pi=0; pi<new_patches[p].size(); ++pi) {
            *(pmap++) = k1m2( new_patches[p][pi], p_idx, nk_flat );
            patch->p_weights[cntr++] = new_weights[p][pi];
        }
        patch->p_count[p] = new_patches.at(p).size();
        patch->p_displ[p] = (p == 0) ? 0 : patch->p_count[p-1] + patch->p_displ[p-1];
    }
    diverge_patching_weights_equalize( patch );
    diverge_patching_mpi_equalize( patch );

    auto [ptmin, ptmax] = std::minmax_element( timings.begin(), timings.end() );
    mpi_tim_printf( "dkm min/max time: %.1fs, %.1fs\n", *ptmin, *ptmax );
    index_t ntot = patch->p_count[patch->n_patches-1] + patch->p_displ[patch->n_patches-1],
            ktot = kdim(mod->nk);
    mpi_vrb_printf( "autofine done. using %li/%li points (%.1f%%)\n", ntot, ktot,
            100.0*(double)ntot/(double)ktot );
}

void diverge_patching_symmetrize_refinement( diverge_model_t* mod, mom_patching_t* patch ) {

    mpi_vrb_printf("starting refinement symmetrizer\n");

    const index_t np = patch->n_patches;
    index_t nk_flat[3];
    memcpy(nk_flat, mod->nk, sizeof(index_t)*3);
    Mat3d latt = latt_matrix(mod);

    if (mod->n_sym == 0) {
        mpi_wrn_printf("symmetrizing refinement is useless with n_sym == 0\n");
        return;
    }
    vector<Mat3d> sym = rs_syms(mod);
    if (mod->n_sym == 1) {
        if ((sym[0] - Mat3d::Identity()).norm() < NPATCH_ALMOST_ZERO)
            mpi_wrn_printf("found only *one* symmetry that is *not* the identity\n");
        mpi_wrn_printf("symmetrizing refinement is useless with n_sym == 1\n");
        return;
    }
    index_t idsym = -1;
    for (uindex_t ss=0; ss<sym.size(); ++ss)
        if ((sym[ss] - Mat3d::Identity()).norm() < NPATCH_ALMOST_ZERO)
            idsym = ss;
    if (idsym == -1) {
        mpi_wrn_printf("could not find the identity\n");
    } else {
        sym.erase( sym.begin() + idsym );
    }

    cbyte_t* reached = bitary_alloc( np );
    KmeshWrapper Kmesh(mod->internals->kmesh);

    // find patching 'ibz'
    vector<index_t> ibz;
    vector<index_t> bz_to_ibz(np, -1);
    index_t* patches = patch->patches;
    // count how large the IBZ is
    index_t ibz_count = 0;
    for (index_t i=0; i<np; ++i) {
        if (!bitary_get(reached,i)) {
            bitary_set(reached,i);
            ibz.push_back(i);
            bz_to_ibz[i] = ibz_count;
            for (uindex_t ss=0; ss<sym.size(); ++ss) {
                index_t candidate_k = kpt_to_idx( sym[ss] * Kmesh[patches[i]], latt, nk_flat );
                index_t* found = std::find( patches, patches+np, candidate_k );
                index_t candidate = found - patches;
                if (candidate < np && candidate >= 0 && candidate != i) {
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
    vector<vector<int>> bz_which_sym(np);
    for (index_t i=0; i<ibz_count; ++i) {
        bz_which_sym[ibz[i]].push_back(-1);
        for (uindex_t ss=0; ss<sym.size(); ++ss) {
            index_t candidate_k = kpt_to_idx( sym[ss] * Kmesh[patches[ibz[i]]], latt, nk_flat );
            index_t* found = std::find( patches, patches+np, candidate_k );
            index_t candidate = found - patches;
            if (candidate < np && candidate >= 0 && candidate != i)
                bz_which_sym[candidate].push_back(ss);
        }
    }

    // need to update patch->count, patch->displ, patch->map, patch->p_weights
    vector<vector<index_t>> p_new_map(np);
    vector<vector<double>> p_new_p_weights(np);
    for (index_t ppp=0; ppp<np; ++ppp) {
        index_t ibz_idx = bz_to_ibz[ppp];
        vector<int> ibz_sym_vec = bz_which_sym[ppp];
        index_t ibz_pat = ibz[ibz_idx];
        index_t* ibz_refine = patch->p_map + patch->p_displ[ibz_pat];
        double* ibz_refine_p_weights = patch->p_weights + patch->p_displ[ibz_pat];
        index_t ibz_refine_sz = patch->p_count[ibz_pat];
        // if this BZ point has low symmetry (i.e. can only be generated from
        // the IBZ with a single symmetry operation) proceed in the simple
        // manner, else we need to construct the symmetrized refinement
        vector<index_t> new_map;
        new_map.reserve( ibz_sym_vec.size() * ibz_refine_sz );
        vector<double> new_p_weights;
        new_p_weights.reserve( ibz_sym_vec.size() * ibz_refine_sz );
        for (int ibz_sym: ibz_sym_vec) {
            for (index_t refine=0; refine<ibz_refine_sz; ++refine) {
                new_p_weights.push_back( ibz_refine_p_weights[refine] );
                if (ibz_sym == -1)
                    new_map.push_back(ibz_refine[refine]);
                else
                    new_map.push_back(kpt_to_idx( sym[ibz_sym] * Kmesh[ibz_refine[refine]], latt, nk_flat ));
            }
        }
        if (ibz_sym_vec.size() > 1u) {
            typedef std::pair<index_t, double> patch_helper_t;
            vector<patch_helper_t> new_map_combined( new_map.size() );
            for (size_t i=0; i<new_map_combined.size(); ++i)
                new_map_combined[i] = { new_map[i], new_p_weights[i] };
            std::sort( new_map_combined.begin(), new_map_combined.end(),
                    [&](const patch_helper_t& A, const patch_helper_t& B)->bool{
                        return A.first < B.first;
                    });
            auto new_map_combined_end = std::unique( new_map_combined.begin(), new_map_combined.end(),
                    [&](const patch_helper_t& A, const patch_helper_t& B)->bool{
                        return A.first == B.first;
                    });
            new_map_combined.erase( new_map_combined_end, new_map_combined.end() );
            new_map.resize( new_map_combined.size() );
            new_p_weights.resize( new_map_combined.size() );
            for (size_t i=0; i<new_map_combined.size(); ++i) {
                new_map[i] = new_map_combined[i].first;
                new_p_weights[i] = new_map_combined[i].second;
            }
        }
        p_new_map[ppp] = new_map;
        p_new_p_weights[ppp] = new_p_weights;
    }

    // store total weights of each patching point so we can reset them
    // accordingly
    vector<double> weight_per_patch(np, 0.0);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t pp=0; pp<np; ++pp)
        for (index_t c=0; c<patch->p_count[pp]; ++c)
            weight_per_patch[pp] += patch->p_weights[patch->p_displ[pp]+c];

    // copy everything over to the patching struct
    index_t total_size = 0;
    for (const vector<index_t>& v: p_new_map) total_size += v.size();
    patch->p_map = (index_t*)realloc(patch->p_map, sizeof(index_t)*total_size);
    patch->p_weights = (double*)realloc(patch->p_weights, sizeof(double)*total_size);
    patch->p_displ[0] = 0;
    patch->p_count[0] = p_new_map[0].size();
    for (index_t ppp=1; ppp<np; ++ppp) {
        patch->p_count[ppp] = p_new_map[ppp].size();
        patch->p_displ[ppp] = patch->p_count[ppp-1] + patch->p_displ[ppp-1];
        memcpy(patch->p_map + patch->p_displ[ppp], p_new_map[ppp].data(),
                sizeof(index_t)*patch->p_count[ppp]);
        memcpy(patch->p_weights + patch->p_displ[ppp], p_new_p_weights[ppp].data(),
                sizeof(double)*patch->p_count[ppp]);
    }

    // generate the new weights of every patch
    vector<double> new_weight_per_patch(np, 0.0);
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t pp=0; pp<np; ++pp)
        for (index_t c=0; c<patch->p_count[pp]; ++c)
            new_weight_per_patch[pp] += patch->p_weights[patch->p_displ[pp]+c];

    // and finally update the weights accordingly
    for (index_t pp=0; pp<np; ++pp)
        for (index_t c=0; c<patch->p_count[pp]; ++c)
            patch->p_weights[patch->p_displ[pp]+c] *= weight_per_patch[pp] / new_weight_per_patch[pp];

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

static vector<Mat3d> rs_syms( diverge_model_t* mod ) {
    if (mod->n_sym < 1)
        return {Mat3d::Identity()};
    vector<Mat3d> sym;
    for (index_t s=0; s<mod->n_sym; ++s)
        sym.push_back( Map<CMat3d>((double*)(&mod->rs_symmetries[s])) );
    return sym;
}

static Mat3d latt_matrix( diverge_model_t* mod ) {
    Mat3d Amatrix;
    memcpy( Amatrix.data(), mod->lattice, sizeof(double)*9 );
    return Amatrix;
}

static inline index_t kpt_to_idx( const Vec3d& kpt, const Mat3d& latt, const index_t nk[3] ) {
    Vec3d kpt_real = latt.transpose() * kpt / (2.*M_PI);
    index_t ipt[3];
    for (int c=0; c<3; ++c) {
        while (kpt_real[c] < 0)
            kpt_real[c] += 1.0;
        kpt_real[c] = fmod(kpt_real[c], 1.0);
        if (kpt_real[c] > 1.0 - NPATCH_ALMOST_ZERO)
            kpt_real[c] -= 1.0;
        ipt[c] = lround(kpt_real[c] * nk[c]);
    }
    index_t result = IDX3( ipt[0], ipt[1], ipt[2], nk[1], nk[2] );
    if (result >= kdim(nk)) {
        result %= kdim(nk);
        mpi_wrn_printf("map kpt to index out of PZ, fix with %% to not get SIGSEGV\n");
    }
    return result;
}

static inline Vec3d kpt_from_mesh( const double* kmesh, index_t k ) {
    return Map<Vec3d>(const_cast<double*>(kmesh)+3*k);
}

static vector<index_t> find_ibz_pts( diverge_model_t* mod ) {
    vector<Mat3d> sym = rs_syms(mod);

    const index_t ns = sym.size(),
                  nk = kdim(mod->nk);

    vector<index_t> ibz;
    ibz.reserve( nk / ns );

    Mat3d latt = latt_matrix( mod );
    const double* kmesh = mod->internals->kmesh;
    cbyte_t* reached = bitary_alloc(nk);
    for (index_t k=0; k<nk; ++k) {
        if (!bitary_get(reached, k)) {
            ibz.push_back(k);
            const Vec3d kpt = kpt_from_mesh( kmesh, k );
            for (index_t s=0; s<ns; ++s)
                bitary_set( reached, kpt_to_idx( sym[s] * kpt, latt, mod->nk ) );
        }
    }
    free(reached);
    return ibz;
}

