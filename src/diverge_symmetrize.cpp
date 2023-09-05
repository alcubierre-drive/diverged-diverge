#include "diverge_model.h"
#include "diverge_symmetrize.h"
#include "diverge_internals_struct.h"
#include "diverge_momentum_gen.h"
#include "diverge_Eigen3.hpp"
#include <vector>
#include <cassert>

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
        a[i] += abs(lround(a[i])) + 3;
    Vec3d ai = (a.array() * nk.array().cast<double>()).round().matrix();

    *idx = IDX3( lround(ai(0))%nk(0), lround(ai(1))%nk(1), lround(ai(2))%nk(2), nk(1), nk(2));
    *k_point = Map<Vec3d>( mesh+3*(*idx) );
    return (*idx < (nk(0)*nk(1)*nk(2)));
}

static inline bool approxEQ(MatXcd A, MatXcd B) {
    return (A-B).lpNorm<Eigen::Infinity>() < 1e-7;
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
    mpi_vrb_printf("generate symmetries \n")

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

    #pragma omp parallel for schedule(static) collapse(2)
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
    #pragma omp parallel for schedule(static) collapse(2)
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

    // orbital maps
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

    #pragma omp parallel for collapse(2)
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
            if((trial-to).squaredNorm() < 1e-6) {
                maps_to = o2;
                mapped_pos[o1+n_orb*s] = to_trial;
                bey_UC = ( to - to_trial );
                assert((bey_UC-(in*UC1+im*UC2+iq*UC3)).norm() < 1e-10);
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
            if(std::abs(overlap) > 1e-5 && (mapped_pos[o1+n_orb*s] - pos).norm() < 1e-8) {
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
        if(sum < 0.99999) {
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
}
