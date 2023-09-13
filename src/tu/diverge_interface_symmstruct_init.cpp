/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_interface.hpp"
#include "../diverge_momentum_gen.h"
#include "../diverge_symmetrize.h"

typedef Eigen::Matrix<double,3,3,Eigen::RowMajor> CMat3d;
typedef Eigen::Matrix<complex128_t,-1,-1,Eigen::RowMajor> CMatXcd;

static inline bool map_back_to_BZ(Vec3d &k_point,
        index_t* helper,diverge_model_t* model) {
    Map<Vec3d> UC1(model->lattice[0]);
    Map<Vec3d> UC2(model->lattice[1]);
    Map<Vec3d> UC3(model->lattice[2]);
    double a0 = k_point.dot(UC1)/(2.0*M_PI);
    double a1 = k_point.dot(UC2)/(2.0*M_PI);
    double a2 = k_point.dot(UC3)/(2.0*M_PI);
    if(std::abs(a0)>0.5+1e-12){a0  -= (double)round(a0);}
    if(std::abs(a1)>0.5+1e-12){a1  -= (double)round(a1);}
    if(std::abs(a2)>0.5+1e-12){a2  -= (double)round(a2);}
    if (a0 < -1e-12) a0 += 1.0;
    if (a1 < -1e-12) a1 += 1.0;
    if (a2 < -1e-12) a2 += 1.0;

    *helper = IDX3(round(std::abs(a0)*model->nk[0]),round(std::abs(a1)*model->nk[1]),
                    round(std::abs(a2)*model->nk[2]),model->nk[1],model->nk[2]);

    Map<Vec3d> kpnt(model->internals->kmesh+*helper*3);
    double kbasis[3][3];
    diverge_model_generate_mom_basis(model->lattice,kbasis);
    Map<Vec3d> mUC1(kbasis[0]); Map<Vec3d> mUC2(kbasis[1]);
    Map<Vec3d> mUC3(kbasis[2]);
    k_point = a0 * mUC1 + a1 * mUC2 + a2 * mUC3;
    return (*helper < ((tu_data_t*)(model->internals->tu_data))->nk
                && (k_point - kpnt).norm() < 1e-6);
}

static inline bool approxEQ(MatXcd A, MatXcd B) { return (A-B).lpNorm<Eigen::Infinity>() < 1e-7; }

bool init_inv_symmerty_map(diverge_model_t* model) {
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);
    symmstruct_t* symm = tu_data->symm;
    complex128_t* orb_symmetries = (complex128_t*)model->orb_symmetries;
    double* rs_symmetries = (double*)model->rs_symmetries;
    index_t n_orb = model->n_orb;
    index_t n_spin = model->n_spin;
    index_t& n_sym = model->n_sym;
    symm->inv_symmetry.resize(n_sym);
    index_t mi = n_orb * n_spin;
    vector<index_t> missing_symmetries;
    CMat3d id_rs = CMat3d::Identity();
    CMatXcd id_orb = CMatXcd::Identity(mi,mi);
    bool added_symm = false;
    index_t which_is_identity = -1;
    for(index_t sym = 0; sym < n_sym; ++sym) {
        Map<CMat3d> sym_rs(rs_symmetries+sym*9);
        Map<CMatXcd> sym_orb((orb_symmetries)+sym*mi*mi,mi,mi);
        if(approxEQ(sym_rs,id_rs) && approxEQ(sym_orb,id_orb)) {
            which_is_identity = sym;
        }
        bool found = false;
        index_t symt = 0;
        while( symt < n_sym && !found) {
            Map<CMat3d> sym_rst(rs_symmetries+symt*9);
            Map<CMatXcd> sym_orbt(orb_symmetries+symt*mi*mi,mi,mi);

            if(approxEQ(sym_rs*sym_rst,id_rs) && approxEQ(sym_orb*sym_orbt,id_orb)) {
                symm->inv_symmetry[sym] = symt;
                found = true;
            }
            symt+=1;
        }
        if(found == false)
            missing_symmetries.push_back(sym);
    }
    if(missing_symmetries.size() != 0) {
        // copy old arrays
        added_symm = true;
        complex128_t* orbsymcpy = (complex128_t*)calloc(n_sym * mi *mi, sizeof(complex128_t));
        memcpy(orbsymcpy, orb_symmetries, n_sym * mi*mi * sizeof(complex128_t));
        index_t n_symold = n_sym;
        n_sym += (index_t)missing_symmetries.size();
        vector<index_t> syms;
        syms.resize(n_sym);
        memcpy(syms.data(),symm->inv_symmetry.data(), n_symold*sizeof(index_t));
        free(model->orb_symmetries);
        model->orb_symmetries = (complex128_t*)calloc(n_sym * mi * mi, sizeof(complex128_t));
        orb_symmetries = model->orb_symmetries;
        memcpy(orb_symmetries,orbsymcpy, n_symold * mi*mi * sizeof(complex128_t));
        free(orbsymcpy);
        for (index_t i = 0; i < (index_t)missing_symmetries.size();++i) {
            index_t sym = missing_symmetries[i];
            mpi_wrn_printf("symmetry %d was missing an inverse, got %d syms \n", sym, n_sym);
            Map<CMat3d> sym_rs(rs_symmetries+sym*9);
            Map<CMatXcd> sym_orb(orb_symmetries+sym*mi*mi,mi,mi);
            Map<CMat3d> sym_rsm(rs_symmetries+(n_symold + i)*9);
            Map<CMatXcd> sym_orbm(orb_symmetries+(n_symold +i)*mi*mi,mi,mi);
            sym_rsm = sym_rs.inverse();
            sym_orbm = sym_orb.inverse();
            syms[sym] = n_symold + i;
            syms[n_symold + i] = sym;
        }
        symm->inv_symmetry.resize(n_sym);
        memcpy(symm->inv_symmetry.data(),syms.data(),n_sym*sizeof(index_t));
    }
    if(which_is_identity == -1) {
        mpi_wrn_printf("identity missing, will be added now \n");
        // copy old arrays
        added_symm = true;
        complex128_t* orbsymcpy = (complex128_t*)calloc(n_sym * mi *mi, sizeof(complex128_t));
        memcpy(orbsymcpy, orb_symmetries, n_sym * mi*mi * sizeof(complex128_t));
        index_t n_symold = n_sym;
        n_sym += 1;
        vector<index_t> syms;
        syms.resize(n_sym);
        memcpy(syms.data(),symm->inv_symmetry.data(), n_symold*sizeof(index_t));
        free(model->orb_symmetries);
        model->orb_symmetries = (complex128_t*)calloc(n_sym * mi * mi, sizeof(complex128_t));
        orb_symmetries = model->orb_symmetries;
        memcpy(orb_symmetries,orbsymcpy, n_symold * mi*mi * sizeof(complex128_t));
        free(orbsymcpy);

        Map<CMat3d> sym_rsm(rs_symmetries+(n_symold)*9);
        Map<CMatXcd> sym_orbm(orb_symmetries+(n_symold)*mi*mi,mi,mi);
        sym_rsm = id_rs;
        sym_orbm = id_orb;
        syms[n_symold] = n_symold;
        syms[n_symold] = n_symold;

        symm->inv_symmetry.resize(n_sym);
        memcpy(symm->inv_symmetry.data(),syms.data(),n_sym*sizeof(index_t));
    }
    return added_symm;
}

void init_ibz(diverge_model_t* model) {
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);
    symmstruct_t* symm = tu_data->symm;
    index_t n_sym = model->n_sym;
    index_t nk = kdim(model->nk);

    index_t* mom_maps_to = model->internals->symm_map_mom_crs;
    bool* is_mapped_to_ibz = (bool*)malloc(sizeof(bool)*nk);
    for(uint i = 0; i < nk; ++i)
        is_mapped_to_ibz[i] = false;

    complex128_t* orb_symmetries = (complex128_t*)model->orb_symmetries;
    double* rs_symmetries = (double*)model->rs_symmetries;
    index_t n_orb = model->n_orb;
    index_t n_spin = model->n_spin;
    index_t mi = n_orb * n_spin;
    CMat3d id_rs = CMat3d::Identity();
    CMatXcd id_orb = CMatXcd::Identity(mi,mi);
    index_t which_is_identity = -1;
    for(index_t sym = 0; sym < n_sym; ++sym) {
        Map<CMat3d> sym_rs(rs_symmetries+sym*9);
        Map<CMatXcd> sym_orb((orb_symmetries)+sym*mi*mi,mi,mi);
        if(approxEQ(sym_rs,id_rs) && approxEQ(sym_orb,id_orb)) {
            which_is_identity = sym;
            sym = n_sym;
        }
    }
    if(which_is_identity < 0) {
        mpi_err_printf("identity not contained in symmetry operations - this will fail!\n");
    }

    index_t current_ibz_idx = 0;
    symm->kmaps_to.resize(nk);
    symm->which_symmetry_used.resize(nk);

    for(index_t ii=0; ii<nk; ++ii) {
        if (!is_mapped_to_ibz[ii]) {
            symm->idx_ibz_in_fullmesh.push_back(ii);
            is_mapped_to_ibz[ii] = true;
            symm->which_symmetry_used[ii] = {which_is_identity, symm->inv_symmetry[which_is_identity]};
            symm->kmaps_to[ii] = current_ibz_idx;
            for (index_t s=0; s<n_sym; ++s) {
                index_t idx = mom_maps_to[s+n_sym*ii];
                if (!is_mapped_to_ibz[idx]) {
                    is_mapped_to_ibz[idx] = true;
                    symm->which_symmetry_used[idx] = {s, symm->inv_symmetry[s]};
                    symm->kmaps_to[idx] = current_ibz_idx;
                }
            }
            ++current_ibz_idx;
        }
    }
    //! Test that all k-points have an IBZ map
    for (index_t i=0; i< nk; ++i) {
        if (!is_mapped_to_ibz[i])
            mpi_wrn_printf("No IBZ-BZ-map at idx: %li\n", i);
    }
    free(is_mapped_to_ibz);
    tu_data->nkibz = (index_t)(symm->idx_ibz_in_fullmesh.size());
    mpi_vrb_printf("#k in ibz: %li\n", tu_data->nkibz);

    //! only necessary for MPI
    //! gives for each IBZ point which points it corresponds to beyond the
    //! IBZ, again in a linearized fashion
    vector<vector<index_t>> dummy;
    dummy.resize(tu_data->nkibz);
    for (index_t ii=0; ii<nk; ++ii) {
        dummy[symm->kmaps_to[ii]].push_back(ii);
    }
    symm->from_ibz_2_bz_off.reserve(tu_data->nkibz+1);
    symm->from_ibz_2_bz_len.reserve(tu_data->nkibz);
    symm->from_ibz_2_bz.reserve(tu_data->nk);
    symm->from_ibz_2_bz_off.push_back(0);
    for (index_t ii=0; ii<tu_data->nkibz; ++ii) {
        symm->from_ibz_2_bz_len.push_back(dummy[ii].size());
        symm->from_ibz_2_bz_off.push_back(symm->from_ibz_2_bz_off.back()
                                            + dummy[ii].size());
        for(index_t jj=0; jj<(index_t)dummy[ii].size(); ++jj)
            symm->from_ibz_2_bz.push_back(dummy[ii][jj]);
    }
    symm->from_ibz_2_bz_off.pop_back();
    symm->from_ibz_2_bz_off.shrink_to_fit();
    init_MPI_dist(model);
    symm->my_nk_in_bz = 0;
    symm->my_nk_in_bz_off = symm->from_ibz_2_bz_off[tu_data->my_nk_off];
    for(index_t ibz = 0; ibz < tu_data->my_nk;++ibz)
        symm->my_nk_in_bz += symm->from_ibz_2_bz_len[ibz+tu_data->my_nk_off];
}


void init_map_for_one_leg(diverge_model_t* model, vector<Vec3d>& beyond_UC) {
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);
    symmstruct_t* symm = tu_data->symm;
    index_t* bond_sizes = tu_data->bond_sizes;
    index_t* bond_offsets = tu_data->bond_offsets;
    tu_formfactor_t* tu_ff = model->tu_ff;

    double* rs_symmetries = (double*)model->rs_symmetries;
    index_t n_orb = model->n_orb;
    index_t n_spin = model->n_spin;
    index_t n_sym = model->n_sym;
    index_t n_orbff = tu_data->n_orbff;

    symm->mi_map_len.resize(tu_data->n_orbff*n_spin*n_spin*n_sym);
    symm->mi_map_off.resize(tu_data->n_orbff*n_spin*n_spin*n_sym);
    symm->mi_prefac_in_out.reserve(tu_data->n_orbff*n_spin*n_spin*n_sym*2);
    symm->mi_prefac_in_in.reserve(tu_data->n_orbff*n_spin*n_spin*n_sym*2);
    symm->mi_maps_to.reserve(tu_data->n_orbff*n_spin*n_spin*n_sym*2);
    index_t* orbmaps_to_len = model->internals->symm_orb_len;
    index_t* orbmaps_to_off = model->internals->symm_orb_off;
    index_t* orbmaps_to = model->internals->symm_map_orb;
    complex128_t* orbmaps_to_prefac = model->internals->symm_pref;
    double* beyond_UC_in = model->internals->symm_beyond_UC;
    int periodic_iter[3];
    periodic_iter[0] = model->internals->periodic_direction[0] ? 3:0;
    periodic_iter[1] = model->internals->periodic_direction[1] ? 3:0;
    periodic_iter[2] = model->internals->periodic_direction[2] ? 3:0;

    Map<Vec3d> UC1(model->lattice[0]);
    Map<Vec3d> UC2(model->lattice[1]);
    Map<Vec3d> UC3(model->lattice[2]);

    for(index_t s = 0; s < n_sym; ++s)
    for(index_t s2 = 0; s2 < n_spin; ++s2)
    for(index_t s1 = 0; s1 < n_spin; ++s1)
    for(index_t mo1 = 0; mo1 < n_orbff; ++mo1) {
        index_t o1 = tu_data->mi_to_ofrom[mo1];
        index_t o2 = tu_data->mi_to_oto[mo1];
        index_t so1 = IDX2(s1,o1,n_orb);
        index_t so2 = IDX2(s2,o2,n_orb);
        Map<CMat3d> sym_rs(rs_symmetries+s*9);
        Map<Vec3d> po1(model->positions[o1]);
        Map<Vec3d> po2(model->positions[o2]);

        Vec3d _from = tu_data->mi_to_R[3*mo1] * UC1 + tu_data->mi_to_R[3*mo1 + 1] * UC2
                     +tu_data->mi_to_R[3*mo1 + 2] * UC3 + po1 - po2;
        Vec3d from = sym_rs * _from;

        index_t found = 0;
        for(index_t id1 = 0; id1 < orbmaps_to_len[so1*n_sym+s]; ++id1)
        for(index_t id2 = 0; id2 < orbmaps_to_len[so2*n_sym+s]; ++id2) {
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
            for (index_t mb1 = 0; mb1 < bond_sizes[symap_o1]; ++mb1)
            if (tu_ff[bond_offsets[symap_o1]+mb1].oto==symap_o2) {
                Map<Vec3d> mpo1(model->positions[symap_o1]);
                Map<Vec3d> mpo2(model->positions[symap_o2]);
                index_t mi_map = tu_data->ob_to_orbff[mb1 + tu_data->n_bonds * symap_o1];
                Vec3d ffc = tu_data->mi_to_R[3*mi_map] * UC1
                        + tu_data->mi_to_R[3*mi_map + 1] * UC2
                        + tu_data->mi_to_R[3*mi_map + 2] * UC3 + mpo1 - mpo2;
                bool mapped_to = false;
                for(int in = -periodic_iter[0];in<=periodic_iter[0];in++)
                for(int im = -periodic_iter[1];im<=periodic_iter[1];im++)
                for(int iq = -periodic_iter[2];iq<=periodic_iter[2];iq++) {
                    Vec3d trial = ffc - from + UC1*in*model->nk[0]
                                      + UC2*im*model->nk[1]
                                      + UC3*iq*model->nk[2];
                    if(trial.norm()<1e-5)
                        mapped_to = true;
                }

                if (mapped_to) {
                    found +=1;
                    symm->mi_maps_to.push_back(mi_map
                            +tu_data->n_orbff * (symap_s1+n_spin*symap_s2));
                    symm->mi_prefac_in_in
                            .push_back((maps_to_pref_o1*maps_to_pref_o2));
                    symm->mi_prefac_in_out
                            .push_back(maps_to_pref_o1*std::conj(maps_to_pref_o2));
                    beyond_UC.push_back(Map<Vec3d>(beyond_UC_in+3*(s+n_sym*o2)));
                }
            }
        }
        index_t mi_idx = mo1+tu_data->n_orbff*(s1 + n_spin*(s2+n_spin*s));
        symm->mi_map_len[mi_idx] = found;
        if(found == 0) {
            mpi_wrn_printf("bond not symmapped: midx%d, s%d\n", mo1, s);
        }
    }

    symm->mi_map_off[0] = 0;
    for(index_t i = 1; i < (index_t)symm->mi_map_len.size(); ++i) {
        symm->mi_map_off[i] = symm->mi_map_len[i-1] + symm->mi_map_off[i-1];
    }
    symm->mi_prefac_in_in.shrink_to_fit();
    symm->mi_prefac_in_out.shrink_to_fit();
    symm->mi_maps_to.shrink_to_fit();
}

void init_symmetry_maps_o2m(diverge_model_t* model, vector<Vec3d>& beyond_UC) {
    mpi_vrb_printf("Generate one-to-many maps\n");
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);
    symmstruct_t* symm = tu_data->symm;
    index_t n_spin = model->n_spin;
    index_t my_nk_in_bz = symm->my_nk_in_bz;
    index_t my_nk_in_bz_off = symm->my_nk_in_bz_off;
    index_t n_orbff = tu_data->n_orbff;
    double* rs_symmetries = (double*)model->rs_symmetries;

    symm->prefactor_symmP.reserve(my_nk_in_bz*(n_orbff)*POW2(n_spin));
    symm->idx_map_symm.reserve(my_nk_in_bz*(n_orbff)*POW2(n_spin));
    symm->prefactor_symmCD.reserve(my_nk_in_bz*(n_orbff)*POW2(n_spin));

    symm->o2m_map_off.resize(POW2(n_spin)*n_orbff*my_nk_in_bz);
    symm->o2m_map_len.resize(POW2(n_spin)*n_orbff*my_nk_in_bz);
    for(index_t s2 = 0; s2<n_spin;++s2)
    for(index_t s1 = 0; s1<n_spin;++s1)
    for(index_t mi1 = 0; mi1<n_orbff;++mi1)
    for(index_t q = 0; q < my_nk_in_bz; ++q) {
        const index_t q_in_full = symm->from_ibz_2_bz[q + my_nk_in_bz_off];
        const index_t which_sym = symm->which_symmetry_used[q_in_full][1];
        assert(symm->kmaps_to[q_in_full] - tu_data->my_nk_off < tu_data->my_nk);
        Map<CMat3d> sym_rs(rs_symmetries+symm->which_symmetry_used[q_in_full][0]*9);

        index_t idx = mi1 + n_orbff*(s1 + n_spin*(s2+n_spin*which_sym));
        Map<Vec3d> kp(model->internals->kmesh+3*q_in_full);
        const complex128_t pref_sym =
                exp(-I128*kp.dot(sym_rs*beyond_UC[idx]));

        for(index_t to = 0; to < symm->mi_map_len[idx]; ++to) {
            symm->idx_map_symm.push_back(symm->mi_maps_to[to + symm->mi_map_off[idx]]);

            complex128_t pref_elem = symm->mi_prefac_in_in[to + symm->mi_map_off[idx]];
            symm->prefactor_symmP.push_back(pref_elem*pref_sym);

            pref_elem = symm->mi_prefac_in_out[to + symm->mi_map_off[idx]];
            symm->prefactor_symmCD.push_back(pref_elem*pref_sym);
        }
        symm->o2m_map_len[IDX4(s2,s1,mi1,q,n_spin,n_orbff,my_nk_in_bz)]
            = symm->mi_map_len[idx];
    }
    symm->o2m_map_off[0] = 0;
    for(index_t i = 1; i < POW2(n_spin)*n_orbff*my_nk_in_bz; ++i)
        symm->o2m_map_off[i] = symm->o2m_map_off[i-1]+symm->o2m_map_len[i-1];
    symm->prefactor_symmCD.shrink_to_fit();
    symm->prefactor_symmP.shrink_to_fit();
    symm->idx_map_symm.shrink_to_fit();
}


void init_symmetries(diverge_model_t* model) {
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);
    symmstruct_t* symm = tu_data->symm;
    if(model->n_sym == 0) {
        symm->use_symmetries = false;
        tu_data->nkibz = tu_data->nk;
        init_MPI_dist(model);
        symm->my_nk_in_bz = tu_data->my_nk;
        symm->my_nk_in_bz_off = tu_data->my_nk_off;
        symm->from_ibz_2_bz.resize(tu_data->nk);
        symm->idx_ibz_in_fullmesh.resize(tu_data->nk);
        for(index_t i = 0; i < tu_data->nk; ++i )symm->from_ibz_2_bz[i] = i;
        for(index_t i = 0; i < tu_data->nk; ++i )symm->idx_ibz_in_fullmesh[i] = i;
        return;
    }
    symm->use_symmetries = true;
    bool added_symm = init_inv_symmerty_map(model);
    if(added_symm) {
        free(model->internals->symm_map_mom_crs); 
        free(model->internals->symm_map_mom_fine);
        free(model->internals->symm_beyond_UC);
        free(model->internals->symm_orb_off);
        free(model->internals->symm_orb_len);
        free(model->internals->symm_map_orb);
        free(model->internals->symm_pref);

        model->internals->symm_map_mom_crs = nullptr;
        model->internals->symm_map_mom_fine = nullptr;
        model->internals->symm_beyond_UC = nullptr;
        model->internals->symm_orb_off = nullptr;
        model->internals->symm_orb_len = nullptr;
        model->internals->symm_map_orb = nullptr;
        model->internals->symm_pref = nullptr;
    }
    diverge_generate_symm_maps(model);
    init_ibz(model);
    vector<Vec3d> hi;
    init_map_for_one_leg(model, hi);
    init_symmetry_maps_o2m(model, hi);
}

