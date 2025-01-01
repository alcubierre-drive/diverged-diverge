/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "projection_wrapper.hpp"
#include <set>
#include <map>
#include <numeric>

Projection::~Projection() {
    free(my_prefac_proj);
    free(my_prefac_reco);
    free(my_prefac_proj_rk);
}

static inline vector<size_t> sort_permutation( const vector<array<index_t,2>>& _t) {
    vector<size_t> p(_t.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](uindex_t i, uindex_t j){
            if (_t[i][1] == _t[j][1])
                return _t[i][0] < _t[j][0];
            return _t[i][1] < _t[j][1]; });
    return p;
}

template <typename T, typename S>
void apply_permutation_in_place( vector<T>& vec, const vector<S>& p) {
    vector<bool> done(vec.size());
    for (uindex_t i=0; i<vec.size(); ++i) {
        if (done[i])
            continue;
        done[i] = true;
        uindex_t prev_j = i;
        uindex_t j = p[i];
        while (i != j) {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

static inline void sort_and_filter(vector<array<index_t,2>>& _t, vector<index_t>& _t_p,
                              vector<array<index_t,2>>& _fhelp, vector<index_t>& _f,
                              vector<std::array<index_t,2>>& _rhelp,vector<index_t>& _r,
                              vector<index_t>& _len,
                              vector<index_t>& _off){
    vector<size_t> p = sort_permutation(_t);
    apply_permutation_in_place(_t,p);
    apply_permutation_in_place(_fhelp,p);
    apply_permutation_in_place(_rhelp,p);
    uindex_t msize = _t.size();
    _t_p.resize(2*msize);
    memcpy((void*)_t_p.data(),(void*)&(_t[0][0]), 2*msize*sizeof(index_t));

    vector<array<index_t,2>>().swap(_t);
    index_t l = _t_p[0]; index_t r = _t_p[1];
    index_t len = 0;
    _f.resize(_t_p.size());
    _r.resize(_t_p.size());
    for(uindex_t i = 0; i < msize; ++i) {
        if(_t_p[2*i] == l && _t_p[2*i+1] == r) {
            len += 1;
        }else{
            _len.push_back(len); len = 1;
            _t.push_back({l,r});
            l = _t_p[2*i]; r = _t_p[2*i+1];
        }
    }
    _len.push_back(len);
    _t.push_back({l,r});

    _off.resize(_len.size());
    _off[0] = 0;
    for(uindex_t i = 1; i < _len.size(); ++i)
        _off[i] = _off[i-1] + _len[i-1];
    for(uindex_t k = 0; k < _len.size(); ++k) {
        vector<uindex_t> p(_len[k]);
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(),
                    [&](uindex_t i, uindex_t j){ if(_fhelp[i+_off[k]][1] == _fhelp[j+_off[k]][1])
                                                    return _fhelp[i+_off[k]][0] < _fhelp[j+_off[k]][0];
                                                 return _fhelp[i+_off[k]][1] < _fhelp[j+_off[k]][1]; 
                    });
        for(index_t q = 0; q < _len[k]; ++q) {
            _f[2*(q+_off[k])] = _fhelp[_off[k]+p[q]][0];
            _f[2*(q+_off[k])+1] = _fhelp[_off[k]+p[q]][1];
            _r[2*(q+_off[k])] = _rhelp[_off[k]+p[q]][0];
            _r[2*(q+_off[k])+1] = _rhelp[_off[k]+p[q]][1];
        }
        _len[k] *= 2; _off[k] *= 2;
    }
}


static inline Vec3i subtract(const tu_formfactor_t* const tu_ff,index_t b1,index_t b2) {
    Vec3i helper;
    helper(0) = tu_ff[b1].R[0]-tu_ff[b2].R[0];
    helper(1) = tu_ff[b1].R[1]-tu_ff[b2].R[1];
    helper(2) = tu_ff[b1].R[2]-tu_ff[b2].R[2];
    return helper;
}

static inline Vec3i subtract(const tu_formfactor_t* const tu_ff,index_t b1,
                            index_t b2, index_t b3) {
    Vec3i helper;
    helper(0) = tu_ff[b1].R[0]-tu_ff[b2].R[0]-tu_ff[b3].R[0];
    helper(1) = tu_ff[b1].R[1]-tu_ff[b2].R[1]-tu_ff[b3].R[1];
    helper(2) = tu_ff[b1].R[2]-tu_ff[b2].R[2]-tu_ff[b3].R[2];
    return helper;
}

static inline Vec3i set(const tu_formfactor_t* const tu_ff,index_t b1, int pref) {
    Vec3i helper;
    helper(0) = tu_ff[b1].R[0]*pref;
    helper(1) = tu_ff[b1].R[1]*pref;
    helper(2) = tu_ff[b1].R[2]*pref;
    return helper;
}

static inline index_t to_idx(Vec3i a, const index_t* nk) {
    return a(2) + nk[2]*(a(1)+nk[1]*a(0));
}

static inline bool check_equal(Vec3i a, Vec3i b) {
    return (a(0) == b(0) && a(1) == b(1) && a(2) == b(2));
}

struct vec_cmp_i {
    vec_cmp_i(Eigen::Vector3i v) : val{v} { }
    inline bool operator()(const Eigen::Vector3i x) const {
        return check_equal(x,val);
    }
private:
    const Eigen::Vector3i val;
};


Projection::Projection(const diverge_model_t* const model)
:
    n_orb{model->n_orb},
    n_bonds{((tu_data_t*)(model->internals->tu_data))->n_bonds},
    n_orbff{((tu_data_t*)(model->internals->tu_data))->n_orbff},
    bond_sizes{((tu_data_t*)(model->internals->tu_data))->bond_sizes},
    bond_offsets{((tu_data_t*)(model->internals->tu_data))->bond_offsets},
    ob_to_orbff{((tu_data_t*)(model->internals->tu_data))->ob_to_orbff},
    tu_ff{model->tu_ff}
{
    double tick, tock;
    tick = diverge_mpi_wtime();
    init_CP_PC_proj(model);
    tock = diverge_mpi_wtime();
    mpi_vrb_printf("init_CP_PC_proj  took %7.2fs\n",tock-tick);

    tick = diverge_mpi_wtime();
    init_CD_DC_proj(model);
    tock = diverge_mpi_wtime();
    mpi_vrb_printf("init_CD_DC_proj  took %7.2fs\n",tock-tick);

    tick = diverge_mpi_wtime();
    init_P_to_D_proj(model);
    tock = diverge_mpi_wtime();
    mpi_vrb_printf("init_P_to_D_proj took %7.2fs\n",tock-tick);

    tick = diverge_mpi_wtime();
    init_D_to_P_proj(model);
    tock = diverge_mpi_wtime();
    mpi_vrb_printf("init_D_to_P_proj took %7.2fs\n",tock-tick);


    tick = diverge_mpi_wtime();
    init_to_mom_map_existing_r(model);
    tock = diverge_mpi_wtime();
    mpi_vrb_printf("constructing existing vectors took %7.2fs\n",tock-tick);

    mpi_communicate();
    init_prefac_vector(model);
    tu_data_t* tu_data = (tu_data_t*)(model -> internals-> tu_data);
    const index_t ntasks =  (m_DC_CD_f.size()/2)*_nr*POW4(model->n_spin);
    cuda_setup_projections(&proj_descr,ntasks,tu_data->my_nk);
}


void Projection::init_prefac_vector(const diverge_model_t* const model) {
    tu_data_t* tu_data = (tu_data_t*)(model -> internals-> tu_data);
    index_t nk = model->nk[0] * model->nk[1] * model->nk[2];
    complex128_t* prefac_proj = (complex128_t*) calloc(nk*_nr,sizeof(complex128_t));

    #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(diverge_omp_num_threads())
    for(index_t rv = 0; rv < _nr; ++rv)
    for(index_t kx = 0; kx < model->nk[0]; ++kx)
    for(index_t ky = 0; ky < model->nk[1]; ++ky)
    for(index_t kz = 0; kz < model->nk[2]; ++kz) {
        index_t k = IDX3(kx, ky, kz, model->nk[1], model->nk[2]);
        prefac_proj[k+nk*rv] = exp(2.*M_PI*I128*(existing_r[rv](0)*kx/(double)model->nk[0]
                                    +existing_r[rv](1)*ky/(double)model->nk[1]
                                    +existing_r[rv](2)*kz/(double)model->nk[2]));
    }
    my_prefac_proj = (complex128_t*) calloc(tu_data->my_nk*_nr,sizeof(complex128_t));
    my_prefac_proj_rk = (complex128_t*) calloc(tu_data->my_nk*_nr,sizeof(complex128_t));
    #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(diverge_omp_num_threads())
    for(index_t rv = 0; rv < _nr; ++rv)
    for(index_t k = 0; k < tu_data->my_nk; ++k) {
        my_prefac_proj[k+tu_data->my_nk*rv] =
            prefac_proj[tu_data->symm->idx_ibz_in_fullmesh[k+tu_data->my_nk_off] + rv*nk];
        my_prefac_proj_rk[rv + _nr*k] =
            prefac_proj[tu_data->symm->idx_ibz_in_fullmesh[k+tu_data->my_nk_off] + rv*nk];
    }

    const index_t my_nk_in_bz = tu_data->symm->my_nk_in_bz;
    const index_t my_nk_in_bz_off = tu_data->symm->my_nk_in_bz_off;
    my_prefac_reco = (complex128_t*) calloc(my_nk_in_bz*_nr,sizeof(complex128_t));
    #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(diverge_omp_num_threads())
    for(index_t k = 0; k < my_nk_in_bz; ++k)
    for(index_t rv = 0; rv < _nr; ++rv) {
        const index_t q_in_full = tu_data->symm->from_ibz_2_bz[k + my_nk_in_bz_off];
        my_prefac_reco[rv + _nr*k] = prefac_proj[q_in_full+nk*rv];
    }
    free(prefac_proj);
}


void Projection::init_CP_PC_proj(const diverge_model_t* const model) {
    if(diverge_mpi_comm_rank() != 0) return;
    vector<array<index_t, 2>> f_helper;
    vector<array<index_t, 2>> r_helper;
    // calculate site dependent index map
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        vector<array<index_t, 2>> my_t;
        my_t.reserve(POW2(n_orbff)/diverge_omp_num_threads());
        vector<array<index_t, 2>> my_help_f;
        my_help_f.reserve(POW2(n_orbff)/diverge_omp_num_threads());
        vector<std::array<index_t,2>> my_rv;
        my_rv.reserve(POW2(n_orbff)/diverge_omp_num_threads());
        Vec3i s1, s2;
        #pragma omp for collapse(3) schedule(dynamic)
        for(index_t o1 = 0; o1 < n_orb; ++o1)
        for(index_t o3 = 0; o3 < n_orb; ++o3) {
            for(index_t b1 = 0; b1 < n_bonds; ++b1)
            for(index_t b3p = 0; b3p < bond_sizes[o3]; ++b3p)
            if(b1 < bond_sizes[o1] && tu_ff[bond_offsets[o1]+b1].oto
                == tu_ff[bond_offsets[o3]+b3p].oto) {
                for(index_t b1p = 0; b1p < bond_sizes[o1]; ++b1p)
                for(index_t b3 = 0; b3 < bond_sizes[o3]; ++b3)
                if(tu_ff[bond_offsets[o1]+b1p].oto
                    == tu_ff[bond_offsets[o3]+b3].oto) {
                    s1 = subtract(tu_ff,bond_offsets[o3]+b3,bond_offsets[o3]+b3p);
                    s2 = subtract(tu_ff,bond_offsets[o1]+b1p,bond_offsets[o1]+b1);

                    s1(0)+=model->nk[0]*2,s1(1)+=model->nk[1]*2,s1(2)+=model->nk[2]*2;
                    s2(0)+=model->nk[0]*2,s2(1)+=model->nk[1]*2,s2(2)+=model->nk[2]*2;

                    s1(0)%=model->nk[0],s1(1)%=model->nk[1],s1(2)%=model->nk[2];
                    s2(0)%=model->nk[0],s2(1)%=model->nk[1],s2(2)%=model->nk[2];

                    if(check_equal(s1,s2)) {
                        array<index_t, 2> trial = {ob_to_orbff[b1+n_bonds*o1],
                                                    ob_to_orbff[b3+n_bonds*o3]};
                        array<index_t, 2> trial2 = {ob_to_orbff[b1p+n_bonds*o1],
                                                    ob_to_orbff[b3p+n_bonds*o3]};
                        array<index_t, 2> rv = {to_idx(s1,model->nk),to_idx(s1,model->nk)};
                        my_rv.push_back(rv);
                        my_t.push_back(trial);
                        my_help_f.push_back(trial2);
                    }
                }
            }
        }
        my_t.shrink_to_fit();
        my_rv.shrink_to_fit();
        my_help_f.shrink_to_fit();

        #pragma omp critical (to_app)
        {
            std::move(my_t.begin(), my_t.end(),
                        std::inserter(m_PC_CP_t, m_PC_CP_t.end()));

            std::move(my_help_f.begin(), my_help_f.end(),
                        std::inserter(f_helper, f_helper.end()));

            std::move(my_rv.begin(), my_rv.end(),
                        std::inserter(r_helper, r_helper.end()));
        }
    }
    sort_and_filter(m_PC_CP_t, m_PC_CP_t_padded, f_helper, m_PC_CP_f,
                             r_helper, PC_r,m_PC_CP_l, m_PC_CP_d);
}


void Projection::init_CD_DC_proj(const diverge_model_t* const model) {
    if(diverge_mpi_comm_rank() != 0) return;
    vector<array<index_t, 2>> f_helper;
    vector<array<index_t, 2>> r_helper;

    // calculate site dependent index map
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        vector<array<index_t, 2>> my_t;
        my_t.reserve(POW2(n_orbff)/diverge_omp_num_threads());
        vector<array<index_t, 2>> my_help_f;
        my_help_f.reserve(POW2(n_orbff)/diverge_omp_num_threads());
        vector<array<index_t, 2>> my_rv;
        my_rv.reserve(POW2(n_orbff)/diverge_omp_num_threads());
        Vec3i s1, s2;
        #pragma omp for collapse(3) schedule(dynamic)
        for(index_t o1 = 0; o1 < n_orb; ++o1)
        for(index_t o4 = 0; o4 < n_orb; ++o4) {
            for(index_t b1p = 0; b1p < n_bonds; ++b1p)
            if(b1p < bond_sizes[o1] && tu_ff[bond_offsets[o1]+b1p].oto == o4) {
                for(index_t b1 = 0; b1 < bond_sizes[o1]; ++b1)
                for(index_t b4 = 0; b4 < bond_sizes[o4]; ++b4) {
                    index_t o3 = tu_ff[bond_offsets[o1]+b1].oto;
                    for(index_t b3p = 0; b3p < bond_sizes[o3]; ++b3p)
                    if(tu_ff[bond_offsets[o4]+b4].oto == tu_ff[bond_offsets[o3]+b3p].oto) {
                        s1 = subtract(tu_ff,bond_offsets[o1]+b1p
                                          ,bond_offsets[o3]+b3p,bond_offsets[o1]+b1);
                        s2 = set(tu_ff,bond_offsets[o4]+b4,-1.0);

                        s1(0)+=model->nk[0]*4,s1(1)+=model->nk[1]*4,s1(2)+=model->nk[2]*4;
                        s2(0)+=model->nk[0],s2(1)+=model->nk[1],s2(2)+=model->nk[2];

                        s1(0)%=model->nk[0],s1(1)%=model->nk[1],s1(2)%=model->nk[2];
                        s2(0)%=model->nk[0],s2(1)%=model->nk[1],s2(2)%=model->nk[2];

                        if(check_equal(s1,s2)) {
                            array<index_t, 2> trial = {ob_to_orbff[b1+n_bonds*o1],
                                                        ob_to_orbff[b4+n_bonds*o4]};
                            array<index_t, 2> trial2 = {ob_to_orbff[b1p+n_bonds*o1],
                                                        ob_to_orbff[b3p+n_bonds*o3]};
                            Vec3i helper = set(tu_ff,bond_offsets[o3]+b3p,1.0);
                            array<index_t, 2> rv = {to_idx(s2,model->nk),to_idx(helper,model->nk)};
                            my_rv.push_back(rv);
                            my_t.push_back(trial);
                            my_help_f.push_back(trial2);
                        }
                    }
                }
            }
        }
        my_t.shrink_to_fit();
        my_rv.shrink_to_fit();
        my_help_f.shrink_to_fit();

        #pragma omp critical (to_app)
        {
            std::move(my_t.begin(), my_t.end(),
                        std::inserter(m_DC_CD_t, m_DC_CD_t.end()));

            std::move(my_help_f.begin(), my_help_f.end(),
                        std::inserter(f_helper, f_helper.end()));

            std::move(my_rv.begin(), my_rv.end(),
                        std::inserter(r_helper, r_helper.end()));
        }
    }
    sort_and_filter(m_DC_CD_t, m_DC_CD_t_padded, f_helper, m_DC_CD_f,
                              r_helper,DC_r,m_DC_CD_l, m_DC_CD_d);
}


void Projection::init_D_to_P_proj(const diverge_model_t* const model) {
    vector<array<index_t, 2>> f_helper;
    vector<array<index_t, 2>> r_helper;
    if(diverge_mpi_comm_rank() != 0) return;
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        vector<array<index_t, 2>> my_t;
        my_t.reserve(POW2(n_orbff)/diverge_omp_num_threads());
        vector<array<index_t, 2>> my_help_f;
        my_help_f.reserve(POW2(n_orbff)/diverge_omp_num_threads());
        vector<std::array<index_t,2>> my_rv;
        my_rv.reserve(POW2(n_orbff)/diverge_omp_num_threads());
        Vec3i s1, s2;
        #pragma omp for collapse(3) schedule(dynamic)
        for(index_t o1 = 0; o1 < n_orb; ++o1)
        for(index_t o3 = 0; o3 < n_orb; ++o3)
        for(index_t b1p = 0; b1p < n_bonds; ++b1p)
        if(b1p < bond_sizes[o1] && tu_ff[bond_offsets[o1]+b1p].oto == o3) {
            for(index_t b1 = 0; b1 < bond_sizes[o1]; ++b1)
            for(index_t b3 = 0; b3 < bond_sizes[o3]; ++b3) {
                index_t o4 = tu_ff[bond_offsets[o3]+b3].oto;

                for(index_t b4p = 0; b4p < bond_sizes[o4]; ++b4p)
                if(tu_ff[bond_offsets[o4]+b4p].oto == tu_ff[bond_offsets[o1]+b1].oto) {

                    s1 = subtract(tu_ff, bond_offsets[o1]+b1p, bond_offsets[o1]+b1);
                    s2 = set(tu_ff, bond_offsets[o4]+b4p,-1.0)
                            + set(tu_ff, bond_offsets[o3]+b3,-1.0);

                    s1(0)+=model->nk[0]*4,s1(1)+=model->nk[1]*4,s1(2)+=model->nk[2]*4;
                    s2(0)+=model->nk[0]*4,s2(1)+=model->nk[1]*4,s2(2)+=model->nk[2]*4;

                    s1(0)%=model->nk[0],s1(1)%=model->nk[1],s1(2)%=model->nk[2];
                    s2(0)%=model->nk[0],s2(1)%=model->nk[1],s2(2)%=model->nk[2];
                    if(check_equal(s1,s2)) {
                        array<index_t, 2> trial = {ob_to_orbff[b1+n_bonds*o1],
                                                    ob_to_orbff[b3+n_bonds*o3]};
                        array<index_t, 2> trial2 = {ob_to_orbff[b1p+n_bonds*o1],
                                                    ob_to_orbff[b4p+n_bonds*o4]};
                        Vec3i helper = subtract(tu_ff, bond_offsets[o1] +b1p,
                                                bond_offsets[o1]+b1);
                        helper(0)+=model->nk[0]*4,helper(1)+=model->nk[1]*4,helper(2)+=model->nk[2]*4;
                        helper(0)%=model->nk[0],helper(1)%=model->nk[1],helper(2)%=model->nk[2];
                        Vec3i helper2 = set(tu_ff, bond_offsets[o4]+b4p,-1.);

                        helper2(0)+=model->nk[0]*4,helper2(1)+=model->nk[1]*4,helper2(2)+=model->nk[2]*4;
                        helper2(0)%=model->nk[0],helper2(1)%=model->nk[1],helper2(2)%=model->nk[2];

                        array<index_t, 2> rv = {to_idx(helper,model->nk),to_idx(helper2,model->nk)};
                        my_rv.push_back(rv);
                        my_t.push_back(trial);
                        my_help_f.push_back(trial2);
                    }
                }
            }
        }
        my_t.shrink_to_fit();
        my_rv.shrink_to_fit();
        my_help_f.shrink_to_fit();

        #pragma omp critical (to_app)
        {
            std::move(my_t.begin(), my_t.end(),
                        std::inserter(m_D_to_P_t, m_D_to_P_t.end()));

            std::move(my_help_f.begin(), my_help_f.end(),
                        std::inserter(f_helper, f_helper.end()));

            std::move(my_rv.begin(), my_rv.end(),
                        std::inserter(r_helper, r_helper.end()));
        }
    }
    sort_and_filter(m_D_to_P_t, m_D_to_P_t_padded, f_helper, m_D_to_P_f,
                              r_helper,DP_r,m_D_to_P_l, m_D_to_P_d);
}


void Projection::init_P_to_D_proj(const diverge_model_t* const model) {
    if(diverge_mpi_comm_rank() != 0) return;
    vector<array<index_t, 2>> f_helper;
    vector<array<index_t, 2>> r_helper;
    #pragma omp parallel num_threads(diverge_omp_num_threads())
    {
        vector<array<index_t, 2>> my_t;
        my_t.reserve(POW2(n_orbff)/diverge_omp_num_threads());
        vector<array<index_t, 2>> my_help_f;
        my_help_f.reserve(POW2(n_orbff)/diverge_omp_num_threads());
        vector<std::array<index_t,2>> my_rv;
        my_rv.reserve(POW2(n_orbff)/diverge_omp_num_threads());
        Vec3i s1, s2;
        #pragma omp for collapse(3) schedule(dynamic)
        for(index_t o1 = 0; o1 < n_orb; ++o1)
        for(index_t o4 = 0; o4 < n_orb; ++o4) {
            for(index_t b4 = 0; b4 < n_bonds; ++b4)
            for(index_t b1p = 0; b1p < bond_sizes[o1]; ++b1p)
            if(b4 < bond_sizes[o4] && tu_ff[bond_offsets[o4]+b4].oto
                    == tu_ff[bond_offsets[o1]+b1p].oto) {
                for(index_t b1 = 0; b1 < bond_sizes[o1]; ++b1) {
                    index_t o3 = tu_ff[bond_offsets[o1]+b1].oto;
                    for(index_t b3p = 0; b3p < bond_sizes[o3]; ++b3p)
                    if(tu_ff[bond_offsets[o3]+b3p].oto == o4) {
                        s1 = set(tu_ff, bond_offsets[o4]+b4,1.);
                        s2 = subtract(tu_ff, bond_offsets[o1]+b1p, bond_offsets[o1]+b1,
                                                bond_offsets[o3]+b3p);
                        s1(0)+=model->nk[0]*4,s1(1)+=model->nk[1]*4,s1(2)+=model->nk[2]*4;
                        s2(0)+=model->nk[0]*4,s2(1)+=model->nk[1]*4,s2(2)+=model->nk[2]*4;

                        s1(0)%=model->nk[0],s1(1)%=model->nk[1],s1(2)%=model->nk[2];
                        s2(0)%=model->nk[0],s2(1)%=model->nk[1],s2(2)%=model->nk[2];

                        if(check_equal(s1, s2)) {
                            array<index_t, 2> trial = {ob_to_orbff[b1+n_bonds*o1],
                                                    ob_to_orbff[b4+n_bonds*o4]};
                            array<index_t, 2> trial2 = {ob_to_orbff[b1p+n_bonds*o1],
                                                        ob_to_orbff[b3p+n_bonds*o3]};

                            Vec3i helper = set(tu_ff,bond_offsets[o4]+b4, 1.0);

                            Vec3i helper2 = set(tu_ff, bond_offsets[o4]+b4, 1.)
                                    + set(tu_ff, bond_offsets[o3]+b3p, 1.);

                            helper2(0)+=model->nk[0]*4,helper2(1)+=model->nk[1]*4,helper2(2)+=model->nk[2]*4;
                            helper2(0)%=model->nk[0],helper2(1)%=model->nk[1],helper2(2)%=model->nk[2];
                            array<index_t, 2> rv = {to_idx(helper,model->nk),to_idx(helper2,model->nk)};
                            my_rv.push_back(rv);
                            my_t.push_back(trial);
                            my_help_f.push_back(trial2);
                        }
                    }
                }
            }
        }
        my_t.shrink_to_fit();
        my_rv.shrink_to_fit();
        my_help_f.shrink_to_fit();

        #pragma omp critical (to_app)
        {
            std::move(my_t.begin(), my_t.end(),
                        std::inserter(m_P_to_D_t, m_P_to_D_t.end()));

            std::move(my_help_f.begin(), my_help_f.end(),
                        std::inserter(f_helper, f_helper.end()));

            std::move(my_rv.begin(), my_rv.end(),
                        std::inserter(r_helper, r_helper.end()));
        }
    }
    sort_and_filter(m_P_to_D_t, m_P_to_D_t_padded, f_helper, m_P_to_D_f,
                              r_helper,PD_r,m_P_to_D_l, m_P_to_D_d);
}


static inline Vec3i unravel_idx(index_t a, const index_t* nk) {
    Vec3i help;
    index_t x = a / (nk[1]*nk[2]);
    a -= x * nk[1] * nk[2];
    index_t y = a / nk[2];
    index_t z = a - y * nk[2];
    help(0) = (int)x; help(1) = (int)y; help(2) = (int)z;
    return help;
}

void Projection::init_to_mom_map_existing_r(const diverge_model_t* model) {
    std::set<index_t> mset(PC_r.cbegin(), PC_r.cend());
    mset.insert(DC_r.cbegin(), DC_r.cend());
    mset.insert(DP_r.cbegin(), DP_r.cend());
    mset.insert(PD_r.cbegin(), PD_r.cend());
    std::map<index_t,index_t> index_to_reduced;
    index_t idx = 0;
    for(index_t i : mset) {
        index_to_reduced.insert({i, idx});
        existing_r.push_back(unravel_idx(i, model->nk));
        idx += 1;
    }
    _nr = idx;
    m_PC_CP_mom.resize(PC_r.size());
    for(uindex_t i = 0; i < PC_r.size();++i)
        m_PC_CP_mom[i] = index_to_reduced.at(PC_r[i]);

    m_DC_CD_mom.resize(DC_r.size());
    for(uindex_t i = 0; i < DC_r.size();++i)
        m_DC_CD_mom[i] = index_to_reduced.at(DC_r[i]);

    m_P_to_D_mom.resize(PD_r.size());
    for(uindex_t i = 0; i < PD_r.size();++i)
        m_P_to_D_mom[i] = index_to_reduced.at(PD_r[i]);

    m_D_to_P_mom.resize(DP_r.size());
    for(uindex_t i = 0; i < DP_r.size();++i)
        m_D_to_P_mom[i] = index_to_reduced.at(DP_r[i]);

}

void Projection::mpi_communicate() {
    index_t n_len = m_PC_CP_d.size();
    index_t n_tot = m_PC_CP_f.size();
    // PC channel
    diverge_mpi_bcast_bytes((void*)(&n_len),sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(&n_tot),sizeof(index_t),0);
    if(diverge_mpi_comm_rank() != 0) {
        m_PC_CP_d.resize(n_len);
        m_PC_CP_l.resize(n_len);
        m_PC_CP_t.resize(n_len);
        m_PC_CP_t_padded.resize(n_tot);
        m_PC_CP_mom.resize(n_tot);
        m_PC_CP_f.resize(n_tot);
    }
    diverge_mpi_bcast_bytes((void*)(m_PC_CP_d.data()),n_len*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_PC_CP_l.data()),n_len*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(&(m_PC_CP_t[0][0])),2*n_len*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_PC_CP_t_padded.data()),n_tot*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_PC_CP_mom.data()),n_tot*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_PC_CP_f.data()),n_tot*sizeof(index_t),0);
    n_len = m_DC_CD_d.size();
    n_tot = m_DC_CD_f.size();
    // DC channel
    diverge_mpi_bcast_bytes((void*)(&n_len),sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(&n_tot),sizeof(index_t),0);
    if(diverge_mpi_comm_rank() != 0) {
        m_DC_CD_d.resize(n_len);
        m_DC_CD_l.resize(n_len);
        m_DC_CD_t.resize(n_len);
        m_DC_CD_t_padded.resize(n_tot);
        m_DC_CD_mom.resize(n_tot);
        m_DC_CD_f.resize(n_tot);
    }
    diverge_mpi_bcast_bytes((void*)(m_DC_CD_d.data()),n_len*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_DC_CD_l.data()),n_len*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(&(m_DC_CD_t[0][0])),2*n_len*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_DC_CD_t_padded.data()),n_tot*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_DC_CD_mom.data()),n_tot*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_DC_CD_f.data()),n_tot*sizeof(index_t),0);

    // DP channel
    n_len = m_D_to_P_d.size();
    n_tot = m_D_to_P_f.size();
    diverge_mpi_bcast_bytes((void*)(&n_len),sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(&n_tot),sizeof(index_t),0);
    if(diverge_mpi_comm_rank() != 0) {
        m_D_to_P_d.resize(n_len);
        m_D_to_P_l.resize(n_len);
        m_D_to_P_t.resize(n_len);
        m_D_to_P_t_padded.resize(n_tot);
        m_D_to_P_mom.resize(n_tot);
        m_D_to_P_f.resize(n_tot);
    }
    diverge_mpi_bcast_bytes((void*)(m_D_to_P_d.data()),n_len*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_D_to_P_l.data()),n_len*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(&(m_D_to_P_t[0][0])),2*n_len*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_D_to_P_t_padded.data()),n_tot*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_D_to_P_mom.data()),n_tot*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_D_to_P_f.data()),n_tot*sizeof(index_t),0);

    // PD channel
    n_len = m_P_to_D_d.size();
    n_tot = m_P_to_D_f.size();
    diverge_mpi_bcast_bytes((void*)(&n_len),sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(&n_tot),sizeof(index_t),0);
    if(diverge_mpi_comm_rank() != 0) {
        m_P_to_D_d.resize(n_len);
        m_P_to_D_l.resize(n_len);
        m_P_to_D_t.resize(n_len);
        m_P_to_D_t_padded.resize(n_tot);
        m_P_to_D_mom.resize(n_tot);
        m_P_to_D_f.resize(n_tot);
    }
    diverge_mpi_bcast_bytes((void*)(m_P_to_D_d.data()),n_len*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_P_to_D_l.data()),n_len*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(&(m_P_to_D_t[0][0])),2*n_len*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_P_to_D_t_padded.data()),n_tot*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_P_to_D_mom.data()),n_tot*sizeof(index_t),0);
    diverge_mpi_bcast_bytes((void*)(m_P_to_D_f.data()),n_tot*sizeof(index_t),0);

    diverge_mpi_bcast_bytes((void*)(&_nr),sizeof(index_t),0);
     if(diverge_mpi_comm_rank() != 0) {
        existing_r.resize(_nr);
    }
    diverge_mpi_bcast_bytes((void*)(&(existing_r[0](0))),_nr*sizeof(int)*3,0);
}
