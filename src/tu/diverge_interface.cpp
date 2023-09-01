#include "diverge_interface.hpp"
#include <fstream>
#include <cassert>
#include <set>
#include <map>

#include "../diverge_model_internals.h"
#include "../misc/cmath_helpers.h"

typedef Eigen::Matrix<double,3,3,Eigen::RowMajor> CMat3d;

void diverge_model_internals_tu(diverge_model_t* model, double max_dist) {
    index_t n_frq = 1;
    tu_data_t* info = new tu_data_t;
    info->nk = model->nk[0] * model->nk[1] * model->nk[2];
    info->nkf = model->nkf[0] * model->nkf[1] * model->nkf[2];
    info->nktot = model->nk[0] * model->nk[1] * model->nk[2]*
                 model->nkf[0] * model->nkf[1] * model->nkf[2];
    info->symm = new symmstruct_t;
    info->use_reduced_loop = true;
    model->internals->tu_data = (void*) info;
    model->internals->tu_data_destructor = tu_data_destructor;
    info->n_frq = n_frq;
    init_bos_freqs(info,n_frq);
    init_ff_struct(max_dist, model);
    generate_reduced_loop_maps(model);
    init_symmetries(model);
    init_doping_per_UC(model);
    info->offload_cpu = 0.0;
}

void tu_data_destructor(void* tu_da) {
    tu_data_t* tu_data = (tu_data_t*) tu_da;
    delete tu_data->symm;
    free(tu_data->bond_sizes);
    free(tu_data->bond_offsets);
    free(tu_data->ff_idx_to_unique_ff_elem);
    free(tu_data->ob_to_orbff);
    free(tu_data->mi_to_ofrom);
    free(tu_data->mi_to_ffidx);
    free(tu_data->mi_to_oto);
    free(tu_data->mi_to_R);
    delete tu_data;
}

void init_bos_freqs(tu_data_t* tu_data, index_t n_frq){
    const double first_freq = 1e-2;
    const double base_increment = 1.2;
    tu_data->bosonic_freq.resize(n_frq);
    tu_data->bosonic_freq[0] = 0.;
    for(int n = 1; n<n_frq; n++){
        tu_data->bosonic_freq[n] = first_freq*(cmath_pow(base_increment,n)-1.0)
                                            /(base_increment-1.0);
    }
}

static index_t map_to_fine(index_t val, index_t nc) {
    if (std::abs(nc - val) < (nc / 2) + 1) val = val - nc;
    return val;
}

inline void fill_existing_ff(diverge_model_t* model) {
    tu_data_t* tu_data = (tu_data_t*)model->internals->tu_data;
    tu_formfactor_t* tu_ff = model->tu_ff;
    for(index_t i = 0; i < model->n_tu_ff; ++i) {
        Vec3i ff = Map<Eigen::Matrix<index_t,3,1>>(tu_ff[i].R).cast<int>();
        index_t idx = std::distance(tu_data->existing_ff.begin(),
                        std::find_if(tu_data->existing_ff.begin(),
                        tu_data->existing_ff.end(), vec3i_cmp(ff)));
        if(idx == (index_t)tu_data->existing_ff.size()) {
            tu_data->existing_ff.push_back(ff);
            tu_ff[i].ffidx = idx;
        }else{
            tu_ff[i].ffidx = idx;
        }
    }
    tu_data->existing_ff.shrink_to_fit();

    tu_data->existing_ff_fine.resize(tu_data->existing_ff.size());
    for(uindex_t i = 0; i < tu_data->existing_ff.size(); ++i) {
        tu_data->existing_ff_fine[i](0) = map_to_fine(tu_data->existing_ff[i](0), model->nk[0]);
        tu_data->existing_ff_fine[i](1) = map_to_fine(tu_data->existing_ff[i](1), model->nk[1]);
        tu_data->existing_ff_fine[i](2) = map_to_fine(tu_data->existing_ff[i](2), model->nk[2]);
    }
}

inline void sort_ff_struct(tu_formfactor_t* tu_ff, index_t n_tu_ff) {
    vector<index_t> idists( n_tu_ff );
    for (index_t d=0; d<n_tu_ff; ++d)
        idists[d] = d;

    std::sort(idists.begin(), idists.end(), [tu_ff](index_t a, index_t b) {
                    if (tu_ff[a].ofrom != tu_ff[b].ofrom)
                        return tu_ff[a].ofrom < tu_ff[b].ofrom;
                    if (std::abs(tu_ff[a].d-tu_ff[b].d) > 1e-7)
                        return tu_ff[a].d < tu_ff[b].d;
                    if (tu_ff[a].R[0] != tu_ff[b].R[0])
                        return tu_ff[a].R[0] < tu_ff[b].R[0];
                    if (tu_ff[a].R[1] != tu_ff[b].R[1])
                        return tu_ff[a].R[1] < tu_ff[b].R[1];
                    return abs(tu_ff[a].ofrom - tu_ff[a].oto) < abs(tu_ff[b].ofrom - tu_ff[b].oto);
                });
    tu_formfactor_t* ff_copy = (tu_formfactor_t*)calloc(n_tu_ff,sizeof(tu_formfactor_t));
    memcpy(ff_copy, tu_ff, n_tu_ff * sizeof(tu_formfactor_t));
    for(index_t i = 0; i < n_tu_ff; ++i)
        tu_ff[i] = ff_copy[idists[i]];
    free(ff_copy);
}


void init_bond_descriptor(diverge_model_t* model) {
    index_t n_tu_ff = model->n_tu_ff;
    tu_formfactor_t* tu_ff = model->tu_ff;
    tu_data_t* tu_data = (tu_data_t*)model->internals->tu_data;
    tu_data->bond_sizes = (index_t*)calloc(model->n_orb, sizeof(index_t));
    tu_data->bond_offsets = (index_t*)calloc(model->n_orb, sizeof(index_t));
    for(index_t i = 0; i < n_tu_ff; ++i)
        tu_data->bond_sizes[tu_ff[i].ofrom] += 1;

    for(index_t i = 1; i < model->n_orb; ++i)
        tu_data->bond_offsets[i] = tu_data->bond_offsets[i-1]+tu_data->bond_sizes[i-1];

    tu_data->n_bonds = *std::max_element(tu_data->bond_sizes,
                                tu_data->bond_sizes + model->n_orb);
    tu_data->ob_to_orbff = (index_t*)calloc(model->n_orb * tu_data->n_bonds, sizeof(index_t));
    tu_data->n_orbff = n_tu_ff;
    tu_data->mi_to_ofrom = (index_t*)calloc(n_tu_ff,sizeof(index_t));
    tu_data->mi_to_oto = (index_t*)calloc(n_tu_ff,sizeof(index_t));
    tu_data->mi_to_R = (index_t*)calloc(3*n_tu_ff,sizeof(index_t));
    tu_data->mi_to_ffidx = (index_t*)calloc(n_tu_ff,sizeof(index_t));
    index_t idx = 0;
    for(index_t b = 0; b < tu_data->n_bonds; ++b)
    for(index_t o = 0; o < model->n_orb; ++o){
        if(b < tu_data->bond_sizes[o]) {
            tu_data->ob_to_orbff[b + tu_data->n_bonds * o] = idx;
            tu_data->mi_to_ofrom[idx] = tu_ff[b+tu_data->bond_offsets[o]].ofrom;
            tu_data->mi_to_oto[idx]   = tu_ff[b+tu_data->bond_offsets[o]].oto;
            tu_data->mi_to_ffidx[idx] = tu_ff[b+tu_data->bond_offsets[o]].ffidx;
            tu_data->mi_to_R[3*idx]   = map_to_fine(tu_ff[b+tu_data->bond_offsets[o]].R[0],
                                                        model->nk[0]);
            tu_data->mi_to_R[3*idx+1] = map_to_fine(tu_ff[b+tu_data->bond_offsets[o]].R[1],
                                                        model->nk[1]);
            tu_data->mi_to_R[3*idx+2] = map_to_fine(tu_ff[b+tu_data->bond_offsets[o]].R[2],
                                                        model->nk[2]);
            idx +=1;
        }else{
            tu_data->ob_to_orbff[b + tu_data->n_bonds * o] = -1;
        }
    }

    for(index_t o = 0; o < model->n_orb; ++o)
        assert(tu_data->mi_to_ofrom[o] == tu_data->mi_to_oto[o]);
    assert(n_tu_ff == idx);
}


void init_ff_struct(const double max_dist, diverge_model_t* model) {
    if(model->n_tu_ff != 0) {
        mpi_log_printf("do not autogenerate bonds (user orbff)\n");
        sort_ff_struct(model->tu_ff, model->n_tu_ff);
        fill_existing_ff(model);
        init_bond_descriptor(model);
        return ;
    }
    mpi_log_printf("autogenerate bonds (orbff)\n");
    int periodic_iter[3];
    periodic_iter[0] = model->internals->periodic_direction[0] ? 3:0;
    periodic_iter[1] = model->internals->periodic_direction[1] ? 3:0;
    periodic_iter[2] = model->internals->periodic_direction[2] ? 3:0;
    tu_data_t* tu_data = (tu_data_t*)model->internals->tu_data;
    index_t n_orb = model->n_orb;
    vector<index_t> orbcomb;
    vector<Vec3i> real_ff;
    vector<double> dist_from;

    Vec3i delta;
    Map<CMat3d> UCV(model->lattice[0]);

    for(index_t sfrom = 0; sfrom< n_orb; ++sfrom) {
        Map<Vec3d> _from(model->positions[sfrom]);
        for(index_t sto = 0; sto < n_orb; ++sto) {
            vector<Vec4d> bond_lookup;
            for(index_t uc_1 = 0; uc_1 < model->nk[0]; ++uc_1)
            for(index_t uc_2 = 0; uc_2 < model->nk[1]; ++uc_2)
            for(index_t uc_3 = 0; uc_3 < model->nk[2]; ++uc_3)  {
                Vec3d to = Map<Vec3d>(model->positions[sto]);

                Vec3d from = _from + UCV.transpose() * Vec3d(uc_1,uc_2,uc_3);
                Vec3d dist = - from + to;
                delta << uc_1, uc_2, uc_3;
                for(int in = -periodic_iter[0];in<=periodic_iter[0];in++)
                for(int im = -periodic_iter[1];im<=periodic_iter[1];im++)
                for(int iq = -periodic_iter[2];iq<=periodic_iter[2];iq++) {
                    Vec3d trial = - from + to + UCV.transpose() *
                        Vec3d(in*model->nk[0], im*model->nk[1], iq*model->nk[2]);
                    if(trial.norm()<dist.norm()+1e-12) {
                        dist = trial;
                    }
                }
                Vec4d helper;
                helper.head(3) = dist;
                helper(3) = sto;

                if (dist.norm() < max_dist+1e-10 &&
                    std::find_if(bond_lookup.begin(), bond_lookup.end(), vec_cmp(helper)) 
                                    == bond_lookup.end()) {
                    #pragma omp critical(write_into)
                    {
                        bond_lookup.push_back(helper);
                        real_ff.push_back(delta);
                        orbcomb.push_back(sfrom);
                        orbcomb.push_back(sto);
                        dist_from.push_back(dist.norm());
                    }
                }
            }
        }
    }

    model->n_tu_ff = real_ff.size();
    model->tu_ff = (tu_formfactor_t*)calloc(real_ff.size(),sizeof(tu_formfactor_t));
    tu_formfactor_t* tu_ff = model->tu_ff;
    for(index_t i = 0; i < model->n_tu_ff; ++i) {
        tu_ff[i].ofrom = orbcomb[2*i];
        tu_ff[i].oto = orbcomb[2*i+1];
        assert(orbcomb[2*i] < n_orb);
        assert(orbcomb[2*i+1] < n_orb);
        tu_ff[i].R[0] = real_ff[i](0);
        tu_ff[i].R[1] = real_ff[i](1);
        tu_ff[i].R[2] = real_ff[i](2);
        tu_ff[i].d = dist_from[i];
    }
    bool refinement_allowed = true;
    for(index_t i = 0; i < model->n_tu_ff; ++i) {
        Vec3d from = UCV.transpose() *
                        Vec3d(map_to_fine(real_ff[i](0),model->nk[0]),
                              map_to_fine(real_ff[i](1),model->nk[1]),
                              map_to_fine(real_ff[i](2),model->nk[2]));
        for(int in = -periodic_iter[0];in<=periodic_iter[0];in++)
        for(int im = -periodic_iter[1];im<=periodic_iter[1];im++)
        for(int iq = -periodic_iter[2];iq<=periodic_iter[2];iq++) {
            Vec3d add = UCV.transpose() *
                        Vec3d(in*model->nk[0], im*model->nk[1], iq*model->nk[2]);
            Vec3d trial = from + add;
            if(std::abs(from.norm() - trial.norm()) < 1e-8 && add.norm() > 1e-8)
                refinement_allowed = false;
        }
    }
    if(!refinement_allowed)
        mpi_wrn_printf("Too many ff - refinement breaks symmetry!\n")

    sort_ff_struct(model->tu_ff, model->n_tu_ff);
    fill_existing_ff(model);
    init_bond_descriptor(model);
    mpi_log_printf("generate %d ff-bonds with <= max. %d bonds per site\n", 
                    real_ff.size(),tu_data->n_bonds)
}


void init_doping_per_UC(diverge_model_t* model) {
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);
    tu_data->doping_per_UC = 0.0;
    double helper = 0.;
    #pragma omp parallel for collapse(2) reduction(+:helper)
    for(index_t nk = 0; nk<tu_data->nktot; ++nk)
    for(index_t os1 = 0; os1<model->n_orb*model->n_spin;++os1) {
        if(model->internals->E[os1+model->n_orb*model->n_spin*nk]<0.0) {
            helper+=1.0;
        }
    }
    tu_data->doping_per_UC = helper/(double)(model->n_orb*model->n_spin*tu_data->nktot);
}


static index_t generate_unique_obff(diverge_model_t* model) {
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);
    index_t n_orb = model->n_orb;
    index_t* bond_sizes = tu_data->bond_sizes;
    index_t* bond_offsets = tu_data->bond_offsets;
    tu_formfactor_t* tu_ff = model->tu_ff;

    index_t num_unique_pairs = 0;
    for(index_t o = 0; o<n_orb;++o) {
        vector<index_t> have;
        for(index_t b = 0; b<bond_sizes[o];++b) {
            index_t site = tu_ff[bond_offsets[o]+b].oto;
            index_t idx = std::distance(have.begin(),
                             std::find(have.begin(),have.end(),site));
            if(idx == (index_t)have.size()) {
                num_unique_pairs += 1;
                have.push_back(site);
            }
        }
    }
    return num_unique_pairs;
}

static index_t wrap_on_fine(index_t val, index_t ncf) {
    if (std::abs(val - ncf) < (ncf / 2) + 1) val = val - ncf;
    return val;
}

static void fill_unique_ff(diverge_model_t* model){
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);
    // create array storing the info of all unique formfactor combinations in R
    index_t nff = tu_data->existing_ff.size();
    tu_data->ff_idx_to_unique_ff_elem = (index_t*)calloc(POW2(nff),sizeof(index_t));

    for(index_t ff1 = 0; ff1 < nff; ++ff1)
    for(index_t ff3 = 0; ff3 < nff; ++ff3){
        Vec3i trial = tu_data->existing_ff_fine[ff1]-tu_data->existing_ff_fine[ff3];
        trial(0) = (trial(0)+4*model->nk[0]*model->nkf[0])%(model->nk[0]*model->nkf[0]);
        trial(1) = (trial(1)+4*model->nk[1]*model->nkf[1])%(model->nk[1]*model->nkf[1]);
        trial(2) = (trial(2)+4*model->nk[2]*model->nkf[2])%(model->nk[2]*model->nkf[2]);
        trial(0) = wrap_on_fine(trial(0),model->nk[0]*model->nkf[0]);
        trial(1) = wrap_on_fine(trial(1),model->nk[1]*model->nkf[1]);
        trial(2) = wrap_on_fine(trial(2),model->nk[2]*model->nkf[2]);
        index_t idx = std::distance(tu_data->unique_ff_combi.begin(),
                    std::find_if(tu_data->unique_ff_combi.begin()
                        ,tu_data->unique_ff_combi.end(), vec3i_cmp(trial)));
        if(idx == (index_t)tu_data->unique_ff_combi.size())
            tu_data->unique_ff_combi.push_back(trial);
        tu_data->ff_idx_to_unique_ff_elem[ff1+nff*ff3] = idx;
    }
}

void generate_reduced_loop_maps(diverge_model_t* model) {
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);

    if(!tu_data->use_reduced_loop || tu_data->nktot==1) {
        mpi_log_printf("deactivate reduced TU loop algorithm\n");
        tu_data->use_reduced_loop = false;
        tu_data->ff_idx_to_unique_ff_elem = (index_t*)calloc(1,sizeof(index_t));
        return;
    }
    index_t n_orb = model->n_orb;
    index_t* mi_to_ofrom = tu_data->mi_to_ofrom;
    index_t* mi_to_oto = tu_data->mi_to_oto;
    index_t* mi_to_ffidx = tu_data->mi_to_ffidx;
    fill_unique_ff(model);
    index_t num_unique_pairs = generate_unique_obff(model);

    if(num_unique_pairs >  (index_t)(0.7*tu_data->n_orbff)) {
        tu_data->use_reduced_loop = false;
        mpi_log_printf("speedup impossible, no orbff maps\n");
        return ;
    }
    mpi_log_printf("%d unique orbff pairs, gen maps…\n", num_unique_pairs);
    vector<vector<array<index_t,2>>> _lmap;
    index_t nff = tu_data->existing_ff.size();
    // iterate over all unique pair combinations
    std::set<index_t> stored_combinations;
    std::map<index_t,index_t> index_to_reduced;
    index_t curr_idx = 0;
    for(index_t m1 = 0; m1<tu_data->n_orbff;++m1)
    for(index_t m3 = 0; m3<tu_data->n_orbff;++m3) {
        index_t fidx1 = mi_to_ffidx[m1];
        index_t fidx3 = mi_to_ffidx[m3];
        index_t trial = IDX5(tu_data->ff_idx_to_unique_ff_elem[fidx1+nff*fidx3],
                          mi_to_oto[m3],mi_to_ofrom[m3],mi_to_oto[m1],mi_to_ofrom[m1],
                          n_orb,n_orb,n_orb,n_orb);
        if(stored_combinations.find(trial) == stored_combinations.end()) {
            stored_combinations.insert(trial);
            index_to_reduced.insert({trial,curr_idx});
            curr_idx += 1;
            array<index_t, 2> l = {m1,m3};
            vector<array<index_t, 2>> b {l};
            _lmap.push_back(b);
        }else{
            index_t idx = index_to_reduced.at(trial);
            array<index_t, 2> l = {m1,m3};
            _lmap[idx].push_back(l);
        }
    }
    tu_data->unique_mi_l.resize(_lmap.size());
    tu_data->unique_mi_r.resize(_lmap.size());
    tu_data->lmap_off.resize(_lmap.size());
    tu_data->lmap_len.resize(_lmap.size());
    tu_data->lmap_off[0] = 0;
    tu_data->lmap_l.reserve(POW2(tu_data->n_orbff));
    tu_data->lmap_r.reserve(POW2(tu_data->n_orbff));
    for(index_t i = 0; i < (int)_lmap.size();++i) {
        tu_data->unique_mi_l[i] = _lmap[i][0][0];
        tu_data->unique_mi_r[i] = _lmap[i][0][1];
        for(index_t j = 0; j < (int)_lmap[i].size();++j){
            tu_data->lmap_l.push_back(_lmap[i][j][0]);
            tu_data->lmap_r.push_back(_lmap[i][j][1]);
        }
        tu_data->lmap_len[i] = _lmap[i].size();
        if(i>0)
            tu_data->lmap_off[i] = tu_data->lmap_off[i-1] + tu_data->lmap_len[i-1];
    }
    tu_data->lmap_l.shrink_to_fit();
    tu_data->lmap_r.shrink_to_fit();
    mpi_log_printf("#unique orbff=%lu (with %lu unique ff from %li)\n",
                tu_data->unique_mi_l.size(), tu_data->unique_ff_combi.size(), POW2(tu_data->n_orbff));
}


void init_MPI_dist(diverge_model_t* model) {
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);
    int rank = diverge_mpi_comm_rank();
    int nranks = diverge_mpi_comm_size();
    // distribute kpoints
    index_t to_be_distributed = tu_data->nkibz;
    if (to_be_distributed < nranks)
        mpi_err_printf("nkibz (%li) < #ranks (%i). reduce #ranks!\n", to_be_distributed, nranks);

    index_t momenta_per_rank = to_be_distributed/nranks;
    index_t remainder = to_be_distributed%nranks;

    if(rank < remainder)
    {
        tu_data->my_nk = momenta_per_rank+1;
        tu_data->my_nk_off = rank*(momenta_per_rank+1);
    }else{
        tu_data->my_nk = momenta_per_rank;
        tu_data->my_nk_off = remainder * (momenta_per_rank+1)
                           + momenta_per_rank * (rank-remainder);
    }
    diverge_mpi_barrier();
    mpi_log_printf_all( "rank=%2d mynk=%4d nkibz=%4d\n", rank, tu_data->my_nk, tu_data->nkibz);
}

