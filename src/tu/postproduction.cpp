/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "../misc/version.h"

#include "postproduction.hpp"
#include "../misc/batched_eigen.h"
#include "../misc/batched_svd.h"
#include "../misc/batched_gemms.h"
#include "../misc/eigen.h"
#include "../diverge_flow_step_internal.hpp"
#include "../diverge_common.h"

static bool check_hermicity(complex128_t* data, index_t linsize){
    double err_max = 0.;
    #pragma omp parallel for reduction(max:err_max) num_threads(diverge_omp_num_threads())
    for(index_t i=0; i<linsize; ++i)
    for(index_t j=0; j<linsize; ++j) {
        double err = std::abs(data[IDX2(i,j,linsize)] - std::conj(data[IDX2(j,i,linsize)]));
        err_max = MAX(err_max, err);
    }
    return err_max < 1e-11;
}

tu_postproduction::tu_postproduction(diverge_flow_step_t* st, bool tu_skip_channel_calc)
:
    skip_channel_calc{tu_skip_channel_calc},
    vertex{st->tu_vertex},
    proj{st->tu_proj},
    model{st->tu_vertex->model}
{
    algo = 'a';
    cutoff = 50;
    cutoff_rel = false;
    tu_n_singular_values = 1;

    index_t linsize = vertex->n_orbff*POW2(vertex->n_spin);
    if(!skip_channel_calc){
        Pval = (double*)calloc(linsize*vertex->my_nk,sizeof(double));
        Cval = (double*)calloc(linsize*vertex->my_nk,sizeof(double));
        Dval = (double*)calloc(linsize*vertex->my_nk,sizeof(double));
        Ptype = (char*)calloc(vertex->my_nk,sizeof(char));
        Ctype = (char*)calloc(vertex->my_nk,sizeof(char));
        Dtype = (char*)calloc(vertex->my_nk,sizeof(char));
        Pvec = st->tu_step->internal_vertex->Pch;
        Cvec = st->tu_step->internal_vertex->Cch;
        Dvec = st->tu_step->internal_vertex->Dch;

        lenP = (index_t*)calloc(vertex->my_nk,sizeof(index_t));
        lenC = (index_t*)calloc(vertex->my_nk,sizeof(index_t));
        lenD = (index_t*)calloc(vertex->my_nk,sizeof(index_t));
        offP = (index_t*)calloc(vertex->my_nk,sizeof(index_t));
        offC = (index_t*)calloc(vertex->my_nk,sizeof(index_t));
        offD = (index_t*)calloc(vertex->my_nk,sizeof(index_t));
    }
}


tu_postproduction::~tu_postproduction() {
    if(!skip_channel_calc){
        free(Pval);
        free(Cval);
        free(Dval);
        free(Ptype);
        free(Ctype);
        free(Dtype);
        free(lenP);
        free(lenC);
        free(lenD);
        free(offP);
        free(offC);
        free(offD);
    }
    if(called_lingap && diverge_mpi_comm_rank() == 0){
        free(Sgap_sc);
        free(Sgap_mag);
        free(Sgap_charge);
        free(Ugap_sc);
        free(Ugap_mag);
        free(Ugap_charge);
        free(Vgap_sc);
        free(Vgap_mag);
        free(Vgap_charge);
    }
    if(called_suscep_full){
        free(suscCh);
        free(suscMa);
        free(suscPp);
    }
    if(called_suscep_ff) {
        free(suscCh_ff);
        free(suscPp_ff);
        free(suscMa_ff);
    }
}

void tu_postproduction::run_and_save(string filename, bool lingaps, bool susc_ff,
                                            bool susc_full, bool save_self, bool save_full,
                                            bool save_symmetries) {
    if (!skip_channel_calc) save_channel_leading_ev();
    if (lingaps) lingap_ff();
    if (susc_ff || susc_full) calc_susc(susc_ff, susc_full);
    to_file(filename, save_self, save_full,save_symmetries);
}

void tu_postproduction::_save_channel_leading_ev(complex128_t* channel,
                complex128_t* vec, double* val, char* type, index_t* len, index_t* off) {

    index_t linsize = vertex->n_orbff*POW2(vertex->n_spin);
    double* _val = (double*)malloc(linsize*sizeof(double));
    complex128_t* _vec = vertex->product_helper;

    double* allval = NULL;
    if (algo == 'b' || algo == 'B') {
        allval = (double*) calloc(linsize*vertex->my_nk, sizeof(double));
        memcpy(vec, channel, sizeof(complex128_t)*POW2(linsize)*vertex->my_nk);
        if (algo == 'b')
            batched_eigen_r( NULL, -2, vec, allval, linsize, vertex->my_nk );
        else
            batched_svd( NULL, -1, vec, vec, NULL, allval, linsize, vertex->my_nk );
    }

    for (index_t q = 0; q < vertex->my_nk;++q) {
        if (q > 0) off[q] = off[q-1] + len[q-1];

        if (allval) {
            type[q] = algo;
            memcpy(_val, allval+q*linsize, sizeof(double)*linsize);
            memcpy(_vec, vec+q*POW2(linsize), sizeof(complex128_t)*POW2(linsize));
        } else {
            memcpy(_vec, channel+q*POW2(linsize), sizeof(complex128_t)*POW2(linsize));
            bool hermitian = check_hermicity(_vec,linsize);
            if (algo == 'e' || (algo == 'a' && hermitian)) {
                eigen_solve( NULL, _vec, _val, linsize);
                type[q] = 'e';
            } else if (algo == 's' || (algo == 'a' && !hermitian)) {
                singular_value_decomp( NULL, _vec, vertex->bubble_helper, _val, linsize);
                type[q] = 's';
            }
            // above the matrices are C order, we must transpose that…
            Map<MatXcd>(_vec, linsize, linsize).transposeInPlace();
        }

        // TODO reimplement eigensolution_sort() s.t. only the first couple of
        // eigenvectors are moved in memory
        eigensolution_sort( _vec, _val, linsize, 'M' );

        double cutoff_local = cutoff;
        if (cutoff_rel) cutoff_local *= fabs(_val[0]);

        for (index_t i=0; i<linsize; ++i)
            if (fabs(_val[i]) >= cutoff_local) len[q]++;

        memcpy( val+off[q], _val, sizeof(double)*len[q] );
        memcpy( vec+off[q]*linsize, _vec, sizeof(complex128_t)*len[q]*linsize );
    }
    free(_val);
    if (allval) free(allval);
}

void tu_postproduction::save_channel_leading_ev() {
    if(vertex->P_flow)
        _save_channel_leading_ev(vertex->Pch,
            Pvec, Pval, Ptype, lenP, offP);
    if(vertex->C_flow)
        _save_channel_leading_ev(vertex->Cch,
            Cvec, Cval, Ctype, lenC, offC);
    if(vertex->D_flow)
        _save_channel_leading_ev(vertex->Dch,
            Dvec, Dval, Dtype, lenD, offD);
}

void tu_postproduction::lingap_ff() {
    // only calculates q = 0 linearized gap
    index_t linsize = vertex->n_orbff*POW2(vertex->n_spin);
    called_lingap = true;
    double* _val;
    _val = (double*)malloc(linsize*sizeof(double));
    tu_n_singular_valuesP = (vertex->P_flow) ? tu_n_singular_values : 0;
    tu_n_singular_valuesC = (vertex->C_flow) ? tu_n_singular_values : 0;
    tu_n_singular_valuesD = (vertex->D_flow) ? tu_n_singular_values : 0;
    if(diverge_mpi_comm_rank()==0) {

        Sgap_sc = (double*)malloc(tu_n_singular_valuesP*sizeof(double));
        Sgap_mag = (double*)malloc(tu_n_singular_valuesC*sizeof(double));
        Sgap_charge = (double*)malloc(tu_n_singular_valuesD*sizeof(double));

        Ugap_sc = (complex128_t*)malloc(tu_n_singular_valuesP*linsize*sizeof(complex128_t));
        Ugap_mag = (complex128_t*)malloc(tu_n_singular_valuesC*linsize*sizeof(complex128_t));
        Ugap_charge = (complex128_t*)malloc(tu_n_singular_valuesD*linsize*sizeof(complex128_t));

        Vgap_sc = (complex128_t*)malloc(tu_n_singular_valuesP*linsize*sizeof(complex128_t));
        Vgap_mag = (complex128_t*)malloc(tu_n_singular_valuesC*linsize*sizeof(complex128_t));
        Vgap_charge = (complex128_t*)malloc(tu_n_singular_valuesD*linsize*sizeof(complex128_t));
    }
    if(vertex->P_flow) {
        P_projection(vertex->projection_helper, vertex, proj);
        if(diverge_mpi_comm_rank()==0) {
            batched_gemm(vertex->projection_helper, vertex->pp_bubble_int,
                    vertex->product_helper, POW2(vertex->n_spin)*vertex->n_orbff,1.0, 0.0, 1);
            singular_value_decomp( NULL, vertex->product_helper,
                    vertex->bubble_helper, _val, linsize);
            for(index_t i = 0; i < tu_n_singular_valuesP;++i) {
                Sgap_sc[i] = _val[i];
                for(index_t j = 0; j < linsize; ++j){
                    Ugap_sc[j+linsize*i] = vertex->product_helper[i+linsize*j];
                    Vgap_sc[j+linsize*i] = std::conj(_val[j+linsize*i]);
                }
            }
        }
    }
    if(vertex->C_flow) {
        C_projection(vertex->projection_helper, vertex, proj);

        if(diverge_mpi_comm_rank()==0) {
            batched_gemm(vertex->projection_helper, vertex->ph_bubble_int,
                    vertex->product_helper, POW2(vertex->n_spin)*vertex->n_orbff,1.0, 0.0, 1);
            singular_value_decomp( NULL, vertex->product_helper,
                    vertex->bubble_helper, _val, linsize);
            for(index_t i = 0; i < tu_n_singular_valuesC;++i) {
                Sgap_mag[i] = _val[i];
                for(index_t j = 0; j < linsize; ++j){
                    Ugap_mag[j+linsize*i] = vertex->product_helper[i+linsize*j];
                    Vgap_mag[j+linsize*i] = std::conj(_val[j+linsize*i]);
                }
            }
        }
    }

    if(vertex->D_flow) {
        if(vertex->SU2) {
            complex128_t* helper = (complex128_t*)malloc(sizeof(complex128_t)*POW2(linsize));
            memcpy(helper,vertex->projection_helper,sizeof(complex128_t)*POW2(linsize));
            D_projection(vertex->projection_helper, vertex, proj);
            for(index_t i = 0; i < POW2(linsize);++i) {
                vertex->projection_helper[i] = 2.*vertex->projection_helper[i] - helper[i];
            }
            free(helper);
        }else{
            D_projection(vertex->projection_helper, vertex, proj);
        }

        if(diverge_mpi_comm_rank()==0) {
            batched_gemm(vertex->projection_helper, vertex->ph_bubble_int,
                    vertex->product_helper, POW2(vertex->n_spin)*vertex->n_orbff,1.0, 0.0, 1);
            singular_value_decomp( NULL, vertex->product_helper,
                    vertex->bubble_helper, _val, linsize);
            for(index_t i = 0; i < tu_n_singular_valuesD;++i) {
                Sgap_charge[i] = _val[i];
                for(index_t j = 0; j < linsize; ++j){
                    Ugap_charge[j+linsize*i] = vertex->product_helper[i+linsize*j];
                    Vgap_charge[j+linsize*i] = std::conj(_val[j+linsize*i]);
                }
            }
        }
    }
    free(_val);
}

static inline void fill_Rvec_from_vec(const diverge_model_t* model, const Vec3i* Rptr, double* Rvec) {
    Map<const Mat3d> L(model->lattice[0]);
    index_t nff = ((tu_data_t*)model->internals->tu_data)->existing_ff.size();
    for (index_t i=0; i<nff; ++i)
        Map<Vec3d>(Rvec+3*i) = L * Rptr[i].cast<double>();
}

static void _susc_o14k(const diverge_model_t* const model,
        const complex128_t* const inp, complex128_t* susc) {

    tu_data_t* tu_data = (tu_data_t*)model -> internals -> tu_data;
    const index_t n_spin = model->n_spin;
    const index_t n_orb = model->n_orb;
    const index_t n_bonds = tu_data->n_bonds;
    const index_t realsp_size = tu_data->n_orbff;
    const index_t nk = tu_data->nk;

    const tu_formfactor_t* tu_ff = model->tu_ff;
    const double* kmesh = model->internals->kmesh;
    const index_t* bond_sizes = tu_data->bond_sizes;
    const index_t* bond_offsets = tu_data->bond_offsets;
    const index_t* ob_to_orbff = tu_data->ob_to_orbff;
    const index_t nff = tu_data->existing_ff.size();

    double* R = (double*)malloc(3*nff*sizeof(double));
    fill_Rvec_from_vec(model, tu_data->existing_ff.data(), R);

    complex128_t* FS_ff = (complex128_t*)calloc(nff,sizeof(complex128_t));
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for(index_t ff=0; ff<nff; ++ff)
    for(index_t k=0; k<tu_data->nk; ++k) {
        FS_ff[ff] += exp(-I128*(Map<const Vec3d>(R+3*ff).dot(Map<const Vec3d>(kmesh+3*k)))) / ((double)tu_data->nk);
    }
    free(R);
    const index_t norb = n_spin*n_orb;

    #pragma omp parallel for schedule(static) collapse(7) num_threads(diverge_omp_num_threads())
    for(index_t k = 0; k< nk; ++k)
    for(index_t s1 = 0; s1 < (n_spin); ++s1)
    for(index_t s2 = 0; s2 < (n_spin); ++s2)
    for(index_t s3 = 0; s3 < (n_spin); ++s3)
    for(index_t s4 = 0; s4 < (n_spin); ++s4)
    for(index_t o1 = 0; o1<n_orb;++o1)
    for(index_t o3 = 0; o3<n_orb;++o3)
    for(index_t b1 = 0; b1<bond_sizes[o1];++b1)
    for(index_t b3 = 0; b3<bond_sizes[o3];++b3){
        const index_t o2 = tu_ff[bond_offsets[o1]+b1].oto;
        const index_t o4 = tu_ff[bond_offsets[o3]+b3].oto;
        const index_t ff = tu_ff[bond_offsets[o1]+b1].ffidx;
        const index_t ff2 = tu_ff[bond_offsets[o3]+b3].ffidx;
        index_t so1 = o1+n_orb*s1;
        index_t so2 = o2+n_orb*s2;
        index_t so3 = o3+n_orb*s3;
        index_t so4 = o4+n_orb*s4;

        index_t sp1 = s1+n_spin*s2;
        index_t sp3 = s3+n_spin*s4;
        const complex128_t ffv = FS_ff[ff];
        const complex128_t ffv2 = FS_ff[ff2];

        susc[IDX5(so4,so3,so2,so1,k, norb,norb,norb,nk)] += std::conj(ffv)*ffv2*
            inp[k + nk * (ob_to_orbff[b1+n_bonds*o1]+realsp_size
                            *(sp1+n_spin*n_spin*(ob_to_orbff[b3+n_bonds*o3]+realsp_size
                                *(sp3))))];
    }
    free(FS_ff);
}

static void vertex_product(complex128_t* result, const complex128_t* const  B,
                    const complex128_t* const C, complex128_t* helper,
                    const index_t productsize, const index_t nbatch,
                    const double prefac)
{
    batched_gemm(B, C, helper, productsize,1.0, 0.0, nbatch);
    batched_gemm(helper, B, result, productsize,prefac, 0.0, nbatch);
}

void tu_postproduction::calc_susc(bool susc_ff, bool susc_full) {
    index_t size = POW4(vertex->n_orb*vertex->n_spin)*vertex->nk;
    index_t size2 = POW4(vertex->n_spin)*POW2(vertex->n_orbff)*vertex->nk;
    called_suscep_full = susc_full;
    called_suscep_ff = susc_ff;
    tu_suscP = 0; tu_suscP_ff = 0;
    tu_suscC = 0; tu_suscC_ff = 0;
    tu_suscD = 0; tu_suscD_ff = 0;
    
    complex128_t* suscff = (complex128_t*)malloc(vertex->full_vert_size*sizeof(complex128_t));
    if(vertex->P_flow) {
        tu_suscP = (called_suscep_full) ? POW4(vertex->n_orb*vertex->n_spin)*vertex->nk : 0;
        tu_suscP_ff = (called_suscep_ff) ? POW2(vertex->n_orbff*POW2(vertex->n_spin))*vertex->nk : 0;
        suscPp_ff = (complex128_t*)malloc(size2*sizeof(complex128_t));
        P_projection(vertex->projection_helper, vertex, proj);
        vertex_product(suscff, vertex->pp_bubble_int,vertex->projection_helper,
                vertex->product_helper, POW2(vertex->n_spin)*vertex->n_orbff, vertex->my_nk,1.0);
        vertex->template reco<'P'>(suscPp_ff, suscff);
        if(called_suscep_full) {
            suscPp = (complex128_t*)calloc(size,sizeof(complex128_t));
            _susc_o14k(model,suscPp_ff,suscPp);
        }
    }
    if(vertex->C_flow) {
        tu_suscC = (called_suscep_full) ? POW4(vertex->n_orb*vertex->n_spin)*vertex->nk : 0;
        tu_suscC_ff = (called_suscep_ff) ? POW2(vertex->n_orbff*POW2(vertex->n_spin))*vertex->nk : 0;
        suscMa_ff = (complex128_t*)malloc(size2*sizeof(complex128_t));
        C_projection(vertex->projection_helper, vertex, proj);
        vertex_product(suscff, vertex->ph_bubble_int,vertex->projection_helper,
                vertex->product_helper, POW2(vertex->n_spin)*vertex->n_orbff, vertex->my_nk,1.0);
        vertex->template reco<'C'>(suscMa_ff, suscff);
        if(called_suscep_full) {
            suscMa = (complex128_t*)calloc(size,sizeof(complex128_t));
            _susc_o14k(model,suscMa_ff,suscMa);
        }
    }
    if(vertex->D_flow) {
        tu_suscD = (called_suscep_full) ? POW4(vertex->n_orb*vertex->n_spin)*vertex->nk : 0;
        tu_suscD_ff = (called_suscep_ff) ? POW2(vertex->n_orbff*POW2(vertex->n_spin))*vertex->nk : 0;
        suscCh_ff = (complex128_t*)malloc(size2*sizeof(complex128_t));
        if(vertex->SU2 && vertex->C_flow) {
            complex128_t* helper;
            helper = (complex128_t*)calloc(vertex->full_vert_size,sizeof(complex128_t));
            D_projection(helper, vertex, proj);
            for(index_t i = 0; i < vertex->full_vert_size;++i)
                vertex->projection_helper[i] = 2.*helper[i] - vertex->projection_helper[i];
            free(helper);
        }else{
            D_projection(vertex->projection_helper, vertex, proj);
        }
        vertex_product(suscff, vertex->ph_bubble_int,vertex->projection_helper,
                vertex->product_helper, POW2(vertex->n_spin)*vertex->n_orbff, vertex->my_nk,1.0);
        vertex->template reco<'D'>(suscCh_ff, suscff);
        if(called_suscep_full) {
            suscCh = (complex128_t*)calloc(size,sizeof(complex128_t));
            _susc_o14k(model,suscCh_ff,suscCh);
        }
    }
    if(!susc_ff) {
        free(suscCh_ff);
        free(suscPp_ff);
        free(suscMa_ff);
    }
    free(suscff);
}

typedef Eigen::Matrix<double,3,3,Eigen::RowMajor> CMat3d;

void tu_postproduction::construct_multiindex_maps(const diverge_model_t* model) {
    mpi_vrb_printf("Generate one-to-many for all q for storing\n");
    tu_data_t* tu_data = (tu_data_t*)(model->internals->tu_data);
    if(!tu_data->symm->use_symmetries) {
        return;
    }
    
    symmstruct_t* symm = tu_data->symm;
    index_t n_spin = model->n_spin;
    index_t nkibz = tu_data->nkibz;
    index_t n_orbff = tu_data->n_orbff;
    
    mi_map_pref_in_in.resize(nkibz*symm->mi_maps_to.size());
    mi_map_pref_in_out.resize(nkibz*symm->mi_maps_to.size());
    mi_map_idx.resize(nkibz*symm->mi_maps_to.size());
    
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for(index_t idx = 0; idx<n_orbff*POW2(n_spin)*model->n_sym;++idx)
    for(index_t q = 0; q < nkibz; ++q) {
        Map<Vec3d> kp(model->internals->kmesh+3*symm->idx_ibz_in_fullmesh[q]);
        const complex128_t pref_sym =
                exp(-I128*kp.dot(symm->beyond_UC[idx]));

        for(index_t to = 0; to < symm->mi_map_len[idx]; ++to) {
            mi_map_idx[q+nkibz*(to+symm->mi_map_off[idx])] = symm->mi_maps_to[to + symm->mi_map_off[idx]]+n_orbff*POW2(n_spin)*q;
            complex128_t pref_elem = symm->mi_prefac_in_in[to + symm->mi_map_off[idx]];
            mi_map_pref_in_in[q+nkibz*(to+symm->mi_map_off[idx])] = (pref_elem*pref_sym);

            pref_elem = symm->mi_prefac_in_out[to + symm->mi_map_off[idx]];
            mi_map_pref_in_out[q+nkibz*(to+symm->mi_map_off[idx])] = (pref_elem*pref_sym);
        }
    }
}


void tu_postproduction::to_file(string filename, bool save_self, bool save_full, bool save_symmetries) {
    index_t* header = (index_t*)calloc(128,sizeof(index_t));
    const char* version = tag_version();
    tu_data_t* tu_data = (tu_data_t*)model -> internals -> tu_data;
    symmstruct_t* symm = tu_data->symm;
    int i = 0;
    index_t my_rank = diverge_mpi_comm_rank();
    const index_t my_nk = vertex->my_nk;
    index_t cop = 0;
    index_t coc = 0;
    index_t cod = 0;
    index_t nk_channel = 0;
    index_t my_cop = 0;
    index_t my_coc = 0;
    index_t my_cod = 0;
    if(!skip_channel_calc) {
        my_cop = lenP[my_nk-1]+offP[my_nk-1];
        my_coc = lenC[my_nk-1]+offC[my_nk-1];
        my_cod = lenD[my_nk-1]+offD[my_nk-1];
        nk_channel = tu_data->nkibz;
        for(index_t rank = 0; rank < diverge_mpi_comm_size();++rank){
            cop = my_cop;
            coc = my_coc;
            cod = my_cod;
            diverge_mpi_bcast_bytes((void*)(&cop), sizeof(index_t), rank);
            diverge_mpi_bcast_bytes((void*)(&coc), sizeof(index_t), rank);
            diverge_mpi_bcast_bytes((void*)(&cod), sizeof(index_t), rank);
            if(my_rank > rank) {
                for(index_t k = 0; k < my_nk; ++k) {
                    offP[k] += cop;
                    offC[k] += coc;
                    offD[k] += cod;
                }
            }
        }
        // total numer of elements per channel
        cop = lenP[my_nk-1]+offP[my_nk-1];
        coc = lenC[my_nk-1]+offC[my_nk-1];
        cod = lenD[my_nk-1]+offD[my_nk-1];
        diverge_mpi_bcast_bytes((void*)(&cop), sizeof(index_t), diverge_mpi_comm_size() - 1);
        diverge_mpi_bcast_bytes((void*)(&coc), sizeof(index_t), diverge_mpi_comm_size() - 1);
        diverge_mpi_bcast_bytes((void*)(&cod), sizeof(index_t), diverge_mpi_comm_size() - 1);

    }

    index_t linsize = vertex->n_orbff*POW2(vertex->n_spin);
    const index_t n_orb = model->n_orb;


    int size_self = 0;
    if(vertex->use_self && save_self)
        size_self = POW2(n_orb*vertex->n_spin)*vertex->nk*vertex->nkf;

    mpi_vrb_printf_all( "rank %d: try opening file %s\n", my_rank, filename.c_str() );
    void* file = diverge_mpi_open_file(filename.c_str());
    header[i] = TU_MAGIC_NUMBER_POST_PROCESSING,++i;
    header[i] = vertex->n_orb, ++i;
    header[i] = vertex->n_spin, ++i;
    header[i] = vertex->nk, ++i;
    header[i] = vertex->nk*vertex->nkf, ++i;
    header[i] = tu_data->nkibz, ++i;
    header[i] = model->SU2, ++i;
    header[i] = vertex->n_orbff, ++i;
    header[i] = vertex->n_bonds, ++i;
    header[i] = model->n_sym, ++i;

    header[i] = 128*sizeof(index_t), ++i;
    header[i] = vertex->n_orbff, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = vertex->n_orbff, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = 3*vertex->n_orbff, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = vertex->n_orb, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = vertex->n_orb, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = tu_data->nkibz, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    // store vertex diagonalizations first
    // P channel
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(char), ++i;
    header[i] = cop, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double), ++i;
    header[i] = cop*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    // C channel
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(char), ++i;
    header[i] = coc, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double), ++i;
    header[i] = coc*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    // D channel
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = nk_channel, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(char), ++i;
    header[i] = cod, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double), ++i;
    header[i] = cod*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    // lingap used?
    header[i] = tu_n_singular_valuesP, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double), ++i;
    header[i] = tu_n_singular_valuesP*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = tu_n_singular_valuesP*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;

    header[i] = tu_n_singular_valuesC, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double), ++i;
    header[i] = tu_n_singular_valuesC*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = tu_n_singular_valuesC*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;

    header[i] = tu_n_singular_valuesD, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double), ++i;
    header[i] = tu_n_singular_valuesD*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = tu_n_singular_valuesD*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;

    header[i] = tu_suscP,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = tu_suscC,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = tu_suscD,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;

    header[i] = tu_suscP_ff,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = tu_suscC_ff,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = tu_suscD_ff,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = size_self,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    if(save_full && !tu_data->tu_selfenergy_flow) {
        header[i] = vertex->P_flow ? POW2(linsize)*vertex->nk: 0,++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = vertex->C_flow ? POW2(linsize)*vertex->nk: 0,++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = vertex->D_flow ? POW2(linsize)*vertex->nk: 0,++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = vertex->P_flow ? POW2(linsize)*vertex->nk: 0,++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = (vertex->C_flow || vertex->D_flow) ? POW2(linsize)*vertex->nk:0, ++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    } else {
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    }

    if (save_symmetries && tu_data->symm->use_symmetries) {
        construct_multiindex_maps(model);
        header[i] = symm->mi_map_len.size();++i;
        header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
        header[i] = symm->mi_map_off.size();++i;
        header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
        header[i] = mi_map_idx.size();++i;
        header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
        header[i] = mi_map_pref_in_in.size();++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = mi_map_pref_in_out.size();++i;
    } else {
        if (save_symmetries) {
            mpi_wrn_printf("no symmetries were used in the calculation! \n");
        }
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
        header[i] = 0;++i;
        header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
        header[i] = 0;++i;
    }
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t); ++i;
    header[i] = strlen(version); ++i;

    if (save_symmetries && tu_data->symm->use_symmetries) {
        header[i] = header[i-2] + header[i-1]*sizeof(char); ++i;
        header[i] = symm->kmaps_to.size(); ++i;
    }

    assert(i < 126);
    header[126] = TU_MAGIC_NUMBER_POST_PROCESSING;
    header[127] = header[i-1]*sizeof(complex128_t)+header[i-2];

    index_t my_nk_off = vertex->my_nk_off;
    index_t sindex = sizeof(index_t);
    index_t schar = sizeof(char);
    index_t sdou = sizeof(double);
    index_t scdou = sizeof(complex128_t);
    index_t new_off = 0;

    new_off = diverge_mpi_write_byte_to_file_master(file, header, new_off, 128*sindex);
    new_off = diverge_mpi_write_byte_to_file_master(file, tu_data->mi_to_ofrom, new_off, vertex->n_orbff*sindex);
    new_off = diverge_mpi_write_byte_to_file_master(file, tu_data->mi_to_oto, new_off, vertex->n_orbff*sindex);
    new_off = diverge_mpi_write_byte_to_file_master(file, tu_data->mi_to_R, new_off, 3*vertex->n_orbff*sindex);
    new_off = diverge_mpi_write_byte_to_file_master(file, tu_data->bond_sizes, new_off, vertex->n_orb*sindex);
    new_off = diverge_mpi_write_byte_to_file_master(file, tu_data->bond_offsets, new_off, vertex->n_orb*sindex);

    new_off = diverge_mpi_write_byte_to_file_master(file, tu_data->symm->idx_ibz_in_fullmesh.data(), new_off, tu_data->nkibz*sindex);
    // P vertex write
    if(!skip_channel_calc) {
        new_off = diverge_mpi_write_byte_to_file(file, lenP, new_off + my_nk_off*sindex, my_nk*sindex);
        new_off = diverge_mpi_write_byte_to_file(file, offP, new_off + my_nk_off*sindex, my_nk*sindex);
        new_off = diverge_mpi_write_byte_to_file(file, Ptype, new_off + my_nk_off*schar, my_nk*schar);
        new_off = diverge_mpi_write_byte_to_file(file, Pval, new_off + offP[0]*sdou, my_cop*sdou);
        new_off = diverge_mpi_write_byte_to_file(file, Pvec, new_off + offP[0]*linsize*scdou, my_cop*linsize*scdou);

        // C vertex write
        new_off = diverge_mpi_write_byte_to_file(file, lenC, new_off + my_nk_off*sindex, my_nk*sindex);
        new_off = diverge_mpi_write_byte_to_file(file, offC, new_off + my_nk_off*sindex, my_nk*sindex);
        new_off = diverge_mpi_write_byte_to_file(file, Ctype, new_off + my_nk_off*schar, my_nk*schar);
        new_off = diverge_mpi_write_byte_to_file(file, Cval, new_off + offC[0]*sdou, my_coc*sdou);
        new_off = diverge_mpi_write_byte_to_file(file, Cvec, new_off + offC[0]*linsize*scdou, my_coc*linsize*scdou);

        // D vertex write
        new_off = diverge_mpi_write_byte_to_file(file, lenD, new_off + my_nk_off*sindex, my_nk*sindex);
        new_off = diverge_mpi_write_byte_to_file(file, offD, new_off + my_nk_off*sindex, my_nk*sindex);
        new_off = diverge_mpi_write_byte_to_file(file, Dtype, new_off + my_nk_off*schar, my_nk*schar);
        new_off = diverge_mpi_write_byte_to_file(file, Dval, new_off + offD[0]*sdou, my_cod*sdou);
        new_off = diverge_mpi_write_byte_to_file(file, Dvec, new_off + offD[0]*linsize*scdou, my_cod*linsize*scdou);
    }
    if(called_lingap) {
        new_off = diverge_mpi_write_byte_to_file_master(file, Sgap_sc, new_off, sdou*tu_n_singular_valuesP);
        new_off = diverge_mpi_write_byte_to_file_master(file, Ugap_sc, new_off, scdou*tu_n_singular_valuesP*linsize);
        new_off = diverge_mpi_write_byte_to_file_master(file, Vgap_sc, new_off, scdou*tu_n_singular_valuesP*linsize);

        new_off = diverge_mpi_write_byte_to_file_master(file, Sgap_mag, new_off, sdou*tu_n_singular_valuesC);
        new_off = diverge_mpi_write_byte_to_file_master(file, Ugap_mag, new_off, scdou*tu_n_singular_valuesC*linsize);
        new_off = diverge_mpi_write_byte_to_file_master(file, Vgap_mag, new_off, scdou*tu_n_singular_valuesC*linsize);

        new_off = diverge_mpi_write_byte_to_file_master(file, Sgap_charge, new_off, sdou*tu_n_singular_valuesD);
        new_off = diverge_mpi_write_byte_to_file_master(file, Ugap_charge, new_off, scdou*tu_n_singular_valuesD*linsize);
        new_off = diverge_mpi_write_byte_to_file_master(file, Vgap_charge, new_off, scdou*tu_n_singular_valuesD*linsize);
    }

    if(called_suscep_full) {
        new_off = diverge_mpi_write_byte_to_file_master(file, suscPp,
                            new_off,scdou*tu_suscP);
        new_off = diverge_mpi_write_byte_to_file_master(file, suscMa,
                            new_off,scdou*tu_suscC);
        new_off = diverge_mpi_write_byte_to_file_master(file, suscCh,
                            new_off,scdou*tu_suscD);

    }
    if(called_suscep_ff) {
        new_off = diverge_mpi_write_byte_to_file_master(file, suscPp_ff,
                            new_off,scdou*tu_suscP_ff);
        new_off = diverge_mpi_write_byte_to_file_master(file, suscMa_ff,
                            new_off,scdou*tu_suscC_ff);
        new_off = diverge_mpi_write_byte_to_file_master(file, suscCh_ff,
                            new_off,scdou*tu_suscD_ff);
    }

    if(vertex->use_self && save_self) {
        new_off = diverge_mpi_write_byte_to_file_master(file, vertex->self_en,
                        new_off,scdou*size_self);

    }
    if(save_full && !tu_data->tu_selfenergy_flow) {
        index_t vsize = POW2(linsize)*vertex->nk*sizeof(complex128_t);
        complex128_t* helper = (complex128_t*)malloc(vsize);
        if(vertex->P_flow) {
            vertex->template reco<'P'>(helper);
            new_off = diverge_mpi_write_byte_to_file_master(file, helper, new_off,vsize);
        }
        if(vertex->C_flow) {
            vertex->template reco<'C'>(helper);
            new_off = diverge_mpi_write_byte_to_file_master(file, helper, new_off,vsize);
        }
        if(vertex->D_flow) {
            vertex->template reco<'D'>(helper);
            new_off = diverge_mpi_write_byte_to_file_master(file, helper, new_off,vsize);
        }
        if(vertex->P_flow) {
            vertex->template reco<'P'>(helper, vertex->pp_bubble_int);
            new_off = diverge_mpi_write_byte_to_file_master(file, helper, new_off,vsize);
        }
        if(vertex->C_flow || vertex->D_flow) {
            vertex->template reco<'C'>(helper, vertex->ph_bubble_int);
            new_off = diverge_mpi_write_byte_to_file_master(file, helper, new_off,vsize);
        }
        free(helper);
    }
    if(save_symmetries && tu_data->symm->use_symmetries) {
        mpi_log_printf("saving symm!\n");
        new_off = diverge_mpi_write_byte_to_file_master(file, symm->mi_map_len.data(), new_off,sindex*symm->mi_map_len.size());
        new_off = diverge_mpi_write_byte_to_file_master(file, symm->mi_map_off.data(), new_off,sindex*symm->mi_map_off.size());
        new_off = diverge_mpi_write_byte_to_file_master(file, mi_map_idx.data(), new_off,sindex*mi_map_idx.size());
        new_off = diverge_mpi_write_byte_to_file_master(file, mi_map_pref_in_in.data(), new_off,scdou*mi_map_pref_in_in.size());
        new_off = diverge_mpi_write_byte_to_file_master(file, mi_map_pref_in_out.data(), new_off,scdou*mi_map_pref_in_out.size());
    }

    new_off = diverge_mpi_write_byte_to_file_master(file, version, new_off, strlen(version));
    new_off = diverge_mpi_write_byte_to_file_master(file, symm->kmaps_to.data(), new_off, symm->kmaps_to.size() * sindex);
    diverge_mpi_close_file(file);
    free(header);
}
