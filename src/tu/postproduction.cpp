/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "postproduction.hpp"
#include "../misc/batched_eigen.h"
#include "../misc/batched_gemms.h"
#include "../misc/eigen.h"

static bool check_hermicity(complex128_t* data, index_t linsize){
    double err_max = 0.;
    #pragma omp parallel for reduction(max:err_max)
    for(index_t i=0; i<linsize; ++i)
    for(index_t j=0; j<linsize; ++j) {
        double err = std::abs(data[IDX2(i,j,linsize)] - std::conj(data[IDX2(j,i,linsize)]));
        err_max = MAX(err_max, err);
    }
    return err_max < 1e-11;
}

tu_postproduction::tu_postproduction(const diverge_model_t* mod, Vertex* Vertexin, Projection* projin)
:
    vertex{Vertexin},
    proj{projin},
    model{mod}
{
    algo = 'a';
    cutoff = 50;
    cutoff_rel = false;
    tu_n_singular_values = 1;

    index_t linsize = vertex->n_orbff*POW2(vertex->n_spin);
    Pval = (double*)calloc(linsize*vertex->my_nk,sizeof(double));
    Cval = (double*)calloc(linsize*vertex->my_nk,sizeof(double));
    Dval = (double*)calloc(linsize*vertex->my_nk,sizeof(double));
    Ptype = (char*)calloc(vertex->my_nk,sizeof(char));
    Ctype = (char*)calloc(vertex->my_nk,sizeof(char));
    Dtype = (char*)calloc(vertex->my_nk,sizeof(char));
    Pvec = (complex128_t*)calloc(POW4(vertex->n_spin)*POW2(vertex->n_orb*vertex->n_bonds)*vertex->my_nk,sizeof(complex128_t));
    Cvec = (complex128_t*)calloc(POW4(vertex->n_spin)*POW2(vertex->n_orb*vertex->n_bonds)*vertex->my_nk,sizeof(complex128_t));
    Dvec = (complex128_t*)calloc(POW4(vertex->n_spin)*POW2(vertex->n_orb*vertex->n_bonds)*vertex->my_nk,sizeof(complex128_t));

    lenP = (index_t*)calloc(vertex->my_nk,sizeof(index_t));
    lenC = (index_t*)calloc(vertex->my_nk,sizeof(index_t));
    lenD = (index_t*)calloc(vertex->my_nk,sizeof(index_t));
    offP = (index_t*)calloc(vertex->my_nk,sizeof(index_t));
    offC = (index_t*)calloc(vertex->my_nk,sizeof(index_t));
    offD = (index_t*)calloc(vertex->my_nk,sizeof(index_t));
}


tu_postproduction::~tu_postproduction() {
    free(Pval);
    free(Cval);
    free(Dval);
    free(Ptype);
    free(Ctype);
    free(Dtype);
    free(Pvec);
    free(Cvec);
    free(Dvec);
    free(lenP);
    free(lenC);
    free(lenD);
    free(offP);
    free(offC);
    free(offD);
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
                                            bool susc_full, bool save_self) {
    safe_channel_leading_ev();
    if(lingaps) lingap_ff();
    if(susc_ff || susc_full) calc_susc(susc_ff, susc_full);
    to_file(filename, save_self);
}

void tu_postproduction::_safe_channel_leading_ev(complex128_t* channel,
                complex128_t* vec, double* val, char* type, index_t* len, index_t* off){

    index_t linsize = vertex->n_orbff*POW2(vertex->n_spin);
    double* _val = (double*)malloc(linsize*sizeof(double));
    complex128_t* _vec = (complex128_t*)malloc(POW2(linsize)*sizeof(complex128_t));
    tu_data_t* tu_data = (tu_data_t*)(model -> internals-> tu_data);
    index_t n_bonds = tu_data->n_bonds;
    index_t n_spin = vertex->n_spin;
    index_t n_orb = vertex->n_orb;
    index_t* bond_sizes = tu_data->bond_sizes;
    index_t* ob_to_orbff = tu_data->ob_to_orbff;

    batched_eigen_shut_up( 1 );
    for(index_t q = 0; q < vertex->my_nk;++q) {
        if(q > 0) off[q] = off[q-1] + len[q-1];
        memcpy(_vec, channel+q*POW2(linsize), sizeof(complex128_t)*POW2(linsize));
        bool hermitian = check_hermicity(_vec,linsize);
        if(algo == 'e' || (algo == 'a' && hermitian)) {
            batched_eigen_r( NULL, -2, _vec, _val, linsize, 1 );
            type[q] = 'e';
        }else{
            singular_value_decomp( NULL, _vec, vertex->bubble_helper, _val, linsize);
            type[q] = 's';
        }
        vector<double> vals;
        vector<index_t> idx;

        double cutoff_local = cutoff;

        if (cutoff_rel) {
            vector<double> vals_sorted(linsize);
            memcpy( vals_sorted.data(), _val, sizeof(double)*linsize );
            std::sort( vals_sorted.begin(), vals_sorted.end(), [](double a, double b)->bool{return std::abs(a)>std::abs(b);} );
            cutoff_local *= std::abs(vals_sorted[0]);
        }

        for(index_t r = 0; r < linsize; ++r) {
            double cur = _val[r];
            if(std::abs(cur) > cutoff_local){
                len[q] += 1;
                vals.push_back(cur);
                idx.push_back(r);
            }
        }
        if(len[q]>0) {
            auto p = sort_permutation(vals);
            apply_permutation_in_place(vals, p);
            apply_permutation_in_place(idx, p);
            memcpy(val+off[q],vals.data(),sizeof(double)*len[q]);

            #pragma omp parallel for schedule(static) collapse(3)
            for(index_t m1 = 0; m1 < POW2(n_spin); ++m1)
            for(uindex_t j = 0; j<idx.size();++j)
            for(index_t o1 = 0; o1<vertex->n_orbff;++o1) {
                vec[o1+vertex->n_orbff*(m1 + POW2(n_spin)*(j))+linsize*off[q]] =
                        _vec[idx[j]+linsize*(o1+vertex->n_orbff*m1)];
            }
        }
    }
    batched_eigen_shut_up( 0 );
    free(_val);
    free(_vec);
}

void tu_postproduction::safe_channel_leading_ev() {
    if(vertex->P_flow)
        _safe_channel_leading_ev(vertex->Pch,
            Pvec, Pval, Ptype, lenP, offP);
    if(vertex->C_flow)
        _safe_channel_leading_ev(vertex->Cch,
            Cvec, Cval, Ctype, lenC, offC);
    if(vertex->D_flow)
        _safe_channel_leading_ev(vertex->Dch,
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
    const Map<Mat3d> L((double*)(model->lattice));
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
    const index_t my_nk = tu_data->my_nk;

    const tu_formfactor_t* tu_ff = model->tu_ff;
    const double* kmesh = model->internals->kmesh;
    const index_t* bond_sizes = tu_data->bond_sizes;
    const index_t* bond_offsets = tu_data->bond_offsets;
    const index_t* ob_to_orbff = tu_data->ob_to_orbff;
    const index_t nff = tu_data->existing_ff.size();

    double* R = (double*)malloc(3*nff*sizeof(double));
    fill_Rvec_from_vec(model, tu_data->existing_ff.data(), R);

    complex128_t* FS_ff = (complex128_t*)calloc(nff,sizeof(complex128_t));
    #pragma omp parallel for
    for(index_t ff=0; ff<nff; ++ff)
    for(index_t k=0; k<tu_data->nk; ++k) {
        FS_ff[ff] += exp(-I128*(R[3*ff+0]*kmesh[3*k+0]
                               +R[3*ff+1]*kmesh[3*k+1]
                               +R[3*ff+2]*kmesh[3*k+2]))/((double)tu_data->nk);
    }
    free(R);
    const index_t norb = n_spin*n_orb;

    #pragma omp parallel for schedule(static) collapse(7)
    for(index_t k = 0; k< my_nk; ++k)
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

        susc[so1+norb*(so2+norb*(so3+norb*(so4+norb*k)))] +=
            std::conj(ffv)*ffv2*inp[ob_to_orbff[b1+n_bonds*o1]+realsp_size
                            *(sp1+n_spin*n_spin*(ob_to_orbff[b3+n_bonds*o3]+realsp_size
                                *(sp3+n_spin*n_spin*k)))];
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
    index_t size = POW4(vertex->n_orb*vertex->n_spin)*vertex->my_nk;
    called_suscep_full = susc_full;
    called_suscep_ff = susc_ff;
    if(susc_full) {
        suscPp = (complex128_t*)malloc(size*sizeof(complex128_t));
        suscMa = (complex128_t*)malloc(size*sizeof(complex128_t));
        suscCh = (complex128_t*)malloc(size*sizeof(complex128_t));
    }
    suscPp_ff = (complex128_t*)malloc(vertex->full_vert_size*sizeof(complex128_t));
    suscMa_ff = (complex128_t*)malloc(vertex->full_vert_size*sizeof(complex128_t));
    suscCh_ff = (complex128_t*)malloc(vertex->full_vert_size*sizeof(complex128_t));
    if(vertex->P_flow) {
        tu_suscP = (called_suscep_full) ? POW4(vertex->n_orb*vertex->n_spin) : 0;
        tu_suscP_ff = (called_suscep_ff) ? POW2(vertex->n_orbff*POW2(vertex->n_spin)) : 0;
        P_projection(vertex->projection_helper, vertex, proj);
        vertex_product(suscPp_ff, vertex->pp_bubble_int,vertex->projection_helper,
                vertex->product_helper, POW2(vertex->n_spin)*vertex->n_orbff, vertex->my_nk,1.0);
        if(called_suscep_full)
            _susc_o14k(model,suscPp_ff,suscPp);
    }
    if(vertex->C_flow) {
        tu_suscC = (called_suscep_full) ? POW4(vertex->n_orb*vertex->n_spin) : 0;
        tu_suscC_ff = (called_suscep_ff) ? POW2(vertex->n_orbff*POW2(vertex->n_spin)) : 0;
        C_projection(vertex->projection_helper, vertex, proj);
        vertex_product(suscMa_ff, vertex->ph_bubble_int,vertex->projection_helper,
                vertex->product_helper, POW2(vertex->n_spin)*vertex->n_orbff, vertex->my_nk,1.0);
        if(called_suscep_full)
            _susc_o14k(model,suscMa_ff,suscMa);
    }
    if(vertex->D_flow) {
        tu_suscD = (called_suscep_full) ? POW4(vertex->n_orb*vertex->n_spin) : 0;
        tu_suscD_ff = (called_suscep_ff) ? POW2(vertex->n_orbff*POW2(vertex->n_spin)) : 0;
        if(vertex->SU2) {
            complex128_t* helper;
            helper = (complex128_t*)calloc(vertex->full_vert_size,sizeof(complex128_t));
            D_projection(helper, vertex, proj);
            for(index_t i = 0; i < vertex->full_vert_size;++i)
                vertex->projection_helper[i] = 2.*helper[i] - vertex->projection_helper[i];
            free(helper);
        }else{
            D_projection(vertex->projection_helper, vertex, proj);
        }
        vertex_product(suscCh_ff, vertex->ph_bubble_int,vertex->projection_helper,
                vertex->product_helper, POW2(vertex->n_spin)*vertex->n_orbff, vertex->my_nk,1.0);
        if(called_suscep_full)
            _susc_o14k(model,suscCh_ff,suscCh);
    }
    if(!susc_ff) {
        free(suscCh_ff);
        free(suscPp_ff);
        free(suscMa_ff);
    }
}


void tu_postproduction::to_file(string filename, bool save_self) {
    index_t* header = (index_t*)calloc(128,sizeof(index_t));
    tu_data_t* tu_data = (tu_data_t*)model -> internals -> tu_data;
    int i = 0;
    index_t my_rank = diverge_mpi_comm_rank();
    const index_t my_nk = vertex->my_nk;
    index_t cop,coc,cod;
    for(index_t rank = 0; rank < diverge_mpi_comm_size();++rank){
        cop = lenP[my_nk-1]+offP[my_nk-1];
        coc = lenC[my_nk-1]+offC[my_nk-1];
        cod = lenD[my_nk-1]+offD[my_nk-1];
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
    // my number of elements per channel
    index_t my_cop = 0;
    index_t my_coc = 0;
    index_t my_cod = 0;
    for(index_t k = 0; k < my_nk; ++k) {
        my_cop += lenP[k];
        my_coc += lenC[k];
        my_cod += lenD[k];
    }
    index_t linsize = vertex->n_orbff*POW2(vertex->n_spin);
    const index_t n_orb = model->n_orb;
    const index_t n_bonds = tu_data->n_bonds;

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
    header[i] = tu_data->nkibz, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = tu_data->nkibz, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = tu_data->nkibz, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(char), ++i;
    header[i] = cop, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double), ++i;
    header[i] = cop*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    // C channel
    header[i] = tu_data->nkibz, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = tu_data->nkibz, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = tu_data->nkibz, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(char), ++i;
    header[i] = coc, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(double), ++i;
    header[i] = coc*linsize, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    // D channel
    header[i] = tu_data->nkibz, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = tu_data->nkibz, ++i;
    header[i] = header[i-2] + header[i-1]*sizeof(index_t), ++i;
    header[i] = tu_data->nkibz, ++i;
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

    header[i] = tu_suscP*tu_data->nkibz,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = tu_suscC*tu_data->nkibz,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = tu_suscD*tu_data->nkibz,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;

    header[i] = tu_suscP_ff*tu_data->nkibz,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = tu_suscC_ff*tu_data->nkibz,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = tu_suscD_ff*tu_data->nkibz,++i;
    header[i] = header[i-2] + header[i-1]*sizeof(complex128_t), ++i;
    header[i] = size_self,++i;
    assert(i < 126);
    header[126] = TU_MAGIC_NUMBER_POST_PROCESSING;
    header[127] = header[i-1]*sizeof(complex128_t)+header[i-2];

    index_t my_nk_off = vertex->my_nk_off;
    index_t sindex = sizeof(index_t);
    index_t schar = sizeof(char);
    index_t sdou = sizeof(double);
    index_t scdou = sizeof(complex128_t);
    index_t new_off = 0;

    new_off = diverge_mpi_write_byte_to_file(file, header, new_off, 128*sindex);
    index_t n_osff = (my_rank == 0) ? vertex->n_orbff : 0;
    new_off = diverge_mpi_write_byte_to_file(file, tu_data->mi_to_ofrom, new_off, n_osff*sindex);
    new_off = diverge_mpi_write_byte_to_file(file, tu_data->mi_to_oto, new_off, n_osff*sindex);
    new_off = diverge_mpi_write_byte_to_file(file, tu_data->mi_to_R, new_off, 3*n_osff*sindex);
    n_osff = (my_rank == 0) ? vertex->n_orb : 0;
    new_off = diverge_mpi_write_byte_to_file(file, tu_data->bond_sizes, new_off, n_osff*sindex);
    new_off = diverge_mpi_write_byte_to_file(file, tu_data->bond_offsets, new_off, n_osff*sindex);

    index_t nibz = (my_rank == 0) ? tu_data->nkibz : 0;
    new_off = diverge_mpi_write_byte_to_file(file, tu_data->symm->idx_ibz_in_fullmesh.data(), new_off, nibz*sindex);
    // P vertex write
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
    if(called_lingap) {
        index_t my_nelem = (my_rank == 0) ? tu_n_singular_valuesP : 0;
        new_off = diverge_mpi_write_byte_to_file(file, Sgap_sc, new_off, sdou*my_nelem);
        new_off = diverge_mpi_write_byte_to_file(file, Ugap_sc, new_off, scdou*my_nelem*linsize);
        new_off = diverge_mpi_write_byte_to_file(file, Vgap_sc, new_off, scdou*my_nelem*linsize);

        my_nelem = (my_rank == 0) ? tu_n_singular_valuesC : 0;
        new_off = diverge_mpi_write_byte_to_file(file, Sgap_mag, new_off, sdou*my_nelem);
        new_off = diverge_mpi_write_byte_to_file(file, Ugap_mag, new_off, scdou*my_nelem*linsize);
        new_off = diverge_mpi_write_byte_to_file(file, Vgap_mag, new_off, scdou*my_nelem*linsize);

        my_nelem = (my_rank == 0) ? tu_n_singular_valuesD : 0;
        new_off = diverge_mpi_write_byte_to_file(file, Sgap_charge, new_off, sdou*my_nelem);
        new_off = diverge_mpi_write_byte_to_file(file, Ugap_charge, new_off, scdou*my_nelem*linsize);
        new_off = diverge_mpi_write_byte_to_file(file, Vgap_charge, new_off, scdou*my_nelem*linsize);
    }

    if(called_suscep_full) {
        new_off = diverge_mpi_write_byte_to_file(file, suscPp,
                            new_off+my_nk_off*tu_suscP*scdou,scdou*tu_suscP*my_nk);
        new_off = diverge_mpi_write_byte_to_file(file, suscCh,
                            new_off+my_nk_off*tu_suscC*scdou,scdou*tu_suscC*my_nk);
        new_off = diverge_mpi_write_byte_to_file(file, suscMa,
                            new_off+my_nk_off*tu_suscD*scdou,scdou*tu_suscD*my_nk);

    }
    if(called_suscep_ff) {
        new_off = diverge_mpi_write_byte_to_file(file, suscPp_ff,
                            new_off+my_nk_off*tu_suscP_ff*scdou,scdou*tu_suscP_ff*my_nk);
        new_off = diverge_mpi_write_byte_to_file(file, suscCh_ff,
                            new_off+my_nk_off*tu_suscC_ff*scdou,scdou*tu_suscC_ff*my_nk);
        new_off = diverge_mpi_write_byte_to_file(file, suscMa_ff,
                            new_off+my_nk_off*tu_suscD_ff*scdou,scdou*tu_suscD_ff*my_nk);

    }

    if(vertex->use_self && save_self) {
        if(diverge_mpi_comm_rank() != 0)
            size_self = 0;
        new_off = diverge_mpi_write_byte_to_file(file, vertex->self_en,
                        new_off,scdou*size_self);

    }
    diverge_mpi_close_file(file);
}
