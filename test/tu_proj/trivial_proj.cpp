#ifndef DIVERGE_SKIP_TESTS

#include "trivial_proj.hpp"
#include "../../src/tu/vertex_details.h"

void t_P_projection(complex128_t* proj_vertex,
                    const tu_vertex_t* const vertex,
                    const diverge_model_t* const model) {
    tu_data_t* tu_data = (tu_data_t*)model -> internals -> tu_data;
    memcpy(proj_vertex, vertex->Pch, sizeof*proj_vertex * vertex->full_vert_size);

    const index_t n_orbff = tu_data->n_orbff;
    const index_t n_orb = model->n_orb;
    const index_t nk = tu_data->nk;
    const index_t my_nk = tu_data->my_nk;
    const index_t my_nk_off = tu_data->my_nk_off;
    const index_t n_bonds = tu_data->n_bonds;
    double* Rvec = (double*)calloc(model->n_tu_ff*3,sizeof(double));
    fill_Rvec(model, Rvec);



    const tu_formfactor_t* tu_ff = model->tu_ff;
    const index_t* bond_sizes = tu_data->bond_sizes;
    const index_t* bond_offsets = tu_data->bond_offsets;
    const index_t* ob_to_orbff = tu_data->ob_to_orbff;
    const double* kmesh = model->internals->kmesh;
    const complex128_t* const C = vertex->Cch;
    const complex128_t prefac = 1.0/POW2((double)nk);

    complex128_t* full_channel = (complex128_t*)malloc(nk*n_orbff*n_orbff*sizeof(complex128_t));
    memset((void*)full_channel,0,nk*n_orbff*n_orbff*sizeof(complex128_t));
    memcpy(full_channel+n_orbff*n_orbff*my_nk_off,C,n_orbff*n_orbff*my_nk*sizeof(complex128_t));
    diverge_mpi_allreduce_complex_sum_inplace(full_channel, nk*n_orbff*n_orbff);

    // C_to_P
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for(index_t qTO = 0; qTO<my_nk;++qTO)
    for(index_t o1 = 0; o1 < n_orb; ++o1)
    for(index_t o3 = 0; o3 < n_orb; ++o3) {
        for(index_t b1 = 0; b1 < bond_sizes[o1]; ++b1)
        for(index_t b3p = 0; b3p < bond_sizes[o3]; ++b3p)
        if(tu_ff[bond_offsets[o1]+b1].oto == tu_ff[bond_offsets[o3]+b3p].oto) {
            for(index_t b1p = 0; b1p < bond_sizes[o1]; ++b1p)
            for(index_t b3 = 0; b3 < bond_sizes[o3]; ++b3)
            if(tu_ff[bond_offsets[o1]+b1p].oto == tu_ff[bond_offsets[o3]+b3].oto) {
                complex128_t sum = 0.0;
                index_t qt = my_nk_off + qTO;
                for(index_t k1 = 0; k1<nk;++k1)
                for(index_t k3 = 0; k3<nk;++k3) {
                    index_t k1x,k3x,qx,qy,k1y,k3y;
                    k1x = k1/model->nk[1], k1y = k1%model->nk[1];
                    qx = qt/model->nk[1], qy = qt%model->nk[1];
                    k3x = k3/model->nk[1], k3y = k3%model->nk[1];

                    index_t x = (k1x + k3x - qx + 2*model->nk[0])%model->nk[0];
                    index_t y = (k1y + k3y - qy + 2*model->nk[1])%model->nk[1];
                    index_t tidx = y + model->nk[1]*x;

                    sum += ff(kmesh+3*k1,Rvec+(bond_offsets[o1]+b1)*3)
                            *std::conj(ff(kmesh +3*k1,Rvec+(bond_offsets[o1]+b1p)*3))*
                           ff(kmesh+3*k3,Rvec+(bond_offsets[o3]+b3p)*3)
                           *std::conj(ff(kmesh+3*k3,Rvec+(bond_offsets[o3]+b3)*3))*
                           full_channel[ob_to_orbff[b1p+n_bonds*o1]+n_orbff*(
                              ob_to_orbff[b3p+n_bonds*o3]+n_orbff*(tidx))];
                }
                proj_vertex[ob_to_orbff[b1+n_bonds*o1]+n_orbff*(
                            ob_to_orbff[b3+n_bonds*o3]+n_orbff*(qTO))] += prefac*sum;
            }
        }
    }
    const complex128_t* const D = vertex->Dch;
    memset((void*)full_channel,0,nk*n_orbff*n_orbff*sizeof(complex128_t));
    memcpy(full_channel+n_orbff*n_orbff*my_nk_off,D,n_orbff*n_orbff*my_nk*sizeof(complex128_t));
    diverge_mpi_allreduce_complex_sum_inplace(full_channel, nk*n_orbff*n_orbff);
    // D_to_P
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for(index_t qTO = 0; qTO<my_nk;++qTO)
    for(index_t o1 = 0; o1 < n_orb; ++o1)
    for(index_t o3 = 0; o3 < n_orb; ++o3)
    for(index_t b1p = 0; b1p < bond_sizes[o1]; ++b1p)
    if(tu_ff[bond_offsets[o1]+b1p].oto == o3) {
        for(index_t b3 = 0; b3 < bond_sizes[o3]; ++b3) {
            index_t o4 = tu_ff[bond_offsets[o3]+b3].oto;
            for(index_t b1 = 0; b1 < bond_sizes[o1]; ++b1)
            for(index_t b4p = 0; b4p < bond_sizes[o4]; ++b4p)
            if(tu_ff[bond_offsets[o1]+b1].oto == tu_ff[bond_offsets[o4]+b4p].oto) {
                complex128_t sum = 0.0;
                index_t qt = my_nk_off + qTO;
                for(index_t k1 = 0; k1<nk;++k1)
                for(index_t k3 = 0; k3<nk;++k3) {
                    index_t k1x,k3x,qx,qy,k1y,k3y;
                    k1x = k1/model->nk[1], k1y = k1%model->nk[1];
                    qx = qt/model->nk[1], qy = qt%model->nk[1];
                    k3x = k3/model->nk[1], k3y = k3%model->nk[1];

                    index_t uidx = (k1y - k3y + 2*model->nk[1])%model->nk[1]
                                    +model->nk[1]*((k1x - k3x + 2*model->nk[0])%model->nk[0]);
                    index_t smk3idx = (qy - k3y + 2*model->nk[1])%model->nk[1]
                                    +model->nk[1]*((qx - k3x + 2*model->nk[0])%model->nk[0]);

                    sum += ff(kmesh+3*k1,Rvec+(bond_offsets[o1]+b1)*3)
                            *std::conj(ff(kmesh+3*k3,Rvec+(bond_offsets[o3]+b3)*3))*
                           ff(kmesh+3*smk3idx,Rvec+(bond_offsets[o4]+b4p)*3)
                           *std::conj(ff(kmesh+3*k1,Rvec+(bond_offsets[o1]+b1p)*3))*
                           full_channel[ob_to_orbff[b1p+n_bonds*o1]+n_orbff*(
                              ob_to_orbff[b4p+n_bonds*o4]+n_orbff*(uidx))];
                }
                proj_vertex[ob_to_orbff[b1+n_bonds*o1]+n_orbff*(
                   ob_to_orbff[b3+n_bonds*o3]+n_orbff*(qTO))] += prefac*sum;
            }
        }
    }
    free(full_channel);
    free(Rvec);
}


void t_C_projection(complex128_t* proj_vertex,
                    const tu_vertex_t* const vertex,
                    const diverge_model_t* const model) {
    tu_data_t* tu_data = (tu_data_t*)model -> internals -> tu_data;
    memcpy(proj_vertex, vertex->Cch, sizeof*proj_vertex * vertex->full_vert_size);

    const index_t n_orbff = tu_data->n_orbff;
    const index_t n_orb = model->n_orb;
    const index_t nk = tu_data->nk;
    const index_t my_nk = tu_data->my_nk;
    const index_t my_nk_off = tu_data->my_nk_off;
    const index_t n_bonds = tu_data->n_bonds;
    double* Rvec = (double*)calloc(model->n_tu_ff*3,sizeof(double));
    fill_Rvec(model, Rvec);

    const tu_formfactor_t* tu_ff = model->tu_ff;
    const index_t* bond_sizes = tu_data->bond_sizes;
    const index_t* bond_offsets = tu_data->bond_offsets;
    const index_t* ob_to_orbff = tu_data->ob_to_orbff;
    const double* kmesh = model->internals->kmesh;
    const complex128_t* const P = vertex->Pch;
    const complex128_t prefac = 1.0/POW2((double)nk);
    complex128_t* full_channel = (complex128_t*)malloc(nk*n_orbff*n_orbff*sizeof(complex128_t));
    memset((void*)full_channel,0,nk*n_orbff*n_orbff*sizeof(complex128_t));
    memcpy(full_channel+n_orbff*n_orbff*my_nk_off,P,n_orbff*n_orbff*my_nk*sizeof(complex128_t));
    diverge_mpi_allreduce_complex_sum_inplace(full_channel, nk*n_orbff*n_orbff);

    // C_to_P
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for(index_t qTO = 0; qTO<my_nk;++qTO)
    for(index_t o1 = 0; o1 < n_orb; ++o1)
    for(index_t o3 = 0; o3 < n_orb; ++o3) {
        for(index_t b1p = 0; b1p < bond_sizes[o1]; ++b1p)
        for(index_t b3 = 0; b3 < bond_sizes[o3]; ++b3)
        if(tu_ff[bond_offsets[o1]+b1p].oto == tu_ff[bond_offsets[o3]+b3].oto) {
            for(index_t b1 = 0; b1 < bond_sizes[o1]; ++b1)
            for(index_t b3p = 0; b3p < bond_sizes[o3]; ++b3p)
            if(tu_ff[bond_offsets[o1]+b1].oto == tu_ff[bond_offsets[o3]+b3p].oto) {
                complex128_t sum = 0.0;
                index_t qt = my_nk_off + qTO;
                for(index_t k1 = 0; k1<nk;++k1)
                for(index_t k3 = 0; k3<nk;++k3) {
                    index_t k1x,k3x,qx,qy,k1y,k3y;
                    k1x = k1/model->nk[1], k1y = k1%model->nk[1];
                    qx = qt/model->nk[1], qy = qt%model->nk[1];
                    k3x = k3/model->nk[1], k3y = k3%model->nk[1];

                    index_t x = (k1x + k3x - qx + 2*model->nk[0])%model->nk[0];
                    index_t y = (k1y + k3y - qy + 2*model->nk[1])%model->nk[1];
                    index_t tidx = y + model->nk[1]*x;

                    sum += ff(kmesh+3*k1,Rvec+(bond_offsets[o1]+b1)*3)
                            *std::conj(ff(kmesh +3*k1,Rvec+(bond_offsets[o1]+b1p)*3))*
                           ff(kmesh+3*k3,Rvec+(bond_offsets[o3]+b3p)*3)
                           *std::conj(ff(kmesh+3*k3,Rvec+(bond_offsets[o3]+b3)*3))*
                           full_channel[ob_to_orbff[b1p+n_bonds*o1]+n_orbff*(
                              ob_to_orbff[b3p+n_bonds*o3]+n_orbff*(tidx))];
                }
                proj_vertex[ob_to_orbff[b1+n_bonds*o1]+n_orbff*(
                            ob_to_orbff[b3+n_bonds*o3]+n_orbff*(qTO))] += prefac*sum;
            }
        }
    }
    const complex128_t* const D = vertex->Dch;
    memset((void*)full_channel,0,nk*n_orbff*n_orbff*sizeof(complex128_t));
    memcpy(full_channel+n_orbff*n_orbff*my_nk_off,D,n_orbff*n_orbff*my_nk*sizeof(complex128_t));
    diverge_mpi_allreduce_complex_sum_inplace(full_channel, nk*n_orbff*n_orbff);
    // D_to_P
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for(index_t qTO = 0; qTO<my_nk;++qTO)
    for(index_t o1 = 0; o1 < n_orb; ++o1)
    for(index_t o3 = 0; o3 < n_orb; ++o3)
    for(index_t b1p = 0; b1p < bond_sizes[o1]; ++b1p)
    if(tu_ff[bond_offsets[o1]+b1p].oto == o3) {
        for(index_t b1 = 0; b1 < bond_sizes[o1]; ++b1) {
            index_t o4 = tu_ff[bond_offsets[o1]+b1].oto;
            for(index_t b3 = 0; b3 < bond_sizes[o3]; ++b3)
            for(index_t b4p = 0; b4p < bond_sizes[o4]; ++b4p)
            if(tu_ff[bond_offsets[o3]+b3].oto == tu_ff[bond_offsets[o4]+b4p].oto) {
                complex128_t sum = 0.0;
                index_t qt = my_nk_off + qTO;
                for(index_t k1 = 0; k1<nk;++k1)
                for(index_t k3 = 0; k3<nk;++k3) {
                    index_t k1x,k3x,qx,qy,k1y,k3y;
                    k1x = k1/model->nk[1], k1y = k1%model->nk[1];
                    qx = qt/model->nk[1], qy = qt%model->nk[1];
                    k3x = k3/model->nk[1], k3y = k3%model->nk[1];


                    index_t uidx = (k1y-k3y+2*model->nk[1])%model->nk[1]
                                    +model->nk[1]*((k1x-k3x+2*model->nk[0])%model->nk[0]);
                    index_t smk3idx = (k1y - qy + 2*model->nk[1])%model->nk[1]
                                    +model->nk[1]*((k1x - qx + 2*model->nk[0])%model->nk[0]);

                    sum += ff(kmesh+3*k1,Rvec+(bond_offsets[o1]+b1)*3)
                            *std::conj(ff(kmesh+3*k3,Rvec+(bond_offsets[o3]+b3)*3))*
                           ff(kmesh+3*smk3idx,Rvec+(bond_offsets[o4]+b4p)*3)
                           *std::conj(ff(kmesh+3*k1,Rvec+(bond_offsets[o1]+b1p)*3))*
                           full_channel[ob_to_orbff[b1p+n_bonds*o1]+n_orbff*(
                              ob_to_orbff[b4p+n_bonds*o4]+n_orbff*(uidx))];
                }
                proj_vertex[ob_to_orbff[b1+n_bonds*o1]+n_orbff*(
                   ob_to_orbff[b3+n_bonds*o3]+n_orbff*(qTO))] += prefac*sum;
            }
        }
    }
    free(full_channel);
    free(Rvec);
}


void t_D_projection(complex128_t* proj_vertex,
                    const tu_vertex_t* const vertex,
                    const diverge_model_t* const model) {
    tu_data_t* tu_data = (tu_data_t*)model -> internals -> tu_data;
    memcpy(proj_vertex, vertex->Dch, sizeof*proj_vertex * vertex->full_vert_size);

    const index_t n_orbff = tu_data->n_orbff;
    const index_t n_orb = model->n_orb;
    const index_t nk = tu_data->nk;
    const index_t my_nk = tu_data->my_nk;
    const index_t my_nk_off = tu_data->my_nk_off;
    const index_t n_bonds = tu_data->n_bonds;
    double* Rvec = (double*)calloc(model->n_tu_ff*3,sizeof(double));
    fill_Rvec(model, Rvec);


    const tu_formfactor_t* tu_ff = model->tu_ff;
    const index_t* bond_sizes = tu_data->bond_sizes;
    const index_t* bond_offsets = tu_data->bond_offsets;
    const index_t* ob_to_orbff = tu_data->ob_to_orbff;
    const double* kmesh = model->internals->kmesh;
    const complex128_t* const C = vertex->Cch;
    const complex128_t prefac = 1.0/POW2((double)nk);
    complex128_t* full_channel = (complex128_t*)malloc(nk*n_orbff*n_orbff*sizeof(complex128_t));
    memset((void*)full_channel,0,nk*n_orbff*n_orbff*sizeof(complex128_t));
    memcpy(full_channel+n_orbff*n_orbff*my_nk_off,C,n_orbff*n_orbff*my_nk*sizeof(complex128_t));
    diverge_mpi_allreduce_complex_sum_inplace(full_channel, nk*n_orbff*n_orbff);

    // C_to_D
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for(index_t qTO = 0; qTO<my_nk;++qTO)
    for(index_t o1 = 0; o1 < n_orb; ++o1)
    for(index_t o4 = 0; o4 < n_orb; ++o4)
    for(index_t b1p = 0; b1p < bond_sizes[o1]; ++b1p)
    if(tu_ff[bond_offsets[o1]+b1p].oto == o4) {
        for(index_t b1 = 0; b1 < bond_sizes[o1]; ++b1) {
            index_t o3 = tu_ff[bond_offsets[o1]+b1].oto;
            for(index_t b3p = 0; b3p < bond_sizes[o3]; ++b3p)
            for(index_t b4 = 0; b4 < bond_sizes[o4]; ++b4)
            if(tu_ff[bond_offsets[o3]+b3p].oto == tu_ff[bond_offsets[o4]+b4].oto) {
                complex128_t sum = 0.0;
                index_t qt = my_nk_off + qTO;
                for(index_t k1 = 0; k1<nk;++k1)
                for(index_t k3 = 0; k3<nk;++k3) {
                    index_t k1x,k3x,qx,qy,k1y,k3y;
                    k1x = k1/model->nk[1], k1y = k1%model->nk[1];
                    qx = qt/model->nk[1], qy = qt%model->nk[1];
                    k3x = k3/model->nk[1], k3y = k3%model->nk[1];

                    index_t uidx = (k1y-k3y + 2*model->nk[1])%model->nk[1]
                                    +model->nk[1]*((k1x-k3x + 2*model->nk[0])%model->nk[0]);
                    index_t smk3idx = (k1y-qy + 2*model->nk[1])%model->nk[1]
                                    +model->nk[1]*((k1x-qx + 2*model->nk[0])%model->nk[0]);

                    sum += ff(kmesh+3*k1,Rvec+(bond_offsets[o1]+b1)*3)
                            *std::conj(ff(kmesh +3*k3,Rvec+(bond_offsets[o4]+b4)*3))*
                           ff(kmesh+3*smk3idx,Rvec+(bond_offsets[o3]+b3p)*3)
                           *std::conj(ff(kmesh+3*k1,Rvec+(bond_offsets[o1]+b1p)*3))*
                           full_channel[ob_to_orbff[b1p+n_bonds*o1]+n_orbff*(
                              ob_to_orbff[b3p+n_bonds*o3]+n_orbff*(uidx))];
                }
                proj_vertex[ob_to_orbff[b1+n_bonds*o1]+n_orbff*(
                            ob_to_orbff[b4+n_bonds*o4]+n_orbff*(qTO))] += prefac*sum;
            }
        }
    }
    const complex128_t* const P = vertex->Pch;
    memset((void*)full_channel,0,nk*n_orbff*n_orbff*sizeof(complex128_t));
    memcpy(full_channel+n_orbff*n_orbff*my_nk_off,P,n_orbff*n_orbff*my_nk*sizeof(complex128_t));
    diverge_mpi_allreduce_complex_sum_inplace(full_channel, nk*n_orbff*n_orbff);
    // P to D
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for(index_t qTO = 0; qTO<my_nk;++qTO)
    for(index_t o1 = 0; o1 < n_orb; ++o1)
    for(index_t o4 = 0; o4 < n_orb; ++o4)
    for(index_t b1p = 0; b1p < bond_sizes[o1]; ++b1p)
    for(index_t b4 = 0; b4 < bond_sizes[o4]; ++b4)
    if(tu_ff[bond_offsets[o1]+b1p].oto == tu_ff[bond_offsets[o4]+b4].oto) {
        for(index_t b1 = 0; b1 < bond_sizes[o1]; ++b1) {
            index_t o3 = tu_ff[bond_offsets[o1]+b1].oto;
            for(index_t b3p = 0; b3p < bond_sizes[o3]; ++b3p)
            if(tu_ff[bond_offsets[o3]+b3p].oto == o4) {
                complex128_t sum = 0.0;
                index_t qt = my_nk_off + qTO;
                for(index_t k1 = 0; k1<nk;++k1)
                for(index_t k3 = 0; k3<nk;++k3) {
                    index_t k1x,k3x,qx,qy,k1y,k3y;
                    k1x = k1/model->nk[1], k1y = k1%model->nk[1];
                    qx = qt/model->nk[1], qy = qt%model->nk[1];
                    k3x = k3/model->nk[1], k3y = k3%model->nk[1];

                    index_t uidx = (k1y+k3y-qy+ 2*model->nk[1])%model->nk[1]
                        +model->nk[1]*((k1x+k3x-qx + 2*model->nk[0])%model->nk[0]);
                    index_t smk3idx = (k1y-qy + 2*model->nk[1])%model->nk[1]
                                    +model->nk[1]*((k1x-qx + 2*model->nk[0])%model->nk[0]);

                    sum += ff(kmesh+3*k1,Rvec+(bond_offsets[o1]+b1)*3)
                            *std::conj(ff(kmesh+3*k3,Rvec+(bond_offsets[o4]+b4)*3))*
                           ff(kmesh+3*smk3idx,Rvec+(bond_offsets[o3]+b3p)*3)
                           *std::conj(ff(kmesh+3*k1,Rvec+(bond_offsets[o1]+b1p)*3))*
                           full_channel[ob_to_orbff[b1p+n_bonds*o1]+n_orbff*(
                              ob_to_orbff[b3p+n_bonds*o3]+n_orbff*(uidx))];
                }
                proj_vertex[ob_to_orbff[b1+n_bonds*o1]+n_orbff*(
                   ob_to_orbff[b4+n_bonds*o4]+n_orbff*(qTO))] += prefac*sum;
            }
        }
    }
    free(full_channel);
    free(Rvec);
}

#endif // DIVERGE_SKIP_TESTS
