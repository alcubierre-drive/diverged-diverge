/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "vertex.hpp"
#include "projection_handler.h"

static inline complex128_t ff(const double* k, const double* r){
    return std::exp(-I128*(k[0]*r[0]+k[1]*r[1]+k[2]*r[2]));
}

static inline void fill_Rvec(const diverge_model_t* model, double* Rvec) {
    const tu_formfactor_t* tu_ff = model->tu_ff;
    for (index_t i=0; i<model->n_tu_ff; ++i)
    for (index_t d=0; d<3; ++d)
        Rvec[IDX2(i,d,3)] = tu_ff[i].R[0] * model->lattice[0][d]
                          + tu_ff[i].R[1] * model->lattice[1][d]
                          + tu_ff[i].R[2] * model->lattice[2][d];
}

void Vertex::project_fullV(const diverge_model_t* const model, complex128_t* full_V,
                            const double * frq) const {
    (void)frq;
    double* Rvec = (double*)calloc(3*model->n_tu_ff,sizeof(double));
    double* kmesh = model->internals->kmesh;
    fill_Rvec(model,Rvec);
    const index_t full_size = POW4(n_spin*n_orb)*POW3(nk);
    const tu_formfactor_t* tu_ff = model->tu_ff;
    const index_t* bond_sizes = tu_data->bond_sizes;
    const index_t* bond_offsets = tu_data->bond_offsets;
    const index_t* ob_to_orbff = tu_data->ob_to_orbff;
    const index_t n_bonds = tu_data->n_bonds;

    const index_t* nkd = model->nk;
    ZERO_OUT(full_V, full_size);
    complex128_t* buffer;
    buffer = (complex128_t*)calloc(nk*POW2(n_orbff*POW2(n_spin)),sizeof(complex128_t));

    index_t no = n_orb*n_spin;
    // unproject P
    if(P_flow) {
        reco('P', buffer);
        #pragma omp parallel for collapse(8) schedule(dynamic) num_threads(diverge_omp_num_threads())
        for(index_t s1 = 0; s1<n_spin;++s1)
        for(index_t s2 = 0; s2<n_spin;++s2)
        for(index_t s3 = 0; s3<n_spin;++s3)
        for(index_t s4 = 0; s4<n_spin;++s4)
        for(index_t o1 = 0; o1<n_orb; ++o1)
        for(index_t o3 = 0; o3<n_orb; ++o3)
        for(index_t k2 = 0; k2<nk;++k2)
        for(index_t k1 = 0; k1<nk;++k1)
        {
            index_t q_P = k1p2(k1,k2,nkd);

            for(index_t b1 = 0; b1<bond_sizes[o1];++b1)
            for(index_t b3 = 0; b3<bond_sizes[o3];++b3)
            {
                const index_t so1 = tu_ff[bond_offsets[o1]+b1].ofrom+n_orb*s1;
                const index_t so2 = tu_ff[bond_offsets[o1]+b1].oto+n_orb*s2;
                const index_t so3 = tu_ff[bond_offsets[o3]+b3].ofrom+n_orb*s3;
                const index_t so4 = tu_ff[bond_offsets[o3]+b3].oto+n_orb*s4;
                for(index_t k3 = 0; k3<nk;++k3)
                {
                    index_t idxV = IDX7(k1,k2,k3,so1,so2,so3,so4, nk,nk,no,no,no,no);
                    index_t idxP =  q_P+nk*(ob_to_orbff[b1+n_bonds*o1]
                                +n_orbff*(s1+n_spin*(s2+n_spin*(
                            ob_to_orbff[b3+n_bonds*o3]+n_orbff*(s3+n_spin*
                                s4)))));

                    full_V[idxV] += std::conj(ff(kmesh+3*k1,Rvec+3*(bond_offsets[o1]+b1)))*
                                    ff(kmesh+3*k3,Rvec+3*(bond_offsets[o3]+b3))*
                                        buffer[idxP];
                }
            }
        }
    }
    // unproject C
    if(C_flow) {
        reco('C', buffer);
        #pragma omp parallel for collapse(8) schedule(dynamic) num_threads(diverge_omp_num_threads())
        for(index_t s1 = 0; s1<n_spin;++s1)
        for(index_t s2 = 0; s2<n_spin;++s2)
        for(index_t s3 = 0; s3<n_spin;++s3)
        for(index_t s4 = 0; s4<n_spin;++s4)
        for(index_t o1 = 0; o1 < n_orb; ++o1)
        for(index_t o3 = 0; o3 < n_orb; ++o3)
        for(index_t k2 = 0; k2<nk;++k2)
        for(index_t k1 = 0; k1<nk;++k1)
        {
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1)
            for(index_t b3 = 0; b3<bond_sizes[o3];++b3)
            {
                const index_t so1 = o1+n_orb*s1;
                const index_t so2 = tu_ff[bond_offsets[o3]+b3].oto+n_orb*s2;
                const index_t so3 = o3+n_orb*s3;
                const index_t so4 = tu_ff[bond_offsets[o1]+b1].oto+n_orb*s4;

                for(index_t k3 = 0; k3<nk;++k3)
                {
                    index_t idxV = IDX7(k1,k2,k3,so1,so2,so3,so4, nk,nk,no,no,no,no);
                    index_t idxC = k1m2(k3,k2,nkd)
                        +nk*(ob_to_orbff[b1+n_bonds*o1]
                                +n_orbff*(s1+n_spin*(s2+n_spin*(
                            ob_to_orbff[b3+n_bonds*o3]+n_orbff*(s3+n_spin*
                                s4)))));

                    full_V[idxV] += std::conj(ff(kmesh+3*k1,Rvec+3*(bond_offsets[o1]+b1)))*
                                        ff(kmesh+3*k3,Rvec+3*(bond_offsets[o3]+b3))*
                                        buffer[idxC];

                }
            }
        }
    }
    // unproject D
    if(D_flow) {
        reco('D', buffer);
        #pragma omp parallel for collapse(8) schedule(dynamic) num_threads(diverge_omp_num_threads())
        for(index_t s1 = 0; s1<n_spin;++s1)
        for(index_t s2 = 0; s2<n_spin;++s2)
        for(index_t s3 = 0; s3<n_spin;++s3)
        for(index_t s4 = 0; s4<n_spin;++s4)
        for(index_t o1 = 0; o1 < n_orb; ++o1)
        for(index_t o4 = 0; o4 < n_orb; ++o4)
        for(index_t k2 = 0; k2<nk;++k2)
        for(index_t k1 = 0; k1<nk;++k1)
        {
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1)
            for(index_t b4 = 0; b4<bond_sizes[o4];++b4)
            {
                const index_t so1 = o1+n_orb*s1;
                const index_t so2 = tu_ff[bond_offsets[o4]+b4].oto+n_orb*s2;
                const index_t so3 = tu_ff[bond_offsets[o1]+b1].oto+n_orb*s3;
                const index_t so4 = o4+n_orb*s4;

                for(index_t k3 = 0; k3<nk;++k3)
                {
                    index_t idxV = IDX7(k1,k2,k3,so1,so2,so3,so4, nk,nk,no,no,no,no);
                    index_t idxD = k1m2(k1,k3,nkd)
                        +nk*(ob_to_orbff[b1+n_bonds*o1]
                            +n_orbff*(s1+n_spin*(s2+n_spin*(
                            ob_to_orbff[b4+n_bonds*o4]+n_orbff*(s3+n_spin*
                                (s4))))));

                    double uidx[3] = {kmesh[3*k1]+kmesh[3*k2]-kmesh[3*k3],
                                    kmesh[3*k1+1]+kmesh[3*k2+1]-kmesh[3*k3+1],
                                    kmesh[3*k1+2]+kmesh[3*k2+2]-kmesh[3*k3+2]};

                    full_V[idxV] +=  std::conj(ff(kmesh+3*k1,Rvec+3*(bond_offsets[o1]+b1)))*
                                        ff(uidx,Rvec+3*(bond_offsets[o4]+b4))*
                                        buffer[idxD];
                }
            }
        }
    }
    free(Rvec);
    free(buffer);
}

void Vertex::init_U_from_fullV(diverge_model_t* model, projection_t* proj) {
    mpi_wrn_printf("initialization from full \n")
    double* Rvec = (double*)calloc(3*model->n_tu_ff,sizeof(double));
    double* kmesh = model->internals->kmesh;

    const tu_formfactor_t* tu_ff = model->tu_ff;
    const index_t* bond_sizes = tu_data->bond_sizes;
    const index_t* bond_offsets = tu_data->bond_offsets;
    const index_t* ob_to_orbff = tu_data->ob_to_orbff;
    fill_Rvec(model,Rvec);

    complex128_t* buf = (complex128_t*)calloc(POW4(n_orb*n_spin), sizeof(complex128_t));
    const index_t* nkd = model->nk;
    if(D_flow) {
        ZERO_OUT(Dch, full_vert_size);
        // D-channel q = k_1 - k_3 = k_4 - k_2
        for(index_t q = 0; q < my_nk; ++q)
        for(index_t k1 = 0; k1 < nk; ++k1)
        for(index_t k4 = 0; k4 < nk; ++k4) {
            index_t k3 = k1m2(k1,symm->idx_ibz_in_fullmesh[q+my_nk_off],nkd);
            index_t k2 = k1m2(k4,symm->idx_ibz_in_fullmesh[q+my_nk_off],nkd);
            (*(model->ffill))( model, k1, k2, k3, buf );
            #pragma omp parallel for collapse(6) schedule(dynamic) num_threads(diverge_omp_num_threads())
            for (index_t s4=0; s4<n_spin; ++s4)
            for (index_t s3=0; s3<n_spin; ++s3)
            for (index_t s2=0; s2<n_spin; ++s2)
            for (index_t s1=0; s1<n_spin; ++s1)
            for(index_t o1 = 0; o1 < n_orb; ++o1)
            for(index_t o4 = 0; o4 < n_orb; ++o4)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1)
            for(index_t b4 = 0; b4<bond_sizes[o4];++b4) {
                index_t o2 = tu_ff[bond_offsets[o4]+b4].oto,
                        o3 = tu_ff[bond_offsets[o1]+b1].oto;
                index_t ob1 = ob_to_orbff[b1+n_bonds*o1],
                        ob4 = ob_to_orbff[b4+n_bonds*o4];
                index_t id_buf = IDX8(s1,o1,s2,o2,s3,o3,s4,o4,n_orb,n_spin,n_orb,n_spin,n_orb,n_spin,n_orb);
                index_t iddch = IDX7(q,s2,s4,ob4,s3,s1,ob1,n_spin,n_spin,n_orbff,n_spin,n_spin,n_orbff);
                Dch[iddch] += buf[id_buf] * ff(kmesh+3*k1,Rvec+3*(bond_offsets[o1]+b1))
                                * std::conj(ff(kmesh+3*k4,Rvec+3*(bond_offsets[o4]+b4)))/(double)POW2(nk);

            }
        }
    }

    if(C_flow) {
        ZERO_OUT(Cch, full_vert_size);
        C_projection(Cch,this,proj);
        for(index_t i = 0; i < full_vert_size; ++i) {Cch[i] *= -1.0;}
        // C-channel q = k_1 - k_4 = k_3 - k_2
        for(index_t q = 0; q < my_nk; ++q)
        for(index_t k1 = 0; k1 < nk; ++k1)
        for(index_t k3 = 0; k3 < nk; ++k3) {
            index_t k2 = k1m2(k3,symm->idx_ibz_in_fullmesh[q+my_nk_off],nkd);
            (*(model->ffill))( model, k1, k2, k3, buf );
            #pragma omp parallel for collapse(6) schedule(dynamic) num_threads(diverge_omp_num_threads())
            for (index_t s4=0; s4<n_spin; ++s4)
            for (index_t s3=0; s3<n_spin; ++s3)
            for (index_t s2=0; s2<n_spin; ++s2)
            for (index_t s1=0; s1<n_spin; ++s1)
            for(index_t o1 = 0; o1 < n_orb; ++o1)
            for(index_t o3 = 0; o3 < n_orb; ++o3)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1)
            for(index_t b3 = 0; b3<bond_sizes[o3];++b3) {
                index_t o2 = tu_ff[bond_offsets[o3]+b3].oto,
                        o4 = tu_ff[bond_offsets[o1]+b1].oto;
                index_t ob1 = ob_to_orbff[b1+n_bonds*o1],
                        ob3 = ob_to_orbff[b3+n_bonds*o3];
                index_t id_buf = IDX8(s1,o1,s2,o2,s3,o3,s4,o4,n_orb,n_spin,n_orb,n_spin,n_orb,n_spin,n_orb);
                index_t iddch = IDX7(q,s2,s3,ob3,s4,s1,ob1,n_spin,n_spin,n_orbff,n_spin,n_spin,n_orbff);
                Cch[iddch] += buf[id_buf] * ff(kmesh+3*k1,Rvec+3*(bond_offsets[o1]+b1))
                                * std::conj(ff(kmesh+3*k3,Rvec+3*(bond_offsets[o3]+b3)))/(double)POW2(nk);

            }
        }
    }


    if(P_flow) {
        ZERO_OUT(Pch, full_vert_size);
        P_projection(Pch,this,proj);
        for(index_t i = 0; i < full_vert_size; ++i) {Pch[i] *= -1.0;}
        // P-channel q = k_1 + k_2 = k_3 + k_4
        for(index_t q = 0; q < my_nk; ++q)
        for(index_t k1 = 0; k1 < nk; ++k1)
        for(index_t k3 = 0; k3 < nk; ++k3) {
            index_t k2 = k1m2(symm->idx_ibz_in_fullmesh[q+my_nk_off], k1, nkd);
            (*(model->ffill))( model, k1, k2, k3, buf );
            #pragma omp parallel for collapse(6) schedule(dynamic) num_threads(diverge_omp_num_threads())
            for (index_t s4=0; s4<n_spin; ++s4)
            for (index_t s3=0; s3<n_spin; ++s3)
            for (index_t s2=0; s2<n_spin; ++s2)
            for (index_t s1=0; s1<n_spin; ++s1)
            for(index_t o1 = 0; o1 < n_orb; ++o1)
            for(index_t o3 = 0; o3 < n_orb; ++o3)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1)
            for(index_t b3 = 0; b3<bond_sizes[o3];++b3) {
                index_t o2 = tu_ff[bond_offsets[o1]+b1].oto,
                        o4 = tu_ff[bond_offsets[o3]+b3].oto;
                index_t ob1 = ob_to_orbff[b1+n_bonds*o1],
                        ob3 = ob_to_orbff[b3+n_bonds*o3];
                index_t id_buf = IDX8(s1,o1,s2,o2,s3,o3,s4,o4,n_orb,n_spin,n_orb,n_spin,n_orb,n_spin,n_orb);
                index_t iddch = IDX7(q,s4,s3,ob3,s2,s1,ob1,n_spin,n_spin,n_orbff,n_spin,n_spin,n_orbff);
                Pch[iddch] += buf[id_buf] * ff(kmesh+3*k1,Rvec+3*(bond_offsets[o1]+b1))
                                * std::conj(ff(kmesh+3*k3,Rvec+3*(bond_offsets[o3]+b3)))/(double)POW2(nk);

            }
        }
    }
    free(Rvec);
    free(buf);
}
