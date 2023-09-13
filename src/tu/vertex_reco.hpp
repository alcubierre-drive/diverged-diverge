/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "../misc/gemm.h"

template<char channel>
inline void Vertex::reco_ft(complex128_t* const buf,
                    const complex128_t * const existing_r_exp, const uint nr,
                        complex128_t * buf_q) const {
    bool was_null = false;
    if(buf_q == nullptr) {
        was_null = true;
        index_t mvsq = symm->use_symmetries ? symm->my_nk_in_bz: my_nk;
        index_t my_vertsize = mvsq*POW2(n_orbff*POW2(n_spin));
        buf_q = (complex128_t*)calloc(my_vertsize,sizeof(complex128_t));
    }
    const index_t rest = POW2(n_orbff*POW2(n_spin));
    if(symm->use_symmetries) {
        const index_t my_nk_in_bz = symm->my_nk_in_bz;
        const index_t my_nk_in_bz_off = symm->my_nk_in_bz_off;

        mfill(buf_q,rest*my_nk_in_bz,0.);
        const index_t nsq = POW2(n_spin);
        const index_t legdim = nsq*n_orbff;
        #pragma omp parallel for schedule(static) collapse(5)
        for(index_t s4 = 0; s4<(n_spin);++s4)
        for(index_t s3 = 0; s3<(n_spin);++s3)
        for(index_t mo3 = 0; mo3<n_orbff;++mo3)
        for(index_t s2 = 0; s2<(n_spin);++s2)
        for(index_t s1 = 0; s1<(n_spin);++s1)
        for(index_t mo1 = 0; mo1<n_orbff;++mo1)
        for(index_t q = 0; q<my_nk_in_bz;++q) {
            const index_t q_in_full = symm->from_ibz_2_bz[q + my_nk_in_bz_off];
            const index_t qFROM_IBZ = symm->kmaps_to[q_in_full] - tu_data->my_nk_off;
            const index_t idx_TO = q + my_nk_in_bz*(mo1 + n_orbff*((s1+n_spin*s2)+nsq*
                    (mo3 + n_orbff*(s3+n_spin*s4))));

            if constexpr(channel == 'P') {
                index_t lidx = IDX4(s2,s1,mo1,q,n_spin,n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s4,s3,mo3,q,n_spin,n_orbff,my_nk_in_bz);

                for(index_t l = 0; l < symm->o2m_map_len[lidx]; ++l)
                for(index_t r = 0; r < symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        symm->idx_map_symm[l+symm->o2m_map_off[lidx]] + legdim * (
                            symm->idx_map_symm[r+symm->o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] +=  symm->prefactor_symmP[l+symm->o2m_map_off[lidx]]*
                            std::conj(symm->prefactor_symmP[r+symm->o2m_map_off[ridx]])*
                                Pch[idx_FROM];
                }
            }
            if constexpr(channel == 'C') {
                index_t lidx = IDX4(s4,s1,mo1,q,n_spin,n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s2,s3,mo3,q,n_spin,n_orbff,my_nk_in_bz);

                for(index_t l = 0; l < symm->o2m_map_len[lidx]; ++l)
                for(index_t r = 0; r < symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        symm->idx_map_symm[l+symm->o2m_map_off[lidx]] + legdim * (
                            symm->idx_map_symm[r+symm->o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] +=  symm->prefactor_symmCD[l+symm->o2m_map_off[lidx]]*
                            std::conj(symm->prefactor_symmCD[r+symm->o2m_map_off[ridx]])*
                                Cch[idx_FROM];
                }
            }
            if constexpr(channel == 'D') {
                index_t lidx = IDX4(s3,s1,mo1,q,n_spin,n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s2,s4,mo3,q,n_spin,n_orbff,my_nk_in_bz);

                for(index_t l = 0; l < symm->o2m_map_len[lidx]; ++l)
                for(index_t r = 0; r < symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        symm->idx_map_symm[l+symm->o2m_map_off[lidx]] + legdim * (
                            symm->idx_map_symm[r+symm->o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] +=  symm->prefactor_symmCD[l+symm->o2m_map_off[lidx]]*
                            std::conj(symm->prefactor_symmCD[r+symm->o2m_map_off[ridx]])*
                                Dch[idx_FROM];
                }
            }
        }

        const double norm = 1./(double)nk;
        gemm(buf_q,existing_r_exp,buf,nr,rest,my_nk_in_bz,norm,0.0);
    }else{
        #pragma omp parallel for schedule(static) collapse(7)
        for(index_t s4 = 0; s4<(n_spin);++s4)
        for(index_t s3 = 0; s3<(n_spin);++s3)
        for(index_t mo3 = 0; mo3<n_orbff;++mo3)
        for(index_t s2 = 0; s2<(n_spin);++s2)
        for(index_t s1 = 0; s1<(n_spin);++s1)
        for(index_t mo1 = 0; mo1<n_orbff;++mo1)
        for(index_t q = 0; q<my_nk;++q) {
            const index_t idx_TO = q + my_nk*(mo1+n_orbff*(s1+n_spin*
                                (s2+n_spin*(mo3+n_orbff*(s3+n_spin*s4)))));
            if constexpr(channel == 'P') {
                buf_q[idx_TO] = Pch[mo1+n_orbff*(s1+n_spin*
                                    (s2+n_spin*(mo3+n_orbff*(s3+n_spin*
                                                        (s4+n_spin*q)))))];
            }
            if constexpr(channel == 'C') {
                buf_q[idx_TO] = Cch[mo1+n_orbff*(s1+n_spin*
                                    (s4+n_spin*(mo3+n_orbff*(s3+n_spin*
                                                        (s2+n_spin*q)))))];
            }
            if constexpr(channel == 'D') {
                buf_q[idx_TO] = Dch[mo1+n_orbff*(s1+n_spin*
                                    (s3+n_spin*(mo3+n_orbff*(s4+n_spin*
                                                        (s2+n_spin*q)))))];
            }
        }
        const double norm = 1./(double)nk;
        gemm(buf_q,existing_r_exp,buf,nr,rest,my_nk,norm,0.0);
    }
    diverge_mpi_allreduce_complex_sum_inplace(buf, nr*rest);

    if(was_null == true)
        free(buf_q);
}



template<char channel>
inline void Vertex::reco(complex128_t* const buf_q) const {
    const index_t rest = POW2(n_orbff*POW2(n_spin));
    memset((void*)buf_q,0,sizeof(complex128_t)*rest*nk);
    if(symm->use_symmetries) {

        const index_t my_nk_in_bz = symm->my_nk_in_bz;
        const index_t my_nk_in_bz_off = symm->my_nk_in_bz_off;

        const index_t nsq = POW2(n_spin);
        const index_t legdim = nsq*n_orbff;
        #pragma omp parallel for schedule(static) collapse(5)
        for(index_t s4 = 0; s4<(n_spin);++s4)
        for(index_t s3 = 0; s3<(n_spin);++s3)
        for(index_t mo3 = 0; mo3<n_orbff;++mo3)
        for(index_t s2 = 0; s2<(n_spin);++s2)
        for(index_t s1 = 0; s1<(n_spin);++s1)
        for(index_t mo1 = 0; mo1<n_orbff;++mo1)
        for(index_t q = 0; q<my_nk_in_bz;++q) {
            const index_t q_in_full = symm->from_ibz_2_bz[q + my_nk_in_bz_off];
            const index_t qFROM_IBZ = symm->kmaps_to[q_in_full] - tu_data->my_nk_off;
            const index_t idx_TO = q_in_full + nk*(mo1 + n_orbff*((s1+n_spin*s2)+nsq*
                    (mo3 + n_orbff*(s3+n_spin*s4))));

            if constexpr(channel == 'P') {
                index_t lidx = IDX4(s2,s1,mo1,q,n_spin,n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s4,s3,mo3,q,n_spin,n_orbff,my_nk_in_bz);

                for(index_t l = 0; l < symm->o2m_map_len[lidx]; ++l)
                for(index_t r = 0; r < symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        symm->idx_map_symm[l+symm->o2m_map_off[lidx]] + legdim * (
                            symm->idx_map_symm[r+symm->o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] +=  symm->prefactor_symmP[l+symm->o2m_map_off[lidx]]*
                            std::conj(symm->prefactor_symmP[r+symm->o2m_map_off[ridx]])*
                                Pch[idx_FROM];
                }
            }
            if constexpr(channel == 'C') {
                index_t lidx = IDX4(s4,s1,mo1,q,n_spin,n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s2,s3,mo3,q,n_spin,n_orbff,my_nk_in_bz);

                for(index_t l = 0; l < symm->o2m_map_len[lidx]; ++l)
                for(index_t r = 0; r < symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        symm->idx_map_symm[l+symm->o2m_map_off[lidx]] + legdim * (
                            symm->idx_map_symm[r+symm->o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] +=  symm->prefactor_symmCD[l+symm->o2m_map_off[lidx]]*
                            std::conj(symm->prefactor_symmCD[r+symm->o2m_map_off[ridx]])*
                                Cch[idx_FROM];
                }
            }
            if constexpr(channel == 'D') {
                index_t lidx = IDX4(s3,s1,mo1,q,n_spin,n_orbff,my_nk_in_bz);
                index_t ridx = IDX4(s2,s4,mo3,q,n_spin,n_orbff,my_nk_in_bz);

                for(index_t l = 0; l < symm->o2m_map_len[lidx]; ++l)
                for(index_t r = 0; r < symm->o2m_map_len[ridx]; ++r) {
                    const index_t idx_FROM =
                        symm->idx_map_symm[l+symm->o2m_map_off[lidx]] + legdim * (
                            symm->idx_map_symm[r+symm->o2m_map_off[ridx]]
                            +legdim * (qFROM_IBZ));

                    buf_q[idx_TO] +=  symm->prefactor_symmCD[l+symm->o2m_map_off[lidx]]*
                            std::conj(symm->prefactor_symmCD[r+symm->o2m_map_off[ridx]])*
                                Dch[idx_FROM];
                }
            }
        }
    }else{
        #pragma omp parallel for schedule(static) collapse(7)
        for(index_t s4 = 0; s4<(n_spin);++s4)
        for(index_t s3 = 0; s3<(n_spin);++s3)
        for(index_t mo3 = 0; mo3<n_orbff;++mo3)
        for(index_t s2 = 0; s2<(n_spin);++s2)
        for(index_t s1 = 0; s1<(n_spin);++s1)
        for(index_t mo1 = 0; mo1<n_orbff;++mo1)
        for(index_t q = 0; q<my_nk;++q) {
            const index_t idx_TO = q + my_nk_off+ nk*(mo1+n_orbff*(s1+n_spin*
                                (s2+n_spin*(mo3+n_orbff*(s3+n_spin*
                                                    s4)))));
            if constexpr(channel == 'P') {
                buf_q[idx_TO] = Pch[mo1+n_orbff*(s1+n_spin*
                                    (s2+n_spin*(mo3+n_orbff*(s3+n_spin*
                                                        (s4+n_spin*q)))))];
            }
            if constexpr(channel == 'C') {
                buf_q[idx_TO] = Cch[mo1+n_orbff*(s1+n_spin*
                                    (s4+n_spin*(mo3+n_orbff*(s3+n_spin*
                                                        (s2+n_spin*q)))))];
            }
            if constexpr(channel == 'D') {
                buf_q[idx_TO] = Dch[mo1+n_orbff*(s1+n_spin*
                                    (s3+n_spin*(mo3+n_orbff*(s4+n_spin*
                                                        (s2+n_spin*q)))))];
            }
        }
    }
    diverge_mpi_allreduce_complex_sum_inplace(buf_q, nk*rest);
}
