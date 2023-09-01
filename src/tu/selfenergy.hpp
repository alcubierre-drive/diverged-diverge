#pragma once

#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "../diverge_Eigen3.hpp"
#include "../diverge_symmetrize.h"
#include "diverge_interface.hpp"
#include "global.hpp"
#include "vertex.hpp"
#include "projection_wrapper.hpp"
#include <fftw3.h>

class selfEnergy
{
public:
    // TU constructor
    selfEnergy(Vertex* vertex);

    ~selfEnergy() {
        free(ff_helper);
        if (nk!=1) {
            free(ffkmesh);
            free(greensfkt_helper);
            fftw_destroy_plan(ifft_self_oop);
            fftw_destroy_plan(fft_self_oop);
            fftw_destroy_plan(ifft_self_ip);
            fftw_destroy_plan(fft_self_ip);

            fftw_destroy_plan(fft_fullself_ip);
            fftw_destroy_plan(ifft_fullself_ip);
            fftw_destroy_plan(fft_self_oop_gfmesh);

            fftw_destroy_plan(ifft_vert_oop);
            fftw_destroy_plan(fft_vert_oop);
            fftw_destroy_plan(ifft_vert_ip);
            fftw_destroy_plan(fft_vert_ip);
        }
    }

    //DELETE_CONSTRUCTOR_RULE_OF_FIVE(selfEnergy);

    void selfenergy_flow(complex128_t* self_en, const complex128_t* P_in,
                        const complex128_t* C_in, const complex128_t* D_in,
                        const complex128_t* GF, const double prefac_sub);

    // calculates buf[i] = a*Reco(inp)[i] + b*buf[i]
    template<char channel>
    inline void reco(complex128_t* const buf, const complex128_t* const inp,
            const double a, const double b = 1.);

    void fft_interpolate(complex128_t * const self_en);

    void selfenergy_flow_rs(complex128_t* self_en, const complex128_t* P_in,
                        const complex128_t* C_in, const complex128_t* D_in,
                        const complex128_t* GF, const double prefac_sub);
    void selfenergy_flow_rs_SU2(complex128_t* self_en, const complex128_t* P_in,
                        const complex128_t* C_in, const complex128_t* D_in,
                        const complex128_t* GF, const double prefac_sub);
    void selfenergy_flow_mom(complex128_t* self_en, const complex128_t* P_in,
                        const complex128_t* C_in, const complex128_t* D_in,
                        const complex128_t* GF, const double prefac_sub);
    void selfenergy_flow_mom_SU2(complex128_t* self_en, const complex128_t* P_in,
                        const complex128_t* C_in, const complex128_t* D_in,
                        const complex128_t* GF, const double prefac_sub);

    fftw_plan ifft_self_oop;
    fftw_plan fft_self_oop;
    fftw_plan ifft_self_ip;
    fftw_plan fft_self_ip;

    fftw_plan fft_self_oop_gfmesh;

    fftw_plan ifft_fullself_ip;
    fftw_plan fft_fullself_ip;

    fftw_plan ifft_vert_oop;
    fftw_plan fft_vert_oop;
    fftw_plan ifft_vert_ip;
    fftw_plan fft_vert_ip;

private:
    //inputs
    diverge_model_t* model;
    const tu_data_t* const tu_data;
    const index_t n_orb;
    const index_t n_spin;
    const index_t ns;
    const index_t n_frq;
    const index_t n_orbff;
    const index_t n_bonds;
    const index_t nk;
    const index_t nkf;
    const index_t nktot;
    const index_t my_nk;
    const index_t my_nk_off;
    const index_t nff;
    const bool SU2;

    complex128_t* ffkmesh;

    const index_t* ob_to_orbff;
    const vector<Vec3i>& existing_ff;

    const tu_formfactor_t* tu_ff;
    const index_t* bond_sizes;
    const index_t* bond_offsets;
    const double* bos_freq;
    const symmstruct_t* symm;

    constexpr inline index_t IBZ_2_BZ(const index_t q)
    {return symm->use_symmetries ? symm->idx_ibz_in_fullmesh[q] : q;}

    void generate_fftw_plans();

    complex128_t* reco_helper = nullptr;
    complex128_t* interpol_helper = nullptr;

    complex128_t* ff_helper;
    complex128_t* greensfkt_helper = nullptr;
};

template<char channel>
inline void selfEnergy::reco(complex128_t* const buf, const complex128_t* const inp,
                                const double a, const double b)
{
    if(std::abs(b) < 1e-10) {
        mfill_0(buf,POW2(n_orbff*n_spin*n_spin)*nk);
    }
    const index_t rest = POW2(n_orbff*POW2(n_spin));
    if(symm->use_symmetries)
    {
        const index_t my_nk_in_bz = symm->my_nk_in_bz;
        const index_t my_nk_in_bz_off = symm->my_nk_in_bz_off;

        if(abs(b-1.) > 1e-10) {
            #pragma omp parallel for schedule(static)
            for(index_t r = 0; r < rest*nk; ++r)
                buf[r] *= b;
        }

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

                    buf[idx_TO] +=  symm->prefactor_symmP[l+symm->o2m_map_off[lidx]]*
                            std::conj(symm->prefactor_symmP[r+symm->o2m_map_off[ridx]])*
                                a*inp[idx_FROM];
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

                    buf[idx_TO] +=  symm->prefactor_symmCD[l+symm->o2m_map_off[lidx]]*
                            std::conj(symm->prefactor_symmCD[r+symm->o2m_map_off[ridx]])*
                                a*inp[idx_FROM];
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

                    buf[idx_TO] += symm->prefactor_symmCD[l+symm->o2m_map_off[lidx]]*
                            std::conj(symm->prefactor_symmCD[r+symm->o2m_map_off[ridx]])*
                                a*inp[idx_FROM];
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
        for(index_t q = 0; q<my_nk;++q)
        {
            const index_t idx_TO = q + my_nk_off+ nk*(mo1+n_orbff*(s1+n_spin*
                                (s2+n_spin*(mo3+n_orbff*(s3+n_spin*
                                                    s4)))));
            if constexpr(channel == 'P') {
                buf[idx_TO] = b*buf[idx_TO]+a*inp[mo1+n_orbff*(s1+n_spin*
                                    (s2+n_spin*(mo3+n_orbff*(s3+n_spin*
                                                        (s4+n_spin*q)))))];
            }
            if constexpr(channel == 'C') {
                buf[idx_TO] = b*buf[idx_TO]+a*inp[mo1+n_orbff*(s1+n_spin*
                                    (s4+n_spin*(mo3+n_orbff*(s3+n_spin*
                                                        (s2+n_spin*q)))))];
            }
            if constexpr(channel == 'D') {
                buf[idx_TO] = b*buf[idx_TO]+a*inp[mo1+n_orbff*(s1+n_spin*
                                    (s3+n_spin*(mo3+n_orbff*(s4+n_spin*
                                                        (s2+n_spin*q)))))];
            }
        }
    }
}


