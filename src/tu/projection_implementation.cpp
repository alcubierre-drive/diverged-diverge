/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "projection_implementation.hpp"

void _to_C(complex128_t* proj_vertex, const complex128_t* const I_channel,
            const vector<array<index_t, 2>>& rsmap_to,
            const vector<index_t>& rsmap_from,
            const vector<index_t>& mom,
            const vector<index_t>& rsmap_off,
            const vector<index_t>& rsmap_len,
            const complex128_t* const proj_prefac,
            const index_t realsp_size,
            const index_t num_spin,const index_t nk,
            const index_t my_nk,
            const index_t nr)
{
    const index_t mapped_indices = rsmap_to.size();
    if(nk == 1) {
        #pragma omp parallel for collapse(5) schedule(dynamic)
        for(index_t s1 = 0; s1<(num_spin);++s1)
        for(index_t s2 = 0; s2<(num_spin);++s2){
            for(index_t s3 = 0; s3<(num_spin);++s3)
            for(index_t s4 = 0; s4<(num_spin);++s4)
            for(index_t orb_map = 0; orb_map< mapped_indices; ++orb_map)
            {
                const index_t idx_FROM = rsmap_from[orb_map*2]+realsp_size*(
                    (s1+s2*num_spin)+POW2(num_spin)*(
                        rsmap_from[orb_map*2+1]+realsp_size*(
                            (s3+s4*num_spin))));

                const index_t idx_TO = rsmap_to[orb_map][0]+realsp_size*(
                (s1+s4*num_spin)+POW2(num_spin)*(
                    rsmap_to[orb_map][1]+realsp_size*(
                        (s3+s2*num_spin))));
                proj_vertex[idx_TO] += I_channel[idx_FROM];
            }
        }
    }else{
        #pragma omp parallel for collapse(6) schedule(dynamic)
        for(index_t qTO = 0; qTO<my_nk;++qTO)
        for(index_t s1 = 0; s1<(num_spin);++s1)
        for(index_t s2 = 0; s2<(num_spin);++s2)
        for(index_t s3 = 0; s3<(num_spin);++s3)
        for(index_t s4 = 0; s4<(num_spin);++s4)
        for(index_t orb_map = 0; orb_map< mapped_indices; ++orb_map){
            complex128_t total = 0.0;
            for(index_t i_idx = 0;i_idx < rsmap_len[orb_map]; i_idx+=2)
            {
                total += I_channel[mom[rsmap_off[orb_map]+i_idx]+nr*
                        (rsmap_from[rsmap_off[orb_map]+i_idx]+realsp_size*(
                            (s1+s2*num_spin)+POW2(num_spin)*(
                                rsmap_from[rsmap_off[orb_map]+i_idx+1]+realsp_size*(
                                    (s3+s4*num_spin)))))]
                        *proj_prefac[qTO+my_nk
                            *mom[rsmap_off[orb_map]+i_idx+1]];
            }

            const index_t idx_TO = rsmap_to[orb_map][0]+realsp_size*(
                    (s1+s4*num_spin)+POW2(num_spin)*(
                        rsmap_to[orb_map][1]+realsp_size*(
                            (s3+s2*num_spin)+POW2(num_spin)*
                                (qTO))));
            proj_vertex[idx_TO] += total;
        }

    }
}

void _to_P(complex128_t* proj_vertex, const complex128_t* const I_channel,
            const vector<array<index_t, 2>>& rsmap_to,
            const vector<index_t>& rsmap_from,
            const vector<index_t>& mom,
            const vector<index_t>& rsmap_off,
            const vector<index_t>& rsmap_len,
            const complex128_t* const proj_prefac,
            const index_t realsp_size,
            const index_t num_spin,const index_t nk,
            const index_t my_nk,
            const index_t nr)
{
    const index_t mapped_indices = rsmap_to.size();
    if(nk == 1) {
        #pragma omp parallel for collapse(5) schedule(dynamic)
        for(index_t s1 = 0; s1<(num_spin);++s1)
        for(index_t s2 = 0; s2<(num_spin);++s2){
            for(index_t s3 = 0; s3<(num_spin);++s3)
            for(index_t s4 = 0; s4<(num_spin);++s4)
            for(index_t orb_map = 0; orb_map<mapped_indices; ++orb_map)
            {
                const index_t idx_FROM = rsmap_from[orb_map*2]+realsp_size*(
                    (s1+s2*num_spin)+POW2(num_spin)*(
                        rsmap_from[orb_map*2+1]+realsp_size*(
                            (s3+s4*num_spin))));

                const index_t idx_TO = rsmap_to[orb_map][0]+realsp_size*(
                (s1+s2*num_spin)+POW2(num_spin)*(
                    rsmap_to[orb_map][1]+realsp_size*(
                        (s3+s4*num_spin))));
                proj_vertex[idx_TO] += I_channel[idx_FROM];
            }
        }
    }else{
        const index_t mapped_indices = rsmap_to.size();
        #pragma omp parallel for collapse(6) schedule(dynamic)
        for(index_t qTO = 0; qTO<my_nk;++qTO){
            for(index_t s1 = 0; s1<(num_spin);++s1)
            for(index_t s2 = 0; s2<(num_spin);++s2){
                for(index_t s3 = 0; s3<(num_spin);++s3)
                for(index_t s4 = 0; s4<(num_spin);++s4)
                for(index_t orb_map = 0; orb_map< mapped_indices; ++orb_map){
                    complex128_t total = 0.0;
                    for(index_t i_idx = 0;i_idx < rsmap_len[orb_map]; i_idx+=2)
                    {
                        total +=  I_channel[mom[rsmap_off[orb_map]+i_idx]+nr*
                                (rsmap_from[rsmap_off[orb_map]+i_idx]+realsp_size*(
                                    (s1+s2*num_spin)+POW2(num_spin)*(
                                        rsmap_from[rsmap_off[orb_map]+i_idx+1]+realsp_size*(
                                            (s3+s4*num_spin)))))]
                                *proj_prefac[qTO+my_nk*mom[rsmap_off[orb_map]+i_idx+1]];
                    }

                    const index_t idx_TO = rsmap_to[orb_map][0]+realsp_size*(
                            (s1+s2*num_spin)+POW2(num_spin)*
                                (rsmap_to[orb_map][1]+realsp_size*(
                                    (s3+s4*num_spin)+POW2(num_spin)*
                                        (qTO))));

                    proj_vertex[idx_TO] += total;
                }
            }
        }
    }
}

void _to_D(complex128_t* proj_vertex, const complex128_t* const I_channel,
            const vector<array<index_t, 2>>& rsmap_to,
            const vector<index_t>& rsmap_from,
            const vector<index_t>& mom,
            const vector<index_t>& rsmap_off,
            const vector<index_t>& rsmap_len,
            const complex128_t* const proj_prefac,
            const index_t realsp_size,
            const index_t num_spin,const index_t nk,
            const index_t my_nk,
            const index_t nr)
{
    const index_t mapped_indices = rsmap_to.size();
    if(nk==1) {
        #pragma omp parallel for collapse(5) schedule(dynamic)
        for(index_t s1 = 0; s1<(num_spin);++s1)
        for(index_t s2 = 0; s2<(num_spin);++s2){
            for(index_t s3 = 0; s3<(num_spin);++s3)
            for(index_t s4 = 0; s4<(num_spin);++s4)
            for(index_t orb_map = 0; orb_map< mapped_indices; ++orb_map)
            {
                const index_t idx_FROM = rsmap_from[orb_map*2]+realsp_size*(
                    (s1+s2*num_spin)+POW2(num_spin)*(
                        rsmap_from[orb_map*2+1]+realsp_size*(
                            (s3+s4*num_spin))));

                const index_t idx_TO = rsmap_to[orb_map][0]+realsp_size*(
                (s1+s3*num_spin)+POW2(num_spin)*(
                    rsmap_to[orb_map][1]+realsp_size*(
                        (s4+s2*num_spin))));
                proj_vertex[idx_TO] += I_channel[idx_FROM];
            }
        }
    }else{
        const index_t mapped_indices = rsmap_to.size();
        #pragma omp parallel for collapse(6) schedule(dynamic)
        for(index_t qTO = 0; qTO<my_nk;++qTO){
            for(index_t s1 = 0; s1<(num_spin);++s1)
            for(index_t s2 = 0; s2<(num_spin);++s2){
                for(index_t s3 = 0; s3<(num_spin);++s3)
                for(index_t s4 = 0; s4<(num_spin);++s4)
                for(index_t orb_map = 0; orb_map< mapped_indices; ++orb_map){
                    complex128_t total = 0.0;
                    for(index_t i_idx = 0;i_idx < rsmap_len[orb_map]; i_idx+=2)
                    {
                        total +=  I_channel[mom[rsmap_off[orb_map]+i_idx]+nr*
                                (rsmap_from[rsmap_off[orb_map]+i_idx]+realsp_size*(
                                    (s1+s2*num_spin)+POW2(num_spin)*(
                                        rsmap_from[rsmap_off[orb_map]+i_idx+1]+realsp_size*(
                                            (s3+s4*num_spin)))))]
                                *proj_prefac[qTO+my_nk*mom[rsmap_off[orb_map]+i_idx+1]];
                    }

                    const index_t idx_TO = rsmap_to[orb_map][0]+realsp_size*(
                            (s1+s3*num_spin)+POW2(num_spin)*
                                (rsmap_to[orb_map][1]+realsp_size*(
                                    (s4+s2*num_spin)+POW2(num_spin)*
                                        (qTO))));

                    proj_vertex[idx_TO] += total;
                }
            }
        }
    }
}

