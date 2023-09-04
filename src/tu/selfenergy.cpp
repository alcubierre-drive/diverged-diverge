#include "selfenergy.hpp"
#include <iostream>
inline static void transpose(complex128_t* inp, complex128_t* buf, index_t nx, index_t ny) {
    for(index_t x = 0; x < nx; ++x)
    for(index_t y = 0; y < ny; ++y) {
        buf[y+ny*x] = inp[x+nx*y];
    }
    memcpy(inp,buf,nx*ny*sizeof(complex128_t));
}

inline static void fftw_execute_dft_helper(const fftw_plan p, complex128_t* in,
                                                                complex128_t* out){
    fftw_execute_dft(p,reinterpret_cast<fftw_complex*>(in),
                        reinterpret_cast<fftw_complex*>(out));
}

inline static index_t rpb1mb3(const index_t* nk,
                              index_t x, index_t y, index_t z,
                              const index_t* vb1, const index_t* vb3) {
    index_t mx = (x+vb1[0]-vb3[0]+2*nk[0])%nk[0];
    index_t my = (y+vb1[1]-vb3[1]+2*nk[1])%nk[1];
    index_t mz = (z+vb1[2]-vb3[2]+2*nk[2])%nk[2];
    return mz + nk[2] *(my + nk[1] * mx);
}

inline static index_t rpb1pb3(const index_t* nk,
                              index_t x, index_t y, index_t z,
                              const index_t* vb1, const index_t* vb3) {
    index_t mx = (x+vb1[0]+vb3[0])%nk[0];
    index_t my = (y+vb1[1]+vb3[1])%nk[1];
    index_t mz = (z+vb1[2]+vb3[2])%nk[2];
    return mz + nk[2] *(my + nk[1] * mx);
}


inline static index_t rpb1(const index_t* nk, index_t x,
                           index_t y, index_t z, const index_t* vb1) {
    index_t mx = (x+vb1[0])%nk[0];
    index_t my = (y+vb1[1])%nk[1];
    index_t mz = (z+vb1[2])%nk[2];
    return mz + nk[2] *(my + nk[1] * mx);
}

void selfEnergy::fft_interpolate(complex128_t * const self_en) {

    index_t nkff[3];
    for (int i=0; i<3; ++i) nkff[i] = model->nk[i] * model->nkf[i];
    // pad with zeros
    mfill_0(ff_helper,nktot*POW2(n_orb*n_spin));
    const double pref = 1./(double)nk;

    float hx = (model->nk[0]>1) ? (float)model->nk[0]/2. : 2;
    float hy = (model->nk[1]>1) ? (float)model->nk[1]/2. : 2;
    float hz = (model->nk[2]>1) ? (float)model->nk[2]/2. : 2;
    for(index_t r = 0; r < POW2(n_orb*n_spin);++r)
    for(index_t x = 0; x < model->nk[0]; ++x)
    for(index_t y = 0; y < model->nk[1]; ++y)
    for(index_t z = 0; z < model->nk[2]; ++z) {
        index_t from = z + model->nk[2]*(y+model->nk[1]*(x+model->nk[0]*r));
        const index_t xto = (x <= hx) ? x : nkff[0]-(model->nk[0]-x);
        const index_t yto = (y <= hy) ? y : nkff[1]-(model->nk[1]-y);
        const index_t zto = (z <= hz) ? z : nkff[2]-(model->nk[2]-z);
        const index_t to = zto + nkff[2]*(yto+nkff[1]*(xto+nkff[0]*r));
        complex128_t helper = self_en[from]*pref;
        // copy edges and corners
        if(std::abs(x - hx) < 1e-10 && std::abs(y - hy) < 1e-10 && std::abs(z - hz) < 1e-10) {
            helper *= 0.125;
            index_t xto2 = nkff[0]-hx;
            index_t yto2 = nkff[1]-hy;
            index_t zto2 = nkff[2]-hz;
            ff_helper[zto2 + nkff[2]*(yto+nkff[1]*(xto+nkff[0]*r))] += helper;
            ff_helper[zto2 + nkff[2]*(yto2+nkff[1]*(xto+nkff[0]*r))] += helper;
            ff_helper[zto + nkff[2]*(yto2+nkff[1]*(xto+nkff[0]*r))] += helper;
            ff_helper[zto2 + nkff[2]*(yto2+nkff[1]*(xto2+nkff[0]*r))] += helper;
            ff_helper[zto + nkff[2]*(yto2+nkff[1]*(xto2+nkff[0]*r))] += helper;
            ff_helper[zto2 + nkff[2]*(yto+nkff[1]*(xto2+nkff[0]*r))] += helper;
            ff_helper[zto + nkff[2]*(yto+nkff[1]*(xto2+nkff[0]*r))] += helper;
        }
        else if(std::abs(x - hx) < 1e-10 && std::abs(y - hy) < 1e-10) {
            helper *= 0.25;
            index_t xto2 = nkff[0]-hx;
            index_t yto2 = nkff[1]-hy;
            ff_helper[zto + nkff[2]*(yto+nkff[1]*(xto2+nkff[0]*r))] += helper;
            ff_helper[zto + nkff[2]*(yto2+nkff[1]*(xto2+nkff[0]*r))] += helper;
            ff_helper[zto + nkff[2]*(yto2+nkff[1]*(xto+nkff[0]*r))] += helper;
        }
        else if(std::abs(x - hx) < 1e-10 && std::abs(z - hz) < 1e-10) {
            helper *= 0.25;
            index_t xto2 = nkff[0]-hx;
            index_t zto2 = nkff[2]-hz;
            ff_helper[zto2 + nkff[2]*(yto+nkff[1]*(xto +nkff[0]*r))] += helper;
            ff_helper[zto2 + nkff[2]*(yto+nkff[1]*(xto2+nkff[0]*r))] += helper;
            ff_helper[zto  + nkff[2]*(yto+nkff[1]*(xto2+nkff[0]*r))] += helper;
        }
        else if(std::abs(y - hy) < 1e-10 && std::abs(z - hz) < 1e-10) {
            helper *= 0.25;
            index_t yto2 = nkff[1]-hy;
            index_t zto2 = nkff[2]-hz;
            ff_helper[zto  + nkff[2]*(yto2+nkff[1]*(xto+nkff[0]*r))] += helper;
            ff_helper[zto2 + nkff[2]*(yto2+nkff[1]*(xto+nkff[0]*r))] += helper;
            ff_helper[zto2 + nkff[2]*(yto +nkff[1]*(xto+nkff[0]*r))] += helper;
        }else if(std::abs(x - hx) < 1e-10 ) {
            helper *= 0.5;
            index_t xto2 = nkff[0]-hx;
            ff_helper[zto + nkff[2]*(yto+nkff[1]*(xto2+nkff[0]*r))] += (helper);
        }
        else if(std::abs(z - hz) < 1e-10) {
            helper *= 0.5;
            index_t zto2 = nkff[2]-hz;
            ff_helper[zto2 + nkff[2]*(yto+nkff[1]*(xto+nkff[0]*r))] += (helper);
        }
        else if(std::abs(y - hy) < 1e-10) {
            helper *= 0.5;
            index_t yto2 = nkff[1]-hy;
            ff_helper[zto + nkff[2]*(yto2+nkff[1]*(xto+nkff[0]*r))] += (helper);
        }
        ff_helper[to] += helper;

    }
    fftw_execute_dft_helper(fft_self_oop_gfmesh,ff_helper,self_en);
}

void selfEnergy::selfenergy_flow(complex128_t* self_en,
            const complex128_t* P_in,const complex128_t* C_in, const complex128_t* D_in,
            const complex128_t* GF, const double prefac_sub) {
    if(nk == 1 && !SU2) {
        return selfenergy_flow_rs(self_en, P_in, C_in, D_in,
                        GF, prefac_sub);
    }else if(nk == 1 && SU2) {
        return selfenergy_flow_rs_SU2(self_en, P_in, C_in, D_in,
                        GF, prefac_sub);
    }else if(nk != 1 && !SU2) {
        return selfenergy_flow_mom(self_en, P_in, C_in, D_in,
                        GF, prefac_sub);
    }else{
        return selfenergy_flow_mom_SU2(self_en, P_in, C_in, D_in,
                        GF, prefac_sub);
    }
}

void selfEnergy::selfenergy_flow_rs(complex128_t* self_en,
            const complex128_t* P_in,const complex128_t* C_in, const complex128_t* D_in,
            const complex128_t* GF, const double pref_subst)
{
    // selfenergy flow equation
    const index_t blocksize_self = POW2(n_orb*n_spin);
    const complex128_t prefac = -pref_subst/(2.0*M_PI);
    //set all elements to zero

    mfill(self_en, blocksize_self*n_frq, 0.0);

    // D-channel -> real part as D is freq. independent =>
    // same for all frequencies
    // again catch the zero frequency independently
    if(D_flow) {
        #pragma omp parallel for collapse(3)
        for(index_t o1 = 0; o1<n_orb;++o1)
        for(index_t s1 = 0; s1< n_spin; ++s1)
        for(index_t s3 = 0; s3< n_spin; ++s3)
        for(index_t b1 = 0; b1<bond_sizes[o1];++b1)
        {
            complex128_t tmp = 0.0;
            for(index_t s2 = 0; s2< (n_spin); ++s2)
            for(index_t s4 = 0; s4< (n_spin); ++s4)
            for(index_t o4 = 0; o4<n_orb;++o4)
            for(index_t b4 = 0; b4<bond_sizes[o4];++b4)
            {
                tmp += (std::conj(GF[tu_ff[bond_offsets[o4]+b4].oto+n_orb*
                                (s2+n_spin*(o4 + n_orb*s4))])
                        + GF[o4+n_orb*(s4+n_spin*
                            (tu_ff[bond_offsets[o4]+b4].oto + n_orb*s2))]
                        ) * D_in[ob_to_orbff[b1+n_bonds*o1]+n_orbff*(s1+n_spin
                                        *(s3+n_spin*(ob_to_orbff[b4+n_bonds*o4]+n_orbff
                                            *(s4+n_spin*s2))))];
            }
            self_en[o1+n_orb*(s1+n_spin*
                    (tu_ff[bond_offsets[o1]+b1].oto+n_orb*(s3)))] -= (prefac*tmp);
        }
    }
    if(P_flow) {
        #pragma omp parallel for collapse(4)
        for(index_t s3 = 0; s3< n_spin; ++s3)
        for(index_t o3 = 0; o3<n_orb;++o3)
        for(index_t s1 = 0; s1< n_spin; ++s1)
        for(index_t o1 = 0; o1<n_orb;++o1)
        {
            complex128_t tmp = 0.0;
            for(index_t s4 = 0; s4< n_spin; ++s4)
            for(index_t s2 = 0; s2< n_spin; ++s2)
            for(index_t b3 = 0; b3<bond_sizes[o3];++b3)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1)
            {
                // +
                const index_t idxGF_P_p = tu_ff[bond_offsets[o3]+b3].oto + n_orb
                            *(s4+n_spin*(tu_ff[bond_offsets[o1]+b1].oto + n_orb*s2));
                const index_t idx_P_p = ob_to_orbff[b1+n_bonds*o1]+n_orbff
                            *(s1+n_spin*(s2+n_spin*(ob_to_orbff[b3+n_bonds*o3]+n_orbff*(s3+n_spin*(s4)))));

                tmp += GF[idxGF_P_p]*P_in[idx_P_p];
                // -
                const index_t idxGF_P_m = tu_ff[bond_offsets[o1]+b1].oto+n_orb
                            *(s2+n_spin*(tu_ff[bond_offsets[o3]+b3].oto + n_orb*s4));
                const index_t idx_P_m = ob_to_orbff[b3+n_bonds*o3]+n_orbff*(s3+n_spin
                            *(s4+n_spin*(ob_to_orbff[b1+n_bonds*o1]+n_orbff
                                *(s1+n_spin*(s2)))));

                tmp += std::conj(GF[idxGF_P_m]*P_in[idx_P_m]);
            }
            self_en[o1+n_orb*(s1+n_spin*(o3+n_orb*(s3)))] -= prefac*(tmp);
        }
    }
    if(C_flow) {
        #pragma omp parallel for collapse(4)
        for(index_t s3 = 0; s3< n_spin; ++s3)
        for(index_t o3 = 0; o3<n_orb;++o3)
        for(index_t s1 = 0; s1< n_spin; ++s1)
        for(index_t o1 = 0; o1<n_orb;++o1)
        {
            complex128_t tmp = 0.0;
            for(index_t s4 = 0; s4< n_spin; ++s4)
            for(index_t s2 = 0; s2< n_spin; ++s2)
            for(index_t b3 = 0; b3<bond_sizes[o3];++b3)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1)
            {
                // +
                const index_t idxGF_C_p = tu_ff[bond_offsets[o1]+b1].oto+n_orb
                            *(s4+n_spin*(tu_ff[bond_offsets[o3]+b3].oto + n_orb*s2));
                const index_t idx_C_p = ob_to_orbff[b1+n_bonds*o1]+n_orbff
                            *(s1+n_spin*(s4+n_spin*(ob_to_orbff[b3+n_bonds*o3]+n_orbff
                                *(s3+n_spin*(s2)))));

                tmp += GF[idxGF_C_p]*std::conj(C_in[idx_C_p]);

                // -
                const index_t idxGF_C_m = tu_ff[bond_offsets[o3]+b3].oto+n_orb
                            *(s2+n_spin*(tu_ff[bond_offsets[o1]+b1].oto + n_orb*s4));
                const index_t idx_C_m = ob_to_orbff[b3+n_bonds*o3]+n_orbff*(s3+n_spin
                            *(s2+n_spin*(ob_to_orbff[b1+n_bonds*o1]+n_orbff
                                *(s1+n_spin*(s4)))));

                tmp += std::conj(GF[idxGF_C_m])*C_in[idx_C_m];
            }
            self_en[o1+n_orb*(s1+n_spin*(o3+n_orb*(s3)))] -=prefac*(tmp);
        }
    }
}


void selfEnergy::selfenergy_flow_rs_SU2(complex128_t* self_en,
            const complex128_t* P_in,const complex128_t* C_in, const complex128_t* D_in,
            const complex128_t* GF, const double pref_subst)
{
    // selfenergy flow equation
    const index_t blocksize_self = POW2(n_orb*n_spin);
    const double prefac = -pref_subst/(2.0*M_PI);
    //set all elements to zero
    mfill(self_en, blocksize_self, 0.0);

    if(D_flow) {
        #pragma omp parallel for
        for(index_t o1 = 0; o1<n_orb;++o1)
        for(index_t b1 = 0; b1<bond_sizes[o1];++b1) {
            const index_t o3 = tu_ff[bond_offsets[o1]+b1].oto;
            complex128_t tmp = 0.0;
            for(index_t o4 = 0; o4<n_orb;++o4)
            for(index_t b4 = 0; b4<bond_sizes[o4];++b4) {
                tmp += std::real(GF[o4+n_orb*tu_ff[bond_offsets[o4]+b4].oto]*
                            D_in[ob_to_orbff[b1+n_bonds*o1]+n_orbff*ob_to_orbff[b4+n_bonds*o4]]);

            }
            self_en[o1+n_orb*o3] = -4.0*(prefac*tmp);
        }
        #pragma omp parallel for collapse(2)
        for(index_t o3 = 0; o3<n_orb;++o3)
        for(index_t o1 = 0; o1<n_orb;++o1){
            complex128_t tmp = 0.0;
            for(index_t b3 = 0; b3<bond_sizes[o3];++b3)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1){

                const index_t idx_GF = tu_ff[bond_offsets[o1]+b1].oto
                                    +n_orb*tu_ff[bond_offsets[o3]+b3].oto;
                const index_t idx_CD = ob_to_orbff[b3+n_bonds*o3]
                                        +n_orbff*ob_to_orbff[b1+n_bonds*o1];

                tmp -= std::real(GF[idx_GF]*D_in[idx_CD]);
            }
            self_en[o1+n_orb*o3] -= prefac*2.0*(tmp);
        }
    }

    if(C_flow) {
        #pragma omp parallel for
        for(index_t o1 = 0; o1<n_orb;++o1)
        for(index_t b1 = 0; b1<bond_sizes[o1];++b1) {
            const index_t o3 = tu_ff[bond_offsets[o1]+b1].oto;
            complex128_t tmp = 0.0;
            for(index_t o4 = 0; o4<n_orb;++o4)
            for(index_t b4 = 0; b4<bond_sizes[o4];++b4) {
                tmp -= std::real(GF[o4+n_orb*tu_ff[bond_offsets[o4]+b4].oto]*
                     C_in[ob_to_orbff[b1+n_bonds*o1]+n_orbff*ob_to_orbff[b4+n_bonds*o4]]);

            }
            self_en[o1+n_orb*o3] -= 2.0*(prefac*tmp);
        }
        #pragma omp parallel for collapse(2)
        for(index_t o3 = 0; o3<n_orb;++o3)
        for(index_t o1 = 0; o1<n_orb;++o1){
            complex128_t tmp = 0.0;
            for(index_t b3 = 0; b3<bond_sizes[o3];++b3)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1){

                const index_t idx_GF = tu_ff[bond_offsets[o1]+b1].oto
                                    +n_orb*tu_ff[bond_offsets[o3]+b3].oto;
                const index_t idx_CD = ob_to_orbff[b3+n_bonds*o3]
                                        +n_orbff*ob_to_orbff[b1+n_bonds*o1];

                tmp+= std::real(GF[idx_GF]*(2.0*C_in[idx_CD]));
            }
            self_en[o1+n_orb*o3] -= prefac*2.0*(tmp);
        }
    }

    if(P_flow) {
        #pragma omp parallel for collapse(2)
        for(index_t o3 = 0; o3<n_orb;++o3)
        for(index_t o1 = 0; o1<n_orb;++o1){
            complex128_t tmp = 0.0;
            for(index_t b3 = 0; b3<bond_sizes[o3];++b3)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1){

                const index_t idx_GF = tu_ff[bond_offsets[o1]+b1].oto
                                    +n_orb*tu_ff[bond_offsets[o3]+b3].oto;
                const index_t idx_P = ob_to_orbff[b1+n_bonds*o1]+n_orbff*ob_to_orbff[b3+n_bonds*o3];

                tmp+= std::real(GF[idx_GF]*
                        (2.0*P_in[idx_P]));
            }
            self_en[o1+n_orb*o3] -= prefac*2.0*(tmp);
        }
        #pragma omp parallel for
        for(index_t o1 = 0; o1<n_orb;++o1)
        for(index_t b1 = 0; b1<bond_sizes[o1];++b1){
            for(index_t o4 = 0; o4<n_orb;++o4)
            for(index_t b4 = 0; b4<bond_sizes[o4];++b4){
                const index_t o3 = tu_ff[bond_offsets[o4]+b4].oto;
                self_en[o1+n_orb*o3] += 2.0*prefac
                                *std::real(GF[tu_ff[bond_offsets[o1]+b1].oto+n_orb*o4]
                                *P_in[ob_to_orbff[b4+n_bonds*o4]
                                                    +n_orbff*ob_to_orbff[b1+n_bonds*o1]]);
            }
        }
    }
}


void selfEnergy::selfenergy_flow_mom(complex128_t* const self_en,
            const complex128_t* const P_in,const complex128_t* const C_in, const complex128_t* const D_in,
            const complex128_t* const  GF, const double pref_subst)
{
    // selfenergy flow equation
    const index_t blocksize_self = POW2(n_orb*n_spin)*nktot;
    mfill(self_en, blocksize_self*n_frq, 0.0);
    mfill(ff_helper, blocksize_self*n_frq, 0.0);
    mfill_0(interpol_helper,n_orbff*POW2(n_spin));
    if(diverge_mpi_comm_rank() == 0) {
        if(D_flow) {
            #pragma omp parallel for collapse(3)
            for(index_t s2 = 0; s2< (n_spin); ++s2)
            for(index_t s4 = 0; s4< (n_spin); ++s4)
            for(index_t o4 = 0; o4<n_orb;++o4)
            for(index_t b4 = 0; b4<bond_sizes[o4];++b4){
                complex128_t tmp2 = 0.0;
                for(index_t k = 0; k<nk;++k)
                {
                    tmp2 += ffkmesh[k+nk*tu_ff[bond_offsets[o4]+b4].ffidx]*
                        ((GF[k+nk*(o4+n_orb*(s4+n_spin
                                *(tu_ff[bond_offsets[o4]+b4].oto + n_orb*(s2))))]) +
                        std::conj(GF[k+nk*(tu_ff[bond_offsets[o4]+b4].oto + n_orb
                                *(s2+n_spin*(o4+n_orb*(s4))))]));
                }
                interpol_helper[bond_offsets[o4]+b4+n_orbff*(s4+n_spin*s2)] += tmp2;
            }

            #pragma omp parallel for collapse(4)
            for(index_t q = 0; q<nk;++q)
            for(index_t o1 = 0; o1<n_orb;++o1)
            for(index_t s1 = 0; s1< n_spin; ++s1)
            for(index_t s3 = 0; s3< n_spin; ++s3)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1){
                complex128_t tmp = 0.0;
                for(index_t s2 = 0; s2< (n_spin); ++s2)
                for(index_t s4 = 0; s4< (n_spin); ++s4)
                for(index_t o4 = 0; o4<n_orb;++o4)
                for(index_t b4 = 0; b4<bond_sizes[o4];++b4){
                    tmp += interpol_helper[bond_offsets[o4]+b4+n_orbff*(s4+n_spin*s2)]
                        *D_in[ob_to_orbff[b1+n_bonds*o1]
                                +n_orbff*(s1+n_spin*(s3
                                    +n_spin*(ob_to_orbff[b4+n_bonds*o4]
                                        +n_orbff*(s4+n_spin*(s2)))))];
                }
                ff_helper[q+nk*(o1+n_orb*(s1+n_spin*(tu_ff[bond_offsets[o1]+b1].oto+n_orb*s3)))]
                        -= std::conj(ffkmesh[q+nk*tu_ff[bond_offsets[o1]+b1].ffidx])*tmp;
            }
            fftw_execute_dft_helper(ifft_self_oop,ff_helper,self_en);
        }
    }
    if(P_flow) {
        reco<'P'>(reco_helper,P_in,1.0,0.0);

        mcopy(greensfkt_helper,GF,POW2(n_spin*n_orb)*nk);
        fftw_execute_dft_helper(ifft_vert_oop,reco_helper,interpol_helper);
        fftw_execute_dft_helper(fft_self_oop,greensfkt_helper,ff_helper);

        #pragma omp parallel for schedule(static)
        for(index_t i = 0; i < POW2(n_orbff)*POW4(n_spin)*nk; ++i)
            reco_helper[i] = std::conj(reco_helper[i]);
        #pragma omp parallel for schedule(static)
        for(index_t i = 0; i < POW2(n_spin*n_orb)*nk; ++i)
            greensfkt_helper[i] = std::conj(greensfkt_helper[i]);

        fftw_execute_dft_helper(ifft_vert_ip,reco_helper,reco_helper);
        fftw_execute_dft_helper(fft_self_ip,greensfkt_helper,greensfkt_helper);

        #pragma omp parallel for collapse(7)
        for(index_t z = 0; z < model->nk[2]; ++z)
        for(index_t y = 0; y < model->nk[1]; ++y)
        for(index_t x = 0; x < model->nk[0]; ++x)
        for(index_t o3 = 0; o3<n_orb;++o3)
        for(index_t s3 = 0; s3< n_spin; ++s3)
        for(index_t s1 = 0; s1< n_spin; ++s1)
        for(index_t o1 = 0; o1<n_orb;++o1){
            index_t ridx = z + model->nk[2]*(y+model->nk[1]*x);
            complex128_t tmp = 0.0;
            for(index_t b3 = 0; b3<bond_sizes[o3];++b3)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1){
                index_t rb1b3 = rpb1mb3(model->nk,x,y,z,tu_ff[bond_offsets[o1]+b1].R,
                                                        tu_ff[bond_offsets[o3]+b3].R);
                for(index_t s4 = 0; s4< n_spin; ++s4)
                for(index_t s2 = 0; s2< n_spin; ++s2)
                {
                    const index_t idx_GFP_p = rb1b3+nk*(tu_ff[bond_offsets[o3]+b3].oto+n_orb*
                            (s4+n_spin*(tu_ff[bond_offsets[o1]+b1].oto + n_orb*(s2))));

                    const index_t idx_GFP_m = rb1b3+nk*(tu_ff[bond_offsets[o1]+b1].oto+n_orb*
                            (s2+n_spin*(tu_ff[bond_offsets[o3]+b3].oto + n_orb*(s4))));

                    const index_t mi1 =  ob_to_orbff[b1+n_bonds*o1];
                    const index_t mi2 =  ob_to_orbff[b3+n_bonds*o3];

                    const complex128_t idx_P_p = interpol_helper[IDX5(s3+n_spin*s4,mi2,s1+n_spin*s2,
                                                            mi1,rb1b3,n_orbff,POW2(n_spin),n_orbff,nk)];

                    const complex128_t idx_P_m = reco_helper[IDX5(s1+n_spin*s2,mi1,s3+n_spin*s4,
                                                            mi2,rb1b3,n_orbff,POW2(n_spin),n_orbff,nk)];

                    tmp += (ff_helper[idx_GFP_p]*idx_P_p
                            +(greensfkt_helper[idx_GFP_m]*idx_P_m));
                }
            }
            self_en[ridx+nk*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))] -= (tmp);
        }
    }

    if(C_flow) {
        reco<'C'>(reco_helper,C_in,1.0,0.0);

        mcopy(greensfkt_helper,GF,POW2(n_spin*n_orb)*nk);
        fftw_execute_dft_helper(ifft_vert_oop,reco_helper,interpol_helper);
        fftw_execute_dft_helper(ifft_self_oop,greensfkt_helper,ff_helper);

        #pragma omp parallel for schedule(static)
        for(index_t i = 0; i < POW2(n_orbff)*POW4(n_spin)*nk; ++i)
            reco_helper[i] = std::conj(reco_helper[i]);
        #pragma omp parallel for schedule(static)
        for(index_t i = 0; i < POW2(n_spin*n_orb)*nk; ++i)
            greensfkt_helper[i] = std::conj(greensfkt_helper[i]);

        fftw_execute_dft_helper(ifft_vert_ip,reco_helper,reco_helper);
        fftw_execute_dft_helper(ifft_self_ip,greensfkt_helper,greensfkt_helper);

        // P and C channel have a similar form -> compress into one
        #pragma omp parallel for collapse(7)
        for(index_t z = 0; z < model->nk[2]; ++z)
        for(index_t y = 0; y < model->nk[1]; ++y)
        for(index_t x = 0; x < model->nk[0]; ++x)
        for(index_t o3 = 0; o3<n_orb;++o3)
        for(index_t s3 = 0; s3< n_spin; ++s3)
        for(index_t s1 = 0; s1< n_spin; ++s1)
        for(index_t o1 = 0; o1<n_orb;++o1){
            index_t ridx = z + model->nk[2]*(y+model->nk[1]*x);
            complex128_t tmp = 0.0;
            for(index_t b3 = 0; b3<bond_sizes[o3];++b3)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1)
            for(index_t s4 = 0; s4< n_spin; ++s4)
            for(index_t s2 = 0; s2< n_spin; ++s2)
            {
                index_t rb1b3 = rpb1mb3(model->nk,x,y,z,tu_ff[bond_offsets[o1]+b1].R,
                                                        tu_ff[bond_offsets[o3]+b3].R);
                const index_t idx_GFC_m = rb1b3+nk*(tu_ff[bond_offsets[o1]+b1].oto+n_orb*
                        (s4+n_spin*(tu_ff[bond_offsets[o3]+b3].oto + n_orb*s2)));

                const index_t idx_GFC_p = rb1b3+nk*(tu_ff[bond_offsets[o3]+b3].oto+n_orb*
                        (s2+n_spin*(tu_ff[bond_offsets[o1]+b1].oto + n_orb*s4)));

                const index_t mi1 =  ob_to_orbff[b1+n_bonds*o1];
                const index_t mi2 =  ob_to_orbff[b3+n_bonds*o3];
                const complex128_t idx_C_p = interpol_helper[IDX5(s3+n_spin*s4,mi2,s1+n_spin*s2,
                                                        mi1,rb1b3, n_orbff,POW2(n_spin),n_orbff,nk)];

                const complex128_t idx_C_m = reco_helper[IDX5(s1+n_spin*s2,mi1,s3+n_spin*s4,mi2,rb1b3,
                                                                                n_orbff,POW2(n_spin),n_orbff,nk)];

                tmp+= greensfkt_helper[idx_GFC_p]*idx_C_p
                        +ff_helper[idx_GFC_m]*idx_C_m;
            }
            self_en[ridx+nk*(o1+n_orb*(s1+n_spin*(o3+n_orb*s3)))] -= tmp;
        }
    }


    const index_t selfsize = POW2(n_orb*n_spin)*nk*n_frq;
    diverge_mpi_allreduce_complex_sum_inplace(self_en, selfsize);
    if(nk != nktot)
    {
        const double prefac = -pref_subst/(2.0*M_PI*(double)nk*(double)nk);
        for(index_t i = 0; i < POW2(n_spin*n_orb)*nk*n_frq; ++i)
            self_en[i] *= prefac;
        fftw_execute_dft_helper(fft_fullself_ip,self_en,self_en);
        transpose(self_en,ff_helper, nk, POW2(n_orb*n_spin));
        diverge_symmetrize_2pt_coarse( model, self_en, ff_helper );
        transpose(self_en,ff_helper, POW2(n_orb*n_spin), nk);
        fftw_execute_dft_helper(ifft_fullself_ip,self_en,self_en);
        fft_interpolate(self_en);
    }else{
        const double prefac = -pref_subst/(2.0*M_PI*(double)nk*(double)nk);
        for(index_t i = 0; i < POW2(n_spin*n_orb)*nk*n_frq; ++i)
            self_en[i] *= prefac;

        fftw_execute_dft_helper(fft_fullself_ip,self_en,self_en);
    }
    transpose(self_en,ff_helper, nktot, POW2(n_orb*n_spin));
    diverge_symmetrize_2pt_fine( model, self_en, ff_helper );
}


void selfEnergy::selfenergy_flow_mom_SU2(complex128_t* self_en,
            const complex128_t* P_in,const complex128_t* C_in, const complex128_t* D_in,
            const complex128_t* GF, const double pref_subst)
{
    // selfenergy flow equation
    const index_t blocksize_self = nktot*POW2(n_spin*n_orb);
    //set all elements to zero
    mfill(self_en, blocksize_self*n_frq, 0.0);
    mfill(ff_helper, blocksize_self, 0.0);


    // D-channel -> real part as D is freq. independent => same for all frequencies
    // again catch the zero frequency independently
    mfill_0(interpol_helper,n_orbff*POW2(n_spin));
    if(diverge_mpi_comm_rank() == 0) {
        if(D_flow || C_flow ) {
            #pragma omp parallel for schedule(dynamic)
            for(index_t o4 = 0; o4<n_orb;++o4)
            for(index_t b4 = 0; b4<bond_sizes[o4];++b4)
            {
                complex128_t tmp = 0.0;
                for(index_t k = 0; k<nk;++k){
                    const complex128_t GF_prefac = (GF[k+nk*(o4+n_orb*tu_ff[bond_offsets[o4]+b4].oto)])
                                            +std::conj(GF[k+nk*(tu_ff[bond_offsets[o4]+b4].oto+n_orb*o4)]);
                    tmp += GF_prefac*ffkmesh[k+nk*tu_ff[bond_offsets[o4]+b4].ffidx];
                }
                interpol_helper[bond_offsets[o4]+b4] += tmp;
            }
        }
        if(D_flow) {
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for(index_t q = 0; q<nk;++q)
            for(index_t o1 = 0; o1<n_orb;++o1)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1){
                complex128_t tmp = 0.0;
                for(index_t o4 = 0; o4<n_orb;++o4)
                for(index_t b4 = 0; b4<bond_sizes[o4];++b4){
                    const index_t mi1 = ob_to_orbff[b1+n_bonds*o1];
                    const index_t mi2 = ob_to_orbff[b4+n_bonds*o4];
                    tmp += interpol_helper[bond_offsets[o4]+b4] * 2.0 * D_in[mi1+n_orbff*(mi2)];
                }
                ff_helper[q+nk*(o1+n_orb*tu_ff[bond_offsets[o1]+b1].oto)] -=
                                (tmp)*std::conj(ffkmesh[q+nk*tu_ff[bond_offsets[o1]+b1].ffidx]);
            }
        }
        if(C_flow) {
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for(index_t q = 0; q<nk;++q)
            for(index_t o1 = 0; o1<n_orb;++o1)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1){
                complex128_t tmp = 0.0;
                for(index_t o4 = 0; o4<n_orb;++o4)
                for(index_t b4 = 0; b4<bond_sizes[o4];++b4){
                    const index_t mi1 = ob_to_orbff[b1+n_bonds*o1];
                    const index_t mi2 = ob_to_orbff[b4+n_bonds*o4];
                    tmp -= interpol_helper[bond_offsets[o4]+b4] * C_in[mi1+n_orbff*(mi2)];
                }
                ff_helper[q+nk*(o1+n_orb*tu_ff[bond_offsets[o1]+b1].oto)] -=
                                (tmp)*std::conj(ffkmesh[q+nk*tu_ff[bond_offsets[o1]+b1].ffidx]);
            }
        }
        if(C_flow || D_flow)
            fftw_execute_dft_helper(ifft_self_oop,ff_helper,self_en);
    }

    if(C_flow)
        reco<'C'>(reco_helper,C_in,1.0,0.0);

    if(D_flow) {
        if(!C_flow) {
            reco<'D'>(reco_helper,D_in,-0.5,0.0);
        }else{
            reco<'D'>(reco_helper,D_in,-0.5);
        }
    }

    if(C_flow || D_flow) {
        mcopy(greensfkt_helper,GF,POW2(n_spin*n_orb)*nk);
        fftw_execute_dft_helper(ifft_vert_oop,reco_helper,interpol_helper);
        fftw_execute_dft_helper(ifft_self_oop,greensfkt_helper,ff_helper);
        #pragma omp parallel for schedule(static)
        for(index_t i = 0; i < POW2(n_orbff)*POW4(n_spin)*nk; ++i)
            reco_helper[i] = std::conj(reco_helper[i]);
        #pragma omp parallel for schedule(static)
        for(index_t i = 0; i < POW2(n_spin*n_orb)*nk; ++i)
            greensfkt_helper[i] = std::conj(greensfkt_helper[i]);
        fftw_execute_dft_helper(ifft_vert_ip,reco_helper,reco_helper);
        fftw_execute_dft_helper(ifft_self_ip,greensfkt_helper,greensfkt_helper);

        // P and C channel have a similar form -> compress into one
        #pragma omp parallel for collapse(5) schedule(dynamic)
        for(index_t z = 0; z < model->nk[2]; ++z)
        for(index_t y = 0; y < model->nk[1]; ++y)
        for(index_t x = 0; x < model->nk[0]; ++x)
        for(index_t o3 = 0; o3<n_orb;++o3)
        for(index_t o1 = 0; o1<n_orb;++o1) {
            complex128_t tmp = 0.0;
            index_t ridx = z + model->nk[2]*(y+model->nk[1]*x);

            for(index_t b3 = 0; b3<bond_sizes[o3];++b3)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1) {
                index_t rb1b3 = rpb1mb3(model->nk,x,y,z,tu_ff[bond_offsets[o1]+b1].R,
                                                                tu_ff[bond_offsets[o3]+b3].R);
                const index_t mi1 =  ob_to_orbff[b1+n_bonds*o1];
                const index_t mi2 =  ob_to_orbff[b3+n_bonds*o3];
                    // + Lambda
                const index_t  idx_GFCD_p = rb1b3+nk*(tu_ff[bond_offsets[o3]+b3].oto+n_orb*
                                            tu_ff[bond_offsets[o1]+b1].oto);
                const index_t  idx_GFCD_m = rb1b3+nk*(tu_ff[bond_offsets[o1]+b1].oto+n_orb*
                                            tu_ff[bond_offsets[o3]+b3].oto);

                const complex128_t idx_C_p = interpol_helper[IDX3(mi2,mi1,rb1b3,n_orbff,nk)];
                const complex128_t idx_C_m = reco_helper[IDX3(mi1,mi2,rb1b3,n_orbff,nk)];

                tmp+= (greensfkt_helper[idx_GFCD_p]*idx_C_p
                            +(ff_helper[idx_GFCD_m]*idx_C_m));
            }
            self_en[ridx+nk*(o1+n_orb*o3)] -= 2.0*(tmp);
        }
    }
    if(P_flow) {
        reco<'P'>(reco_helper,P_in,1.0,0.0);

        mcopy(greensfkt_helper,GF,POW2(n_spin*n_orb)*nk);
        fftw_execute_dft_helper(ifft_vert_oop,reco_helper,interpol_helper);
        fftw_execute_dft_helper(fft_self_oop,greensfkt_helper,ff_helper);

        #pragma omp parallel for schedule(static)
        for(index_t i = 0; i < POW2(n_orbff)*POW4(n_spin)*nk; ++i)
            reco_helper[i] = std::conj(reco_helper[i]);
        #pragma omp parallel for schedule(static)
        for(index_t i = 0; i < POW2(n_spin*n_orb)*nk; ++i)
            greensfkt_helper[i] = std::conj(greensfkt_helper[i]);
        fftw_execute_dft_helper(ifft_vert_ip,reco_helper,reco_helper);
        fftw_execute_dft_helper(fft_self_ip,greensfkt_helper,greensfkt_helper);

        #pragma omp parallel for collapse(5) schedule(dynamic)
        for(index_t z = 0; z < model->nk[2]; ++z)
        for(index_t y = 0; y < model->nk[1]; ++y)
        for(index_t x = 0; x < model->nk[0]; ++x)
        for(index_t o3 = 0; o3<n_orb;++o3)
        for(index_t o1 = 0; o1<n_orb;++o1) {
            complex128_t tmp = 0.0;
            index_t ridx = z + model->nk[2]*(y+model->nk[1]*x);

            for(index_t b3 = 0; b3<bond_sizes[o3];++b3)
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1){
                index_t rb1b3 = rpb1mb3(model->nk,x,y,z,tu_ff[bond_offsets[o1]+b1].R,
                                                                tu_ff[bond_offsets[o3]+b3].R);
                // + Lambda
                const index_t mi1 =  ob_to_orbff[b1+n_bonds*o1];
                const index_t mi2 =  ob_to_orbff[b3+n_bonds*o3];
                const index_t idx_GFP_p = rb1b3+nk*(tu_ff[bond_offsets[o3]+b3].oto+n_orb*
                                            tu_ff[bond_offsets[o1]+b1].oto);
                const index_t idx_GFP_m = rb1b3+nk*(tu_ff[bond_offsets[o1]+b1].oto+n_orb*
                                            tu_ff[bond_offsets[o3]+b3].oto);

                const complex128_t idx_P_p = interpol_helper[IDX3(mi2,mi1,rb1b3,n_orbff,nk)];
                const complex128_t idx_P_m = reco_helper[IDX3(mi1,mi2,rb1b3,n_orbff,nk)];
                tmp += (ff_helper[idx_GFP_p]*idx_P_p
                        +(greensfkt_helper[idx_GFP_m]*idx_P_m));
            }
            self_en[ridx+nk*(o1+n_orb*o3)] -= (tmp)*2.0;
        }
        #pragma omp parallel for collapse(4) schedule(dynamic)
        for(index_t z = 0; z < model->nk[2]; ++z)
        for(index_t y = 0; y < model->nk[1]; ++y)
        for(index_t x = 0; x < model->nk[0]; ++x)
        for(index_t o1 = 0; o1<n_orb;++o1)
        for(index_t o4 = 0; o4<n_orb;++o4)
        for(index_t b4 = 0; b4<bond_sizes[o4];++b4) {
            complex128_t tmp = 0.0;
            index_t ridx = z + model->nk[2]*(y+model->nk[1]*x);
            for(index_t b1 = 0; b1<bond_sizes[o1];++b1) {
                const index_t mi1 =  ob_to_orbff[b1+n_bonds*o1];
                const index_t mi2 =  ob_to_orbff[b4+n_bonds*o4];
                index_t gfrb1b3 = rpb1pb3(model->nk,x,y,z,tu_ff[bond_offsets[o1]+b1].R,
                                                        tu_ff[bond_offsets[o4]+b4].R);
                index_t vrb1b3 = rpb1(model->nk,x,y,z,tu_ff[bond_offsets[o1]+b1].R);

                const index_t  idx_GFP_p = gfrb1b3+nk*
                            (o4+n_orb*tu_ff[bond_offsets[o1]+b1].oto);
                const index_t  idx_GFP_m = gfrb1b3+nk*(tu_ff[bond_offsets[o1]+b1].oto+n_orb*o4);

                const complex128_t idx_P_p = interpol_helper[IDX3(mi2,mi1,vrb1b3,n_orbff,nk)];
                const complex128_t idx_P_m = reco_helper[IDX3(mi1,mi2,vrb1b3,n_orbff,nk)];

                tmp += ff_helper[idx_GFP_p]*idx_P_p
                    +greensfkt_helper[idx_GFP_m]*idx_P_m;
            }
            self_en[ridx+nk*(o1+n_orb*tu_ff[bond_offsets[o4]+b4].oto)] += tmp;
        }
    }

    const index_t selfsize = POW2(n_orb*n_spin)*nk*n_frq;
    diverge_mpi_allreduce_complex_sum_inplace(self_en, selfsize);

    if(nk != nktot) {
        const double prefac = -pref_subst/(2.0*M_PI*(double)nk*(double)nk);
        for(index_t i = 0; i < POW2(n_spin*n_orb)*nk; ++i)
            self_en[i] *= prefac;
        fftw_execute_dft_helper(fft_fullself_ip,self_en,self_en);
        transpose(self_en,ff_helper, nk, POW2(n_orb*n_spin));
        diverge_symmetrize_2pt_coarse( model, self_en, ff_helper );
        transpose(self_en,ff_helper, POW2(n_orb*n_spin), nk);
        fftw_execute_dft_helper(ifft_fullself_ip,self_en,self_en);
        fft_interpolate(self_en);
    }else{
        const double prefac = -pref_subst/(2.0*M_PI*(double)nk*(double)nk);
        for(index_t i = 0; i < POW2(n_spin*n_orb)*nk; ++i)
            self_en[i] *= prefac;
        fftw_execute_dft_helper(fft_fullself_ip,self_en,self_en);
    }
    transpose(self_en,ff_helper, nktot, POW2(n_orb*n_spin));
    diverge_symmetrize_2pt_fine( model, self_en, ff_helper );
}
