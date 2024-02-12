/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "selfenergy.hpp"

typedef Eigen::Matrix<double,3,3,Eigen::RowMajor> CMat3d;

selfEnergy::selfEnergy(Vertex* vertex)
:
    model{vertex->model},
    tu_data{vertex->tu_data},
    n_orb{vertex->n_orb},
    n_spin{vertex->n_spin},
    ns{POW2(n_spin)},
    n_orbff{vertex->n_orbff},
    n_bonds{tu_data->n_bonds},
    nk{vertex->nk},
    nkf{vertex->nkf},
    nktot{nk*nkf},
    my_nk{vertex->my_nk},
    my_nk_off{vertex->my_nk_off},
    nff{(index_t)tu_data->existing_ff.size()},
    SU2{model->SU2 > 0},
    ob_to_orbff{tu_data->ob_to_orbff},
    existing_ff{tu_data->existing_ff},
    tu_ff{model->tu_ff},
    bond_sizes{tu_data->bond_sizes},
    bond_offsets{tu_data->bond_offsets},
    bos_freq{tu_data->bosonic_freq.data()},
    symm{vertex->symm},
    P_flow{vertex->P_flow},
    C_flow{vertex->C_flow},
    D_flow{vertex->D_flow}
{
    reco_helper = vertex->product_helper;
    interpol_helper = vertex->bubble_helper;
    ff_helper = (complex128_t*)calloc(POW2(n_orb*n_spin)*nktot,sizeof(complex128_t));
    if (nk!=1) {
        greensfkt_helper = (complex128_t*)calloc(POW2(n_spin*n_orb)*nktot,sizeof(complex128_t));
        generate_fftw_plans();
        ffkmesh = (complex128_t*) calloc(nk*nff, sizeof(complex128_t));
        const vector<Vec3i>& existing_ff = tu_data->existing_ff;

        #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(diverge_omp_num_threads())
        for(index_t rv = 0; rv < nff; ++rv)
        for(index_t kx = 0; kx < model->nk[0]; ++kx)
        for(index_t ky = 0; ky < model->nk[1]; ++ky)
        for(index_t kz = 0; kz < model->nk[2]; ++kz) {
            index_t k = IDX3(kx, ky, kz, model->nk[1], model->nk[2]);
            ffkmesh[k+nk*rv] = exp(2.*M_PI*I128
                                *(existing_ff[rv](0)*kx/(double)model->nk[0]
                                    +existing_ff[rv](1)*ky/(double)model->nk[1]
                                    +existing_ff[rv](2)*kz/(double)model->nk[2]));
        }
    }
}


void selfEnergy::generate_fftw_plans() {

    int dims[] = {(int)model->nk[0],(int)model->nk[1],(int)model->nk[2]};
    ifft_self_oop = fftw_plan_many_dft(3, dims,
        POW2(n_spin*n_orb),
        reinterpret_cast<fftw_complex*>(ff_helper),
        dims,1,nk,
        reinterpret_cast<fftw_complex*>(greensfkt_helper),
        dims,1,nk,FFTW_BACKWARD, FFTW_ESTIMATE);
    fft_self_oop = fftw_plan_many_dft(3, dims,
        POW2(n_spin*n_orb),
        reinterpret_cast<fftw_complex*>(ff_helper),
        dims,1,nk,
        reinterpret_cast<fftw_complex*>(greensfkt_helper),
        dims,1,nk,FFTW_FORWARD, FFTW_ESTIMATE);
    ifft_self_ip = fftw_plan_many_dft(3, dims,
        POW2(n_spin*n_orb),
        reinterpret_cast<fftw_complex*>(ff_helper),
        dims,1,nk,
        reinterpret_cast<fftw_complex*>(ff_helper),
        dims,1,nk,FFTW_BACKWARD, FFTW_ESTIMATE);
    fft_self_ip = fftw_plan_many_dft(3, dims,
        POW2(n_spin*n_orb),
        reinterpret_cast<fftw_complex*>(ff_helper),
        dims,1,nk,
        reinterpret_cast<fftw_complex*>(ff_helper),
        dims,1,nk,FFTW_FORWARD, FFTW_ESTIMATE);

    int nkff[3];
    for (int i=0; i<3; ++i) nkff[i] = model->nk[i] * model->nkf[i];

    fft_self_oop_gfmesh = fftw_plan_many_dft(3, nkff,
        POW2(n_spin*n_orb),
        reinterpret_cast<fftw_complex*>(ff_helper),
        nkff,1,nktot,
        reinterpret_cast<fftw_complex*>(greensfkt_helper),
        nkff,1,nktot,FFTW_FORWARD, FFTW_ESTIMATE);


    ifft_fullself_ip = fftw_plan_many_dft(3, dims,
        POW2(n_spin*n_orb),
        reinterpret_cast<fftw_complex*>(ff_helper),
        dims,1,nk,
        reinterpret_cast<fftw_complex*>(ff_helper),
        dims,1,nk,FFTW_BACKWARD, FFTW_ESTIMATE);
    fft_fullself_ip = fftw_plan_many_dft(3, dims,
        POW2(n_spin*n_orb),
        reinterpret_cast<fftw_complex*>(ff_helper),
        dims,1,nk,
        reinterpret_cast<fftw_complex*>(ff_helper),
        dims,1,nk,FFTW_FORWARD, FFTW_ESTIMATE);

    ifft_vert_oop = fftw_plan_many_dft(3, dims,
        POW2(n_orbff)*POW4(n_spin),
        reinterpret_cast<fftw_complex*>(reco_helper),
        dims,1,nk,
        reinterpret_cast<fftw_complex*>(interpol_helper),
        dims,1,nk,FFTW_BACKWARD, FFTW_ESTIMATE);
    fft_vert_oop = fftw_plan_many_dft(3, dims,
        POW2(n_orbff)*POW4(n_spin),
        reinterpret_cast<fftw_complex*>(reco_helper),
        dims,1,nk,
        reinterpret_cast<fftw_complex*>(interpol_helper),
        dims,1,nk,FFTW_FORWARD, FFTW_ESTIMATE);
    ifft_vert_ip = fftw_plan_many_dft(3, dims,
        POW2(n_orbff)*POW4(n_spin),
        reinterpret_cast<fftw_complex*>(reco_helper),
        dims,1,nk,
        reinterpret_cast<fftw_complex*>(reco_helper),
        dims,1,nk,FFTW_BACKWARD, FFTW_ESTIMATE);
    fft_vert_ip = fftw_plan_many_dft(3, dims,
        POW2(n_orbff)*POW4(n_spin),
        reinterpret_cast<fftw_complex*>(reco_helper),
        dims,1,nk,
        reinterpret_cast<fftw_complex*>(reco_helper),
        dims,1,nk,FFTW_FORWARD, FFTW_ESTIMATE);
}

