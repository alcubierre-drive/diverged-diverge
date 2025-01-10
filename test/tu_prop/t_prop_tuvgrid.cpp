#ifndef DIVERGE_SKIP_TESTS

#include "../catch.hpp"
#include "../../src/diverge_model.h"
#include "../../src/diverge_flow_step.h"
#include "../../src/diverge_flow_step_internal.hpp"
#include "../model_examples/old.h"
#include "../../src/diverge_model_internals.h"
#include "../model_examples/model_constructor.hpp"
#include "../../src/tu/propagator.h"
#include "../../src/grid/loops.hpp"
#include "../../src/tu/vertex.h"
#include "../../src/tu/projection.h"
#include "../helper_functions.hpp"

void unproject_loop(complex128_t* buf,complex128_t* loop,const diverge_model_t* model) {
    tu_data_t* tu_data = (tu_data_t*)(model -> internals-> tu_data);
    index_t* mi_to_ofrom=tu_data->mi_to_ofrom;
    index_t* mi_to_oto=tu_data->mi_to_oto;
    index_t* mi_to_R=tu_data->mi_to_R;
    double* Rvec = (double*)calloc(model->n_tu_ff*3,sizeof(double));
    fill_Rvec_from_vec(model, mi_to_R,Rvec);
    index_t nk = tu_data->nk;
    index_t my_nk = tu_data->my_nk;
    index_t my_nk_off = tu_data->my_nk_off;
    index_t n_orbff = tu_data->n_orbff;
    index_t n_spin = model->n_spin;
    index_t n_orb = model->n_orb;
    index_t nb = n_orb*n_spin;
    double* kmesh = model->internals->kmesh;
    memset((void*)buf,0,POW4(nb)*POW2(nk)*sizeof(complex128_t));
    for(index_t q = 0; q < my_nk; ++q)
    for(index_t k = 0; k < nk; ++k)
    for(index_t s1 = 0; s1 < n_spin; ++s1)
    for(index_t s2 = 0; s2 < n_spin; ++s2)
    for(index_t s3 = 0; s3 < n_spin; ++s3)
    for(index_t s4 = 0; s4 < n_spin; ++s4)
    for(index_t m1 = 0; m1 < n_orbff; ++m1)
    for(index_t m2 = 0; m2 < n_orbff; ++m2){
        index_t o1 = mi_to_ofrom[m1];
        index_t o2 = mi_to_oto[m1];
        index_t o3 = mi_to_ofrom[m2];
        index_t o4 = mi_to_oto[m2];
        index_t bid = o4+n_orb*(s4+n_spin*(o3+n_orb*(s3+n_spin*(o2+n_orb*(
                                s2+n_spin*(o1+n_orb*s1))))));

        buf[bid+POW4(n_spin*n_orb)*(k+nk*(q+my_nk_off))] += loop[m1 + n_orbff*(s1+n_spin*(
                    s2+n_spin*(m2+n_orbff*(s3+n_spin*(s4+n_spin*q)))))]*
                    std::conj(ff(kmesh+3*k,Rvec+3*m1))*ff(kmesh+3*k,Rvec+3*m2)/(double)nk;
    }
    diverge_mpi_allreduce_complex_sum_inplace(buf, POW4(nb)*POW2(nk));
    free(Rvec);
}


TEST_CASE("loop tu v grid", "[prop][looptugrid]") {
    diverge_model_t* mod_tu = gen_honeycomb_lat_hub(6, 6);
    diverge_model_internals_tu(mod_tu, 12.0);
    diverge_model_t* mod_grid = gen_honeycomb_lat_hub(6, 6);
    diverge_model_internals_grid(mod_grid);
    tu_data_t* tu_data = (tu_data_t*)(mod_tu -> internals-> tu_data);
    index_t nk = tu_data->nk;
    index_t n_orbff = tu_data->n_orbff;
    index_t n_spin = mod_tu->n_spin;
    index_t n_orb = mod_tu->n_orb;
    index_t nb = n_orb*n_spin;

    index_t fsize = POW4(nb)*POW2(nk);
    index_t tusize = POW2(n_spin*n_spin*n_orbff)*nk;

    complex128_t* Ltu = (complex128_t*)calloc(tusize,sizeof(complex128_t));
    complex128_t* Lgrid;
    complex128_t* Lreco = (complex128_t*)calloc(fsize,sizeof(complex128_t));
    complex128_t* Lrecog = (complex128_t*)calloc(fsize,sizeof(complex128_t));

    tu_prop_t* tu_loop = tu_prop_init(mod_tu);
    projection_t* ProjMom = projection_init(mod_tu);
    Vertex* VertMom = Vertex_init(mod_tu, ProjMom, tu_loop, 1,1,1,0);

    grid::Loops grid_loop (mod_grid);
    tu_prop_GF_zero_T( tu_loop, I128*0.1, NULL );
    tu_prop_ph_loop( tu_loop, Ltu, NULL );
    Lgrid = grid_loop.X_C(0.1);
    memset((void*)Lrecog,0,fsize*sizeof(complex128_t));
    memcpy(Lrecog + grid_loop.get_qstart()*POW4(nb)*nk,Lgrid,POW4(nb)*grid_loop.get_qsize()*nk*sizeof(complex128_t));
    diverge_mpi_allreduce_complex_sum_inplace(Lrecog, fsize);
    unproject_loop(Lreco,Ltu,mod_tu);
    CHECK(check_hermicity(Ltu, n_spin*n_spin*n_orbff,nk));

    std::sort(Lrecog, Lrecog+fsize, [](complex128_t i, complex128_t j)
                        { return std::abs(i) > std::abs(j); });
    std::sort(Lreco, Lreco+fsize, [](complex128_t i, complex128_t j)
                        { return std::abs(i) > std::abs(j); });

    for(index_t i = 0; i < fsize;++i) {
        CHECK(std::abs(std::abs(Lrecog[i])-std::abs(Lreco[i])) < 1e-10);
    }


    tu_prop_pp_loop( tu_loop, Ltu, NULL );
    Lgrid = grid_loop.X_P(0.1);
    memset((void*)Lrecog,0,fsize*sizeof(complex128_t));
    memcpy(Lrecog + grid_loop.get_qstart()*POW4(nb)*nk,Lgrid,POW4(nb)*grid_loop.get_qsize()*nk*sizeof(complex128_t));
    diverge_mpi_allreduce_complex_sum_inplace(Lrecog, fsize);
    unproject_loop(Lreco,Ltu,mod_tu);
    CHECK(check_hermicity(Ltu, n_spin*n_spin*n_orbff,nk));

    std::sort(Lrecog, Lrecog+fsize, [](complex128_t i, complex128_t j)
                        { return std::abs(i) > std::abs(j); });
    std::sort(Lreco, Lreco+fsize, [](complex128_t i, complex128_t j)
                        { return std::abs(i) > std::abs(j); });

    for(index_t i = 0; i < fsize;++i) {
        CHECK(std::abs(std::abs(Lrecog[i])-std::abs(Lreco[i])) < 1e-10);
    }
    free(Ltu);
    free(Lrecog);
    free(Lreco);
    Vertex_free(VertMom);
    tu_prop_free(tu_loop);
    projection_free(ProjMom);
    diverge_model_free(mod_tu);
    diverge_model_free(mod_grid);
}

#endif // DIVERGE_SKIP_TESTS
