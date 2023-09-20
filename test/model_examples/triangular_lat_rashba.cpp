#ifndef DIVERGE_SKIP_TESTS

#include "model_constructor.hpp"
#include "../../src/diverge_model.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/diverge_internals_struct.h"
#include "../../src/diverge_Eigen3.hpp"
#include "../../src/misc/generate_symmetries.h"
#include <random>

diverge_model_t* gen_triangular_lat_rashba(index_t nk) {
    diverge_model_t* model = diverge_model_init();
    model->n_orb = 1;
    model->SU2 = false;
    model->n_spin = 2;
    site_descr_t sites[2];

    sites[0].n_functions = 1;
    sites[0].amplitude[0] = 1.;
    sites[0].function[0] = orb_s;

    model->lattice[0][0] = 1.0, model->lattice[0][1] = 0.0, model->lattice[0][2] = 0.0;
    model->lattice[1][0] = -0.5, model->lattice[1][1] = 0.5*sqrt(3.), model->lattice[1][2] = 0.0;
    model->lattice[2][0] = 0.0, model->lattice[2][1] = 0.0, model->lattice[2][2] = 1.0;

    model->hop = (rs_hopping_t*)calloc(300,sizeof(rs_hopping_t));
    model->nk[0] = nk, model->nk[1] = nk;
    model->nkf[0] = 1, model->nkf[1] = 1;
    double t = 1.0;
    Mat2cd pauli_x; pauli_x << 0,1,1,0;
    Mat2cd pauli_y; pauli_y << 0,-I128,I128,0;
    for (int Rx=-2; Rx<=2; ++Rx)
    for (int Ry=-2; Ry<=2; ++Ry) {
        Vec3d bond = Rx * Map<Vec3d>((double*)(model->lattice)) +
                     Ry * Map<Vec3d>((double*)(model->lattice)+3);
        Mat2cd pauli_bond = bond(0) * pauli_y - bond(1) * pauli_x;
        double t_alpha = 0.0;
        if (fabs(bond.norm() - 1.0) < 1.e-7) t_alpha = t * 0.5;

        for (int s1=0; s1<2; ++s1)
        for (int s2=0; s2<2; ++s2)
            model->hop[model->n_hop++] = rs_hopping_t{ {Rx,Ry,0}, 0,0,s1,s2, -I128*pauli_bond(s2,s1)*t_alpha };

        if(fabs(bond.norm() - 1.0) < 1.e-7)
        for (int s1=0; s1<2; ++s1)
            model->hop[model->n_hop++] = rs_hopping_t{ {Rx,Ry,0}, 0,0,s1,s1, t };

    }

    model->n_vert = model->n_orb*model->n_orb;
    model->vert = (rs_vertex_t*)calloc(model -> n_vert,sizeof(rs_vertex_t));
    for(index_t o1 = 0; o1 < model->n_orb; ++o1){
        model->vert[o1].chan = 'D', model->vert[o1].V = 3.0;
        model->vert[o1].o1 = o1, model->vert[o1].o2 = o1,model->vert[o1].s1 = -1;
    }

    model->n_sym = 12;

    index_t cnt = 0;
    model->orb_symmetries = (complex128_t*)calloc(20
                            *POW2(model->n_orb*model->n_spin),sizeof(complex128_t));
    index_t symsize = POW2(model->n_orb*model->n_spin);
    sym_op_t curr_symm[3];
    curr_symm[0].type = 'R'; curr_symm[0].normal_vector[0] = 0.;
    curr_symm[0].normal_vector[1] = 0.;curr_symm[0].normal_vector[2] = 1.;
    curr_symm[1].type = 'M'; curr_symm[1].normal_vector[0] = 1.;
    curr_symm[1].normal_vector[1] = 0.;curr_symm[1].normal_vector[2] = 0.;
    curr_symm[2].type = 'R'; curr_symm[2].normal_vector[0] = 0.;
    curr_symm[2].normal_vector[1] = 0.;curr_symm[2].normal_vector[2] = 1.;
    // C6

    for(cnt = 0; cnt < 6; ++cnt) {
        curr_symm[0].angle = 60*cnt;
        diverge_generate_symm_trafo(model->n_spin,sites, model->n_orb,
                curr_symm, 1,&(model->rs_symmetries[cnt][0][0]),model->orb_symmetries+cnt*symsize);
    }

    // mirror planes
    for(cnt = 6; cnt < 9; ++cnt) {
        curr_symm[0].angle = -120*(cnt-6);
        curr_symm[2].angle = 120*(cnt-6);
        diverge_generate_symm_trafo(model->n_spin,sites, model->n_orb,
                curr_symm, 3,&(model->rs_symmetries[cnt][0][0]),model->orb_symmetries+cnt*symsize);
    }
    curr_symm[1].type = 'M'; curr_symm[1].normal_vector[0] = 0.;
    curr_symm[1].normal_vector[1] = 1.;
    for(cnt = 9; cnt < 12; ++cnt) {
        curr_symm[0].angle = -120*(cnt-9);
        curr_symm[2].angle = 120*(cnt-9);
        diverge_generate_symm_trafo(model->n_spin,sites, model->n_orb,
                curr_symm, 3,&(model->rs_symmetries[cnt][0][0]),model->orb_symmetries+cnt*symsize);
    }


    diverge_model_internals_common( model );

    return model;
}

#endif // DIVERGE_SKIP_TESTS
