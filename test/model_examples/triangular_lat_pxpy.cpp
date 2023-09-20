#ifndef DIVERGE_SKIP_TESTS

#include "model_constructor.hpp"
#include "../../src/diverge_model.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/diverge_internals_struct.h"
#include "../../src/diverge_Eigen3.hpp"
#include "../../src/misc/generate_symmetries.h"
#include <random>


typedef Eigen::Matrix<complex128_t,2,2,Eigen::RowMajor> CMat2cd;
const Eigen::Matrix2cd xy_pm =
            (Eigen::Matrix2cd() << 1, I128,
                                    1,-I128).finished()/sqrt(2.);

const Eigen::Matrix2cd xx = (Eigen::Matrix2cd() << 1, 0,
                                   0, 0).finished();
const Eigen::Matrix2cd yy = (Eigen::Matrix2cd() << 0, 0,
                                   0, 1).finished();
const Eigen::Matrix2cd xy = (Eigen::Matrix2cd() << 0, 1,
                                   0, 0).finished();
const Eigen::Matrix2cd yx = (Eigen::Matrix2cd() << 0, 0,
                                   1, 0).finished();

diverge_model_t* gen_triangular_pxpy_sym(index_t nk, bool to_ppm) {
    diverge_model_t* model = diverge_model_init();
    model->n_orb = 2;
    model->SU2 = true;
    site_descr_t sites[2];

    if(to_ppm) {
        sites[0].n_functions = 2;
        sites[1].n_functions = 2;
        sites[0].amplitude[0] = 1./sqrt(2);
        sites[0].function[0] = orb_px;
        sites[0].amplitude[1] = I128/sqrt(2);
        sites[0].function[1] = orb_py;
        sites[1].amplitude[0] = 1./sqrt(2);
        sites[1].function[0] = orb_px;
        sites[1].amplitude[1] = -I128/sqrt(2);
        sites[1].function[1] = orb_py;
    }else{
        sites[0].n_functions = 1;
        sites[0].amplitude[0] = 1.;
        sites[0].function[0] = orb_px;
        sites[1].n_functions = 1;
        sites[1].amplitude[0] = 1.;
        sites[1].function[0] = orb_py;
    }

    model->lattice[0][0] = 1.0, model->lattice[0][1] = 0.0, model->lattice[0][2] = 0.0;
    model->lattice[1][0] = -0.5, model->lattice[1][1] = 0.5*sqrt(3.), model->lattice[1][2] = 0.0;
    model->lattice[2][0] = 0.0, model->lattice[2][1] = 0.0, model->lattice[2][2] = 1.0;

    model->hop = (rs_hopping_t*)calloc(300,sizeof(rs_hopping_t));
    model->nk[0] = nk, model->nk[1] = nk;
    model->nkf[0] = 1, model->nkf[1] = 1;
    double t_xx = 1.0;
    double t_yy = 0.8;
    double t_xy = -sqrt(3.)*(t_yy-t_xx)/4.;
    if(to_ppm){
        Eigen::Matrix2cd xx_t = xy_pm*xx*xy_pm.adjoint();
        Eigen::Matrix2cd yy_t = xy_pm*yy*xy_pm.adjoint();
        Eigen::Matrix2cd xy_t = xy_pm*xy*xy_pm.adjoint();
        Eigen::Matrix2cd yx_t = xy_pm*yx*xy_pm.adjoint();
        for(index_t o1 = 0; o1 < model->n_orb; ++o1)
        for(index_t o2 = 0; o2 < model->n_orb; ++o2) {
            // xx hopping
            model->hop[model->n_hop++] = rs_hopping_t{{1,0,0},o1,o2,0,0,xx_t(o1,o2)*t_xx};
            model->hop[model->n_hop++] = rs_hopping_t{{-1,0,0},o1,o2,0,0,xx_t(o1,o2)*t_xx};
            model->hop[model->n_hop++] = rs_hopping_t{{0,1,0},o1,o2,0,0,xx_t(o1,o2)*(t_xx+3.*t_yy)/4.};
            model->hop[model->n_hop++] = rs_hopping_t{{0,-1,0},o1,o2,0,0,xx_t(o1,o2)*(t_xx+3.*t_yy)/4.};
            model->hop[model->n_hop++] = rs_hopping_t{{1,1,0},o1,o2,0,0,xx_t(o1,o2)*(t_xx+3.*t_yy)/4.};
            model->hop[model->n_hop++] = rs_hopping_t{{-1,-1,0},o1,o2,0,0,xx_t(o1,o2)*(t_xx+3.*t_yy)/4.};
            // yy hopping
            model->hop[model->n_hop++] = rs_hopping_t{{1,0,0},o1,o2,0,0,yy_t(o1,o2)*t_yy};
            model->hop[model->n_hop++] = rs_hopping_t{{-1,0,0},o1,o2,0,0,yy_t(o1,o2)*t_yy};
            model->hop[model->n_hop++] = rs_hopping_t{{0,1,0},o1,o2,0,0,yy_t(o1,o2)*(t_yy+3.*t_xx)/4.};
            model->hop[model->n_hop++] = rs_hopping_t{{0,-1,0},o1,o2,0,0,yy_t(o1,o2)*(t_yy+3.*t_xx)/4.};
            model->hop[model->n_hop++] = rs_hopping_t{{1,1,0},o1,o2,0,0,yy_t(o1,o2)*(t_yy+3.*t_xx)/4.};
            model->hop[model->n_hop++] = rs_hopping_t{{-1,-1,0},o1,o2,0,0,yy_t(o1,o2)*(t_yy+3.*t_xx)/4.};
            // xy hopping
            model->hop[model->n_hop++] = rs_hopping_t{{0,1,0},o1,o2,0,0,-xy_t(o1,o2)*t_xy};
            model->hop[model->n_hop++] = rs_hopping_t{{0,-1,0},o1,o2,0,0,-xy_t(o1,o2)*t_xy};
            model->hop[model->n_hop++] = rs_hopping_t{{1,1,0},o1,o2,0,0,xy_t(o1,o2)*t_xy};
            model->hop[model->n_hop++] = rs_hopping_t{{-1,-1,0},o1,o2,0,0,xy_t(o1,o2)*t_xy};
            // yx hopping
            model->hop[model->n_hop++] = rs_hopping_t{{0,1,0},o1,o2,0,0,-yx_t(o1,o2)*t_xy};
            model->hop[model->n_hop++] = rs_hopping_t{{0,-1,0},o1,o2,0,0,-yx_t(o1,o2)*t_xy};
            model->hop[model->n_hop++] = rs_hopping_t{{1,1,0},o1,o2,0,0,yx_t(o1,o2)*t_xy};
            model->hop[model->n_hop++] = rs_hopping_t{{-1,-1,0},o1,o2,0,0,yx_t(o1,o2)*t_xy};
        }
    }else{
        // xx hopping
        model->hop[model->n_hop++] = rs_hopping_t{{1,0,0},0,0,0,0,t_xx};
        model->hop[model->n_hop++] = rs_hopping_t{{-1,0,0},0,0,0,0,t_xx};
        model->hop[model->n_hop++] = rs_hopping_t{{0,1,0},0,0,0,0,(t_xx+3.*t_yy)/4.};
        model->hop[model->n_hop++] = rs_hopping_t{{0,-1,0},0,0,0,0,(t_xx+3.*t_yy)/4.};
        model->hop[model->n_hop++] = rs_hopping_t{{1,1,0},0,0,0,0,(t_xx+3.*t_yy)/4.};
        model->hop[model->n_hop++] = rs_hopping_t{{-1,-1,0},0,0,0,0,(t_xx+3.*t_yy)/4.};
        // yy hopping
        model->hop[model->n_hop++] = rs_hopping_t{{1,0,0},1,1,0,0,t_yy};
        model->hop[model->n_hop++] = rs_hopping_t{{-1,0,0},1,1,0,0,t_yy};
        model->hop[model->n_hop++] = rs_hopping_t{{0,1,0},1,1,0,0,(t_yy+3.*t_xx)/4.};
        model->hop[model->n_hop++] = rs_hopping_t{{0,-1,0},1,1,0,0,(t_yy+3.*t_xx)/4.};
        model->hop[model->n_hop++] = rs_hopping_t{{1,1,0},1,1,0,0,(t_yy+3.*t_xx)/4.};
        model->hop[model->n_hop++] = rs_hopping_t{{-1,-1,0},1,1,0,0,(t_yy+3.*t_xx)/4.};
        // xy hopping
        model->hop[model->n_hop++] = rs_hopping_t{{0,1,0},0,1,0,0, -t_xy};
        model->hop[model->n_hop++] = rs_hopping_t{{0,-1,0},0,1,0,0, -t_xy};
        model->hop[model->n_hop++] = rs_hopping_t{{1,1,0},0,1,0,0, t_xy};
        model->hop[model->n_hop++] = rs_hopping_t{{-1,-1,0},0,1,0,0, t_xy};

        model->hop[model->n_hop++] = rs_hopping_t{{0,1,0},1,0,0,0, -t_xy};
        model->hop[model->n_hop++] = rs_hopping_t{{0,-1,0},1,0,0,0, -t_xy};
        model->hop[model->n_hop++] = rs_hopping_t{{1,1,0},1,0,0,0, t_xy};
        model->hop[model->n_hop++] = rs_hopping_t{{-1,-1,0},1,0,0,0, t_xy};
    }

    model->n_vert = model->n_orb*model->n_orb;
    model->vert = (rs_vertex_t*)calloc(model -> n_vert,sizeof(rs_vertex_t));
    for(index_t o1 = 0; o1 < model->n_orb; ++o1)
    for(index_t o2 = 0; o2 < model->n_orb; ++o2) {
        index_t idx = o1 + model->n_orb * o2;
        model->vert[idx].chan = 'D', model->vert[idx].V = 3.0;
        model->vert[idx].o1 = o1, model->vert[idx].o2 = o2;
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
    /*
    Map<Vec3d> UC1(model->lattice[0]);
    Map<Vec3d> UC2(model->lattice[1]);
    Map<Vec3d> UC3(model->lattice[2]);

    for(index_t k = 0; k < nk*nk; ++k)
    for(index_t o1 = 0; o1 < 2; ++o1)
    for(index_t o2 = 0; o2 < 2; ++o2) {

        Map<CMat2cd> result( model->internals->ham + k*4);
        Map<Vec3d> kp(model->internals->kmesh + 3*k);
        result = Mat2cd::Zero();

        for (index_t i = 0; i < model->n_hop; ++i) {
            result(model->hop[i].o1,model->hop[i].o2) +=
                     model->hop[i].t * exp(-I128*kp.dot( model->hop[i].R[0]*UC1 +
                                                       model->hop[i].R[1]*UC2 +
                                                       model->hop[i].R[2]*UC3 ));
            if(model->hop[i].o1!=model->hop[i].o2)
                result(model->hop[i].o2,model->hop[i].o1) +=
                        std::conj(model->hop[i].t * exp(-I128*kp.dot( model->hop[i].R[0]*UC1 +
                                                       model->hop[i].R[1]*UC2 +
                                                       model->hop[i].R[2]*UC3 )));
        }
        Mat2cd herm = 0.5*(result + result.adjoint().transpose());
        double diff = (herm - result).norm();
        result = herm;

        if(to_ppm)
            result = xy_pm*result*xy_pm.adjoint();
    }
    */
    return model;
}

#endif // DIVERGE_SKIP_TESTS
