#include "all.h"
#include "../../src/diverge_model.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/diverge_internals_struct.h"
#include "../../src/misc/generate_symmetries.h"
#include "../../src/misc/linalg.h"
LINALG_IMPLEMENT

#ifndef DIVERGE_SKIP_TESTS

#define isqrt2 0.7071067811865475
static const claM2cd_t xy_pm = {{{isqrt2, I*isqrt2},
                                 {isqrt2,-I*isqrt2}}},
                       xx = {{{1,0},
                              {0,0}}},
                       yy = {{{0,0},
                              {0,1}}},
                       xy = {{{0,1},
                              {0,0}}},
                       yx = {{{0,0},
                              {1,0}}};
static inline claM2cd_t mult_to_ppm( claM2cd_t in ) {
    return claM2cd_matmul(xy_pm, claM2cd_matmul(in, claM2cd_adjoint(xy_pm)));
}

diverge_model_t* gen_triangular_pxpy_sym(index_t nkx, index_t nky, int to_ppm) {
    diverge_model_t* model = diverge_model_init();
    model->n_orb = 2;
    model->SU2 = true;
    site_descr_t sites[2] = {0};

    if (to_ppm) {
        sites[0].n_functions = 2;
        sites[1].n_functions = 2;
        sites[0].amplitude[0] = 1./sqrt(2);
        sites[0].function[0] = orb_px;
        sites[0].amplitude[1] = I/sqrt(2);
        sites[0].function[1] = orb_py;
        sites[1].amplitude[0] = 1./sqrt(2);
        sites[1].function[0] = orb_px;
        sites[1].amplitude[1] = -I/sqrt(2);
        sites[1].function[1] = orb_py;
    } else {
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

    model->hop = calloc(1024,sizeof(rs_hopping_t));
    model->nk[0] = nkx, model->nk[1] = nky;
    model->nkf[0] = 1, model->nkf[1] = 1;
    double t_xx = 1.0;
    double t_yy = 0.8;
    double t_xy = -sqrt(3.)*(t_yy-t_xx)/4.;
    if (to_ppm) {
        claM2cd_t xx_t = mult_to_ppm(xx), yy_t = mult_to_ppm(yy),
                  xy_t = mult_to_ppm(xy), yx_t = mult_to_ppm(yx);
        for (index_t o1 = 0; o1 < model->n_orb; ++o1)
        for (index_t o2 = 0; o2 < model->n_orb; ++o2) {
            // xx hopping
            model->hop[model->n_hop++] = (rs_hopping_t){{1,0,0},o1,o2,0,0,xx_t.m[o1][o2]*t_xx};
            model->hop[model->n_hop++] = (rs_hopping_t){{-1,0,0},o1,o2,0,0,xx_t.m[o1][o2]*t_xx};
            model->hop[model->n_hop++] = (rs_hopping_t){{0,1,0},o1,o2,0,0,xx_t.m[o1][o2]*(t_xx+3.*t_yy)/4.};
            model->hop[model->n_hop++] = (rs_hopping_t){{0,-1,0},o1,o2,0,0,xx_t.m[o1][o2]*(t_xx+3.*t_yy)/4.};
            model->hop[model->n_hop++] = (rs_hopping_t){{1,1,0},o1,o2,0,0,xx_t.m[o1][o2]*(t_xx+3.*t_yy)/4.};
            model->hop[model->n_hop++] = (rs_hopping_t){{-1,-1,0},o1,o2,0,0,xx_t.m[o1][o2]*(t_xx+3.*t_yy)/4.};
            // yy hopping
            model->hop[model->n_hop++] = (rs_hopping_t){{1,0,0},o1,o2,0,0,yy_t.m[o1][o2]*t_yy};
            model->hop[model->n_hop++] = (rs_hopping_t){{-1,0,0},o1,o2,0,0,yy_t.m[o1][o2]*t_yy};
            model->hop[model->n_hop++] = (rs_hopping_t){{0,1,0},o1,o2,0,0,yy_t.m[o1][o2]*(t_yy+3.*t_xx)/4.};
            model->hop[model->n_hop++] = (rs_hopping_t){{0,-1,0},o1,o2,0,0,yy_t.m[o1][o2]*(t_yy+3.*t_xx)/4.};
            model->hop[model->n_hop++] = (rs_hopping_t){{1,1,0},o1,o2,0,0,yy_t.m[o1][o2]*(t_yy+3.*t_xx)/4.};
            model->hop[model->n_hop++] = (rs_hopping_t){{-1,-1,0},o1,o2,0,0,yy_t.m[o1][o2]*(t_yy+3.*t_xx)/4.};
            // xy hopping
            model->hop[model->n_hop++] = (rs_hopping_t){{0,1,0},o1,o2,0,0,-xy_t.m[o1][o2]*t_xy};
            model->hop[model->n_hop++] = (rs_hopping_t){{0,-1,0},o1,o2,0,0,-xy_t.m[o1][o2]*t_xy};
            model->hop[model->n_hop++] = (rs_hopping_t){{1,1,0},o1,o2,0,0,xy_t.m[o1][o2]*t_xy};
            model->hop[model->n_hop++] = (rs_hopping_t){{-1,-1,0},o1,o2,0,0,xy_t.m[o1][o2]*t_xy};
            // yx hopping
            model->hop[model->n_hop++] = (rs_hopping_t){{0,1,0},o1,o2,0,0,-yx_t.m[o1][o2]*t_xy};
            model->hop[model->n_hop++] = (rs_hopping_t){{0,-1,0},o1,o2,0,0,-yx_t.m[o1][o2]*t_xy};
            model->hop[model->n_hop++] = (rs_hopping_t){{1,1,0},o1,o2,0,0,yx_t.m[o1][o2]*t_xy};
            model->hop[model->n_hop++] = (rs_hopping_t){{-1,-1,0},o1,o2,0,0,yx_t.m[o1][o2]*t_xy};
        }
    }else{
        // xx hopping
        model->hop[model->n_hop++] = (rs_hopping_t){{1,0,0},0,0,0,0,t_xx};
        model->hop[model->n_hop++] = (rs_hopping_t){{-1,0,0},0,0,0,0,t_xx};
        model->hop[model->n_hop++] = (rs_hopping_t){{0,1,0},0,0,0,0,(t_xx+3.*t_yy)/4.};
        model->hop[model->n_hop++] = (rs_hopping_t){{0,-1,0},0,0,0,0,(t_xx+3.*t_yy)/4.};
        model->hop[model->n_hop++] = (rs_hopping_t){{1,1,0},0,0,0,0,(t_xx+3.*t_yy)/4.};
        model->hop[model->n_hop++] = (rs_hopping_t){{-1,-1,0},0,0,0,0,(t_xx+3.*t_yy)/4.};
        // yy hopping
        model->hop[model->n_hop++] = (rs_hopping_t){{1,0,0},1,1,0,0,t_yy};
        model->hop[model->n_hop++] = (rs_hopping_t){{-1,0,0},1,1,0,0,t_yy};
        model->hop[model->n_hop++] = (rs_hopping_t){{0,1,0},1,1,0,0,(t_yy+3.*t_xx)/4.};
        model->hop[model->n_hop++] = (rs_hopping_t){{0,-1,0},1,1,0,0,(t_yy+3.*t_xx)/4.};
        model->hop[model->n_hop++] = (rs_hopping_t){{1,1,0},1,1,0,0,(t_yy+3.*t_xx)/4.};
        model->hop[model->n_hop++] = (rs_hopping_t){{-1,-1,0},1,1,0,0,(t_yy+3.*t_xx)/4.};
        // xy hopping
        model->hop[model->n_hop++] = (rs_hopping_t){{0,1,0},0,1,0,0, -t_xy};
        model->hop[model->n_hop++] = (rs_hopping_t){{0,-1,0},0,1,0,0, -t_xy};
        model->hop[model->n_hop++] = (rs_hopping_t){{1,1,0},0,1,0,0, t_xy};
        model->hop[model->n_hop++] = (rs_hopping_t){{-1,-1,0},0,1,0,0, t_xy};

        model->hop[model->n_hop++] = (rs_hopping_t){{0,1,0},1,0,0,0, -t_xy};
        model->hop[model->n_hop++] = (rs_hopping_t){{0,-1,0},1,0,0,0, -t_xy};
        model->hop[model->n_hop++] = (rs_hopping_t){{1,1,0},1,0,0,0, t_xy};
        model->hop[model->n_hop++] = (rs_hopping_t){{-1,-1,0},1,0,0,0, t_xy};
    }

    model->n_vert = model->n_orb*model->n_orb;
    model->vert = calloc(1024,sizeof(rs_vertex_t));
    for (index_t o1 = 0; o1<model->n_orb; ++o1)
    for (index_t o2 = 0; o2<model->n_orb; ++o2) {
        index_t idx = o1 + model->n_orb * o2;
        model->vert[idx].chan = 'D', model->vert[idx].V = 3.0;
        model->vert[idx].o1 = o1, model->vert[idx].o2 = o2;
    }

    model->n_sym = 12;

    index_t cnt = 0;
    model->orb_symmetries = calloc(20*POW2(model->n_orb*model->n_spin),sizeof(complex128_t));
    index_t symsize = POW2(model->n_orb*model->n_spin);
    sym_op_t curr_symm[3] = {0};
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
