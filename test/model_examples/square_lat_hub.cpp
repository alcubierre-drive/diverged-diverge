#ifndef DIVERGE_SKIP_TESTS

#include "model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../../src/diverge_Eigen3.hpp"
#include "../../src/misc/generate_symmetries.h"



diverge_model_t* gen_square_lat_hub( index_t nkx, index_t nky, index_t lx,
        index_t ly,double t, double tp, double U, double mu,
        index_t nkxf, index_t nkyf ) {
    diverge_model_t* model = diverge_model_init();
    model->nk[0] = nkx, model->nk[1] = nky;
    model->nkf[0] = nkxf, model->nkf[1] = nkyf;

    model->n_orb = lx*ly;
    model->SU2 = true;
    model->lattice[0][0] = lx, model->lattice[0][1] = 0.0, model->lattice[0][2] = 0.0;
    model->lattice[1][0] = 0.0, model->lattice[1][1] = ly, model->lattice[1][2] = 0.0;
    model->lattice[2][0] = 0.0, model->lattice[2][1] = 0.0, model->lattice[2][2] = 1.0;

    model->n_hop = 9*lx*ly;
    model->hop = (rs_hopping_t*)calloc(model -> n_hop,sizeof(rs_hopping_t));
    index_t elem = 0;
    for(index_t x = 0; x < lx; ++x)
    for(index_t y = 0; y < ly; ++y){
        index_t id = y+ly*x;
        model->hop[elem].t = mu;
        model->hop[elem].o1 = id;
        model->hop[elem].o2 = id;
        model -> positions[id][0] = (double)x;
        model -> positions[id][1] = (double)y;
        model -> positions[id][2] = 0.0;
        elem+=1;
    }
    for(index_t x = 0; x < lx; ++x)
    for(index_t y = 0; y < ly; ++y)
    for(index_t xp = 0; xp < lx; ++xp)
    for(index_t yp = 0; yp < ly; ++yp)
    for(index_t nx = -2; nx < 3; ++nx)
    for(index_t ny = -2; ny < 3; ++ny) {
        index_t idf = y+ly*x;
        index_t idt = yp+ly*xp;
        double from[2] = {(double)x,(double)y};
        double to[2] = {(double)(xp-lx*nx),(double)(yp-ly*ny)};

        if(fabs(1.-sqrt(POW2(from[0]-to[0]) + POW2(from[1]-to[1]))) < 1e-5) {
            model->hop[elem].t = -t,model->hop[elem].o1 = idf,model->hop[elem].o2 = idt;
            model->hop[elem].R[0] = nx,model->hop[elem].R[1] = ny;
            elem += 1;
        }
        if(fabs(sqrt(2.)-sqrt(POW2(from[0]-to[0]) + POW2(from[1]-to[1]))) < 1e-5) {
            model->hop[elem].t = -tp,model->hop[elem].o1 = idf,model->hop[elem].o2 = idt;
            model->hop[elem].R[0] = nx,model->hop[elem].R[1] = ny;
            elem += 1;
        }
    }

    model->n_ibz_path = 4;
    model->ibz_path[1][0] = 0.5;
    model->ibz_path[2][0] = 0.5;
    model->ibz_path[2][1] = 0.5;

    model->n_vert = model->n_orb;
    model->vert = (rs_vertex_t*)calloc(model -> n_vert,sizeof(rs_vertex_t));
    for(index_t o1 = 0; o1 < model->n_orb; ++o1) {
        model->vert[o1].chan = 'D', model->vert[o1].V = U;
        model->vert[o1].o1 = o1, model->vert[o1].o2 = o1;
    }
    diverge_model_internals_common( model );
    return model;
}


diverge_model_t* gen_square_lat_hub_sym(index_t nkx, index_t nky,
        index_t nkxf, index_t nkyf,
        double t, double tp, double U, double mu) {
    diverge_model_t* mod = diverge_model_init();
    mod->n_orb = 1;
    mod->n_spin = 1;
    mod->SU2 = true;
    mod->lattice[0][0] = mod->lattice[1][1] = mod->lattice[2][2] = 1.0;
    mod->hop = (rs_hopping_t*)calloc(100, sizeof(rs_hopping_t));
    int s = 0;
    mod->hop[mod->n_hop++] = rs_hopping_t{ {0,0,0}, 0,0,s,s, -mu };
    mod->hop[mod->n_hop++] = rs_hopping_t{ { 1,0,0}, 0,0,s,s, t };
    mod->hop[mod->n_hop++] = rs_hopping_t{ {-1,0,0}, 0,0,s,s, t };
    mod->hop[mod->n_hop++] = rs_hopping_t{ {0, 1,0}, 0,0,s,s, t };
    mod->hop[mod->n_hop++] = rs_hopping_t{ {0,-1,0}, 0,0,s,s, t };
    mod->hop[mod->n_hop++] = rs_hopping_t{ { 1, 1,0}, 0,0,s,s, tp };
    mod->hop[mod->n_hop++] = rs_hopping_t{ {-1, 1,0}, 0,0,s,s, tp };
    mod->hop[mod->n_hop++] = rs_hopping_t{ {-1,-1,0}, 0,0,s,s, tp };
    mod->hop[mod->n_hop++] = rs_hopping_t{ { 1,-1,0}, 0,0,s,s, tp };

    mod->nk[0] = nkx, mod->nk[1] = nky;
    mod->nkf[0] = nkxf, mod->nkf[1] = nkyf;

    mod->n_ibz_path = 0;

    mod->vert = (rs_vertex_t*)calloc(100, sizeof(rs_vertex_t));
    mod->vert[mod->n_vert++] = {'D', {0,0,0}, 0,0, 0,0,0,0, U};

    mod->n_sym = 8;
    site_descr_t sites[1];

    sites[0].n_functions = 1;
    sites[0].amplitude[0] = 1.;
    sites[0].function[0] = orb_s;

    index_t cnt = 0;
    mod->orb_symmetries = (complex128_t*)calloc(mod->n_sym
                            *POW2(mod->n_orb*mod->n_spin),sizeof(complex128_t));
    index_t symsize = POW2(mod->n_orb*mod->n_spin);
    sym_op_t curr_symm[3];
    curr_symm[0].type = 'R'; curr_symm[0].normal_vector[0] = 0.;
    curr_symm[0].normal_vector[1] = 0.;curr_symm[0].normal_vector[2] = 1.;
    // C6
    for(cnt = 0; cnt < 4; ++cnt) {
        curr_symm[0].angle = 90*cnt;
        if(curr_symm[0].angle > 181)
            curr_symm[0].angle -= 360;
        diverge_generate_symm_trafo(mod->n_spin,sites, mod->n_orb,
                curr_symm, 1,&(mod->rs_symmetries[cnt][0][0]),mod->orb_symmetries+cnt*symsize);
    }

    curr_symm[0].type = 'M';
    curr_symm[0].normal_vector[2] = 0.;
    curr_symm[0].normal_vector[0] = 0.;
    curr_symm[0].normal_vector[1] = 1.;
    diverge_generate_symm_trafo(mod->n_spin,sites, mod->n_orb,
            curr_symm, 1,&(mod->rs_symmetries[cnt][0][0]),mod->orb_symmetries+cnt*symsize);
    cnt++;
    curr_symm[0].normal_vector[0] = 1.;
    curr_symm[0].normal_vector[1] = 0.;
    diverge_generate_symm_trafo(mod->n_spin,sites, mod->n_orb,
            curr_symm, 1,&(mod->rs_symmetries[cnt][0][0]),mod->orb_symmetries+cnt*symsize);
    cnt++;

    curr_symm[0].normal_vector[0] = 1.;
    curr_symm[0].normal_vector[1] = 1.;
    diverge_generate_symm_trafo(mod->n_spin,sites, mod->n_orb,
            curr_symm, 1,&(mod->rs_symmetries[cnt][0][0]),mod->orb_symmetries+cnt*symsize);
    cnt++;
    curr_symm[0].normal_vector[0] = 1.;
    curr_symm[0].normal_vector[1] = -1.;
    diverge_generate_symm_trafo(mod->n_spin,sites, mod->n_orb,
            curr_symm, 1,&(mod->rs_symmetries[cnt][0][0]),mod->orb_symmetries+cnt*symsize);
    cnt++;

    diverge_model_internals_common( mod );
    return mod;
}

#endif // DIVERGE_SKIP_TESTS
