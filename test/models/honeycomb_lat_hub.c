#ifndef DIVERGE_SKIP_TESTS

#include "all.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/misc/generate_symmetries.h"
#include "../../src/misc/linalg.h"
LINALG_IMPLEMENT

diverge_model_t* gen_honeycomb_lat_hub( index_t nkx, index_t nky, index_t lx,
        index_t ly, double t, double tp, double U, double mu ) {
    diverge_model_t* model = diverge_model_init();
    model->n_orb = 2*lx*ly;
    model->SU2 = true;

    model->lattice[0][0] = 0.5*sqrt(3.)*lx, model->lattice[0][1] = -0.5*lx, model->lattice[0][2] = 0.0;
    model->lattice[1][0] = 0.5*sqrt(3.)*ly, model->lattice[1][1] = 0.5*ly, model->lattice[1][2] = 0.0;
    model->lattice[2][0] = 0.0, model->lattice[2][1] = 0.0, model->lattice[2][2] = 1.0;

    double bv1[2] = {0.5*sqrt(3.),-0.5};
    double bv2[2] = {0.5*sqrt(3.), 0.5};
    double pos1[3] = {sqrt(3.)/3.,0.,0.};
    double pos2[3] = {2.*sqrt(3.)/3.,0.,0.};

    model->n_hop = 20*lx*ly;
    model->hop = calloc(model -> n_hop,sizeof(rs_hopping_t));
    index_t elem = 0;
    for(index_t x = 0; x < lx; ++x)
    for(index_t y = 0; y < ly; ++y){
        index_t id = 0+2*y+2*ly*x;
        model->hop[elem].t = mu;
        model->hop[elem].o1 = id;
        model->hop[elem].o2 = id;
        elem+=1;
        model -> positions[id][0] = pos1[0] + bv1[0]*x + bv2[0]*y;
        model -> positions[id][1] = pos1[1] + bv1[1]*x + bv2[1]*y;

        id = 1+2*y+2*ly*x;
        model->hop[elem].t = mu;
        model->hop[elem].o1 = id;
        model->hop[elem].o2 = id;
        elem+=1;
        model -> positions[id][0] = pos2[0] + bv1[0]*x + bv2[0]*y;
        model -> positions[id][1] = pos2[1] + bv1[1]*x + bv2[1]*y;
    }

    model->nk[0] = nkx, model->nk[1] = nky;
    model->nkf[0] = 1, model->nkf[1] = 1;

    model->n_ibz_path = 4;
    model->ibz_path[1][0] = 0.5;
    model->ibz_path[2][0] = 2./3.;
    model->ibz_path[2][1] = 1./3.;

    for(index_t o1 = 0; o1 < model->n_orb; ++o1)
    for(index_t o2 = 0; o2 < model->n_orb; ++o2)
    for(index_t nx = -2; nx < 3; ++nx)
    for(index_t ny = -2; ny < 3; ++ny){
        double from[2] = {(double)model -> positions[o1][0],(double)model -> positions[o1][1]};
        double to[2] = {(double)(model -> positions[o2][0]+nx*model->lattice[0][0]+ny*model->lattice[1][0])
                        ,(double)(model -> positions[o2][1]+nx*model->lattice[0][1]+ny*model->lattice[1][1])};

        if(fabs(sqrt(1./3.)-sqrt(POW2(-from[0]+to[0]) + POW2(-from[1]+to[1]))) < 1e-5) {
            model->hop[elem].t = -t,model->hop[elem].o1 = o1,model->hop[elem].o2 = o2;
            model->hop[elem].R[0] = nx,model->hop[elem].R[1] = ny;
            elem += 1;
        }
        if(fabs(1.-sqrt(POW2(-from[0]+to[0]) + POW2(-from[1]+to[1]))) < 1e-5) {
            model->hop[elem].t = -tp,model->hop[elem].o1 = o1,model->hop[elem].o2 = o2;
            model->hop[elem].R[0] = nx,model->hop[elem].R[1] = ny;
            elem += 1;
        }
    }

    model->n_vert = 2*lx*ly;
    model->vert = calloc(model -> n_vert,sizeof(rs_vertex_t));
    for(index_t o1 = 0; o1 < model->n_orb; ++o1)
    {
        model->vert[o1].chan = 'D', model->vert[o1].V = U;
        model->vert[o1].o1 = o1, model->vert[o1].o2 = o1;
    }
    diverge_model_internals_common( model );
    return model;
}



diverge_model_t* gen_honeycomb_lat_hub_sym( index_t nkx, index_t nky,
        index_t nkxf, index_t nkyf,
        double t, double tp, double U, double mu) {
    diverge_model_t* model = diverge_model_init();
    model->n_orb = 2;
    model->SU2 = true;

    model->lattice[0][0] = 0.5*sqrt(3.), model->lattice[0][1] = -0.5, model->lattice[0][2] = 0.0;
    model->lattice[1][0] = 0.5*sqrt(3.), model->lattice[1][1] = 0.5, model->lattice[1][2] = 0.0;
    model->lattice[2][0] = 0.0, model->lattice[2][1] = 0.0, model->lattice[2][2] = 1.0;

    double pos1[3] = {sqrt(3.)/3.,0.,0.};
    double pos2[3] = {2.*sqrt(3.)/3.,0.,0.};

    model->n_hop = 20;
    model->hop = calloc(model -> n_hop,sizeof(rs_hopping_t));
    index_t elem = 0;
    index_t id = 0;
    model->hop[elem].t = mu;
    model->hop[elem].o1 = id;
    model->hop[elem].o2 = id;
    elem+=1;
    model -> positions[id][0] = pos1[0];
    model -> positions[id][1] = pos1[1];

    id = 1;
    model->hop[elem].t = mu;
    model->hop[elem].o1 = id;
    model->hop[elem].o2 = id;
    elem+=1;
    model -> positions[id][0] = pos2[0];
    model -> positions[id][1] = pos2[1];


    model->nk[0] = nkx, model->nk[1] = nky;
    model->nkf[0] = nkxf, model->nkf[1] = nkyf;

    model->n_ibz_path = 4;
    model->ibz_path[1][0] = 0.5;
    model->ibz_path[2][0] = 2./3.;
    model->ibz_path[2][1] = 1./3.;

    for(index_t o1 = 0; o1 < model->n_orb; ++o1)
    for(index_t o2 = 0; o2 < model->n_orb; ++o2)
    for(index_t nx = -2; nx < 3; ++nx)
    for(index_t ny = -2; ny < 3; ++ny) {
        double from[2] = {(double)model -> positions[o1][0],(double)model -> positions[o1][1]};
        double to[2] = {(double)(model -> positions[o2][0]+nx*model->lattice[0][0]+ny*model->lattice[1][0])
                        ,(double)(model -> positions[o2][1]+nx*model->lattice[0][1]+ny*model->lattice[1][1])};

        if(fabs(sqrt(1./3.)-sqrt(POW2(-from[0]+to[0]) + POW2(-from[1]+to[1]))) < 1e-5) {
            model->hop[elem].t = -t,model->hop[elem].o1 = o1,model->hop[elem].o2 = o2;
            model->hop[elem].R[0] = nx,model->hop[elem].R[1] = ny;
            elem += 1;
        }
        if(fabs(1.-sqrt(POW2(-from[0]+to[0]) + POW2(-from[1]+to[1]))) < 1e-5) {
            model->hop[elem].t = -tp,model->hop[elem].o1 = o1,model->hop[elem].o2 = o2;
            model->hop[elem].R[0] = nx,model->hop[elem].R[1] = ny;
            elem += 1;
        }
    }

    model->n_vert = 2;
    model->vert = calloc(model -> n_vert,sizeof(rs_vertex_t));
    for(index_t o1 = 0; o1 < model->n_orb; ++o1)
    {
        model->vert[o1].chan = 'D', model->vert[o1].V = U;
        model->vert[o1].o1 = o1, model->vert[o1].o2 = o1;
    }

    model->n_sym = 12;
    site_descr_t sites[2] = {0};

    sites[0].n_functions = 1;
    sites[0].amplitude[0] = 1.;
    sites[0].function[0] = orb_s;
    sites[1].n_functions = 1;
    sites[1].amplitude[0] = 1.;
    sites[1].function[0] = orb_s;

    index_t cnt = 0;
    model->orb_symmetries = calloc(20*POW2(model->n_orb*model->n_spin),sizeof(complex128_t));
    index_t symsize = POW2(model->n_orb*model->n_spin);
    sym_op_t curr_symm[3] = {0};

    curr_symm[0].type = 'M'; curr_symm[0].normal_vector[0] = 0.;
    curr_symm[0].normal_vector[1] = 1.;curr_symm[0].normal_vector[2] = 0.;

    diverge_generate_symm_trafo(model->n_spin,sites, model->n_orb,
            curr_symm, 1,&(model->rs_symmetries[cnt][0][0]),model->orb_symmetries+cnt*symsize);
    cnt ++;
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
    model->n_sym = cnt;
    diverge_model_internals_common( model );
    return model;
}


diverge_model_t* gen_honeycomb_lat_rash( index_t nkx, index_t nky, index_t lx,
        index_t ly, index_t nkxf, index_t nkyf ) {
    double t=1; double tp=0.1; double U=3;
    diverge_model_t* model = diverge_model_init();
    model->n_orb = 2*lx*ly;
    model->n_spin = 2;
    model->SU2 = false;

    model->lattice[0][0] = 0.5*sqrt(3.)*lx, model->lattice[0][1] = -0.5*lx, model->lattice[0][2] = 0.0;
    model->lattice[1][0] = 0.5*sqrt(3.)*ly, model->lattice[1][1] = 0.5*ly, model->lattice[1][2] = 0.0;
    model->lattice[2][0] = 0.0, model->lattice[2][1] = 0.0, model->lattice[2][2] = 1.0;

    double bv1[2] = {0.5*sqrt(3.),-0.5};
    double bv2[2] = {0.5*sqrt(3.), 0.5};
    double pos1[3] = {sqrt(3.)/3.,0.,0.};
    double pos2[3] = {2.*sqrt(3.)/3.,0.,0.};

    model->hop = calloc(128*lx*ly,sizeof(rs_hopping_t));
    index_t elem = 0;
    for(index_t x = 0; x < lx; ++x)
    for(index_t y = 0; y < ly; ++y){
        index_t id = 0+2*y+2*ly*x;
        model -> positions[id][0] = pos1[0] + bv1[0]*x + bv2[0]*y;
        model -> positions[id][1] = pos1[1] + bv1[1]*x + bv2[1]*y;

        id = 1+2*y+2*ly*x;
        model -> positions[id][0] = pos2[0] + bv1[0]*x + bv2[0]*y;
        model -> positions[id][1] = pos2[1] + bv1[1]*x + bv2[1]*y;
    }

    model->nk[0] = nkx, model->nk[1] = nky;
    model->nkf[0] = nkxf, model->nkf[1] = nkyf;

    model->n_ibz_path = 4;
    model->ibz_path[1][0] = 0.5;
    model->ibz_path[2][0] = 2./3.;
    model->ibz_path[2][1] = 1./3.;
    claM2cd_t pauli_x = {{{0,1},{1,0}}}, pauli_y = {{{0,-I},{I,0}}};

    for(index_t o1 = 0; o1 < model->n_orb; ++o1)
    for(index_t o2 = 0; o2 < model->n_orb; ++o2)
    for(index_t nx = -3; nx < 4; ++nx)
    for(index_t ny = -3; ny < 4; ++ny){
        claV3d_t po1 = claV3d_map(model->positions[o1]),
                 po2 = claV3d_map(model->positions[o2]),
                 Rx = claV3d_map(model->lattice[0]),
                 Ry = claV3d_map(model->lattice[1]);
        claV3d_t bond = claV3d_add( claV3d_sub(po2,po1), claV3d_add(claV3d_scale(Rx,nx), claV3d_scale(Ry,ny)) );
        double bond_length = claV3d_norm(bond);
        if(fabs(sqrt(1./3.)-bond_length) < 1e-5) {
            model->hop[elem++] = (rs_hopping_t) {{nx,ny,0}, o1,o2,0,0, t };
            model->hop[elem++] = (rs_hopping_t) {{nx,ny,0}, o1,o2,1,1, t };
        }
        if(fabs(1.-bond_length) < 1e-5) {
            model->hop[elem++] = (rs_hopping_t) {{nx,ny,0}, o1,o2,0,0, -tp };
            model->hop[elem++] = (rs_hopping_t) {{nx,ny,0}, o1,o2,1,1, -tp };
        }
        claM2cd_t pauli_bond = claM2cd_sub( claM2cd_scale(pauli_y, bond.v[0]),
                                            claM2cd_scale(pauli_x, bond.v[1]) );
        if (fabs(sqrt(1./3.)-bond_length) < 1e-5)
        for (int s1=0; s1<2; ++s1)
        for (int s2=0; s2<2; ++s2) {
            model->hop[elem++] = (rs_hopping_t) {{nx,ny,0}, o1,o2,s1,s2, -I*pauli_bond.m[s2][s1]*t*0.3 };
        }
        if (fabs(1.-bond_length) < 1e-5)
        for (int s1=0; s1<2; ++s1)
        for (int s2=0; s2<2; ++s2) {
            model->hop[elem++] = (rs_hopping_t) {{nx,ny,0}, o1,o2,s1,s2, -I*pauli_bond.m[s2][s1]*tp*0.1 };
        }
    }
    model->n_hop = elem;

    model->n_vert = 2*lx*ly;
    model->vert = calloc(model -> n_vert,sizeof(rs_vertex_t));
    for(index_t o1 = 0; o1 < model->n_orb; ++o1)
    {
        model->vert[o1].chan = 'D', model->vert[o1].V = U;
        model->vert[o1].s1 = -1,model->vert[o1].o1 = o1, model->vert[o1].o2 = o1;
    }
    diverge_model_internals_common( model );
    return model;
}

#endif // DIVERGE_SKIP_TESTS
