#include "model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../../src/diverge_Eigen3.hpp"
#include "../../src/misc/generate_symmetries.h"


diverge_model_t* gen_square_lat_rashba(index_t nkx, index_t nky, double t,
        double tp, double U, double mu, double alpha) {
    diverge_model_t* mod = diverge_model_init();
    mod->n_orb = 1;
    mod->n_spin = 2;
    mod->SU2 = false; // enforce exchange symmetry

    mod->lattice[0][0] = mod->lattice[1][1] = mod->lattice[2][2] = 1.0;
    mod->hop = (rs_hopping_t*)calloc(100, sizeof(rs_hopping_t));
    for (int s=0; s<2; ++s) {
        mod->hop[mod->n_hop++] = rs_hopping_t{ {0,0,0}, 0,0,s,s, -mu };
        mod->hop[mod->n_hop++] = rs_hopping_t{ { 1,0,0}, 0,0,s,s, t };
        mod->hop[mod->n_hop++] = rs_hopping_t{ {-1,0,0}, 0,0,s,s, t };
        mod->hop[mod->n_hop++] = rs_hopping_t{ {0, 1,0}, 0,0,s,s, t };
        mod->hop[mod->n_hop++] = rs_hopping_t{ {0,-1,0}, 0,0,s,s, t };
        mod->hop[mod->n_hop++] = rs_hopping_t{ { 1, 1,0}, 0,0,s,s, tp };
        mod->hop[mod->n_hop++] = rs_hopping_t{ {-1, 1,0}, 0,0,s,s, tp };
        mod->hop[mod->n_hop++] = rs_hopping_t{ {-1,-1,0}, 0,0,s,s, tp };
        mod->hop[mod->n_hop++] = rs_hopping_t{ { 1,-1,0}, 0,0,s,s, tp };
    }
    Mat2cd pauli_x; pauli_x << 0,1,1,0;
    Mat2cd pauli_y; pauli_y << 0,-I128,I128,0;
    for (int Rx=-1; Rx<=1; ++Rx)
    for (int Ry=-1; Ry<=1; ++Ry) {
        Vec3d bond = Rx * Map<Vec3d>((double*)(mod->lattice)) +
                     Ry * Map<Vec3d>((double*)(mod->lattice)+3);
        Mat2cd pauli_bond = bond(0) * pauli_y - bond(1) * pauli_x;
        double t_alpha = 0.0;
        if (fabs(bond.norm() - 1.0) < 1.e-7) t_alpha = t * alpha;
        if (fabs(bond.norm() - sqrt(2.0)) < 1.e-7) t_alpha = tp * alpha;

        for (int s1=0; s1<2; ++s1)
        for (int s2=0; s2<2; ++s2)
            mod->hop[mod->n_hop++] = rs_hopping_t{ {Rx,Ry,0}, 0,0,s1,s2, -I128*pauli_bond(s2,s1)*t_alpha };
    }

    mod->nk[0] = nkx, mod->nk[1] = nky;
    mod->nkf[0] = 1, mod->nkf[1] = 1;

    mod->n_ibz_path = 4;
    mod->ibz_path[1][0] = 0.5;
    mod->ibz_path[2][0] = 0.5;
    mod->ibz_path[2][1] = 0.5;

    mod->vert = (rs_vertex_t*)calloc(100, sizeof(rs_vertex_t));
    mod->vert[mod->n_vert++] = {'D', {0,0,0}, 0,0, -1,0,0,0, U};
    diverge_model_internals_common( mod );
    return mod;
}

diverge_model_t* gen_square_lat_rashba_sym(index_t nkx, index_t nky,
        index_t nkxf, index_t nkyf, double t,
        double tp, double U, double mu, double alpha) {
    diverge_model_t* mod = diverge_model_init();
    mod->n_orb = 1;
    mod->n_spin = 2;
    mod->SU2 = false; // enforce exchange symmetry

    mod->lattice[0][0] = mod->lattice[1][1] = mod->lattice[2][2] = 1.0;
    mod->hop = (rs_hopping_t*)calloc(100, sizeof(rs_hopping_t));
    for (int s=0; s<2; ++s) {
        mod->hop[mod->n_hop++] = rs_hopping_t{ {0,0,0}, 0,0,s,s, -mu };
        mod->hop[mod->n_hop++] = rs_hopping_t{ { 1,0,0}, 0,0,s,s, t };
        mod->hop[mod->n_hop++] = rs_hopping_t{ {-1,0,0}, 0,0,s,s, t };
        mod->hop[mod->n_hop++] = rs_hopping_t{ {0, 1,0}, 0,0,s,s, t };
        mod->hop[mod->n_hop++] = rs_hopping_t{ {0,-1,0}, 0,0,s,s, t };
        mod->hop[mod->n_hop++] = rs_hopping_t{ { 1, 1,0}, 0,0,s,s, tp };
        mod->hop[mod->n_hop++] = rs_hopping_t{ {-1, 1,0}, 0,0,s,s, tp };
        mod->hop[mod->n_hop++] = rs_hopping_t{ {-1,-1,0}, 0,0,s,s, tp };
        mod->hop[mod->n_hop++] = rs_hopping_t{ { 1,-1,0}, 0,0,s,s, tp };
    }
    Mat2cd pauli_x; pauli_x << 0,1,1,0;
    Mat2cd pauli_y; pauli_y << 0,-I128,I128,0;
    for (int Rx=-1; Rx<=1; ++Rx)
    for (int Ry=-1; Ry<=1; ++Ry) {
        Vec3d bond = Rx * Map<Vec3d>((double*)(mod->lattice)) +
                     Ry * Map<Vec3d>((double*)(mod->lattice)+3);
        Mat2cd pauli_bond =  bond(0) * pauli_y - bond(1) * pauli_x;
        double t_alpha = 0.0;
        if (fabs(bond.norm() - 1.0) < 1.e-7) t_alpha = t * alpha;
        if (fabs(bond.norm() - sqrt(2.0)) < 1.e-7) t_alpha = tp * alpha;

        for (int s1=0; s1<2; ++s1)
        for (int s2=0; s2<2; ++s2)
            mod->hop[mod->n_hop++] = rs_hopping_t{ {Rx,Ry,0}, 0,0,s1,s2, -I128*pauli_bond(s2,s1)*t_alpha };
    }

    mod->nk[0] = nkx, mod->nk[1] = nky;
    mod->nkf[0] = nkxf, mod->nkf[1] = nkyf;

    mod->n_ibz_path = 4;
    mod->ibz_path[1][0] = 0.5;
    mod->ibz_path[2][0] = 0.5;
    mod->ibz_path[2][1] = 0.5;

    mod->vert = (rs_vertex_t*)calloc(100, sizeof(rs_vertex_t));
    mod->vert[mod->n_vert++] = {'D', {0,0,0}, 0,0, -1,0,0,0, U};

    index_t cnt = 0;
    mod->orb_symmetries = (complex128_t*)calloc(20
                            *POW2(mod->n_orb*mod->n_spin),sizeof(complex128_t));
    index_t symsize = POW2(mod->n_orb*mod->n_spin);
    sym_op_t curr_symm[3];
    site_descr_t sites[1];

    sites[0].n_functions = 1;
    sites[0].amplitude[0] = 1.;
    sites[0].function[0] = orb_s;

    curr_symm[0].type = 'R';
    curr_symm[0].normal_vector[2] = 1.;
    curr_symm[0].normal_vector[0] = 0.;
    curr_symm[0].normal_vector[1] = 0.;

    // C6
    for(cnt = 0; cnt < 4; ++cnt) {
        curr_symm[0].angle = 90*cnt;
        curr_symm[0].angle -= curr_symm[0].angle > 181 ? 360: 0;
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
    mod->n_sym = cnt;



    diverge_model_internals_common( mod );
    return mod;
}

