#include "old.h"
#include "../../src/misc/mpi_functions.h"
#include "../../src/diverge_model.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/diverge_Eigen3.hpp"
#include "../../src/diverge_patching.h"

typedef Eigen::Matrix<double,3,3,Eigen::RowMajor> CMat3d;

diverge_model_t* model_examples_hubbard( bool grid ) {
    diverge_model_t* mod = diverge_model_init();
    sprintf(mod->name, "test_diverge_hubbard");
    mod->nk[0] = mod->nk[1] = grid ? 6 : 200;
    mod->nkf[0] = mod->nkf[1] = grid ? 5 : 1;

    mod->lattice[0][0] = 1.0;
    mod->lattice[1][1] = 1.0;
    mod->lattice[2][2] = 1.0;
    mod->n_orb = 1;

    mod->n_hop = 4;
    mod->hop = (rs_hopping_t*)malloc(sizeof(rs_hopping_t)*4);
    mod->hop[0] = {{ 1,0,0}, 0,0,0,0, 1};
    mod->hop[1] = {{-1,0,0}, 0,0,0,0, 1};
    mod->hop[2] = {{0, 1,0}, 0,0,0,0, 1};
    mod->hop[3] = {{0,-1,0}, 0,0,0,0, 1};
    mod->SU2 = true;
    mod->n_spin = 1;

    mod->n_sym = 8;
    mod->orb_symmetries = (complex128_t*)malloc(sizeof(complex128_t)*8);
    for (index_t i=0; i<8; ++i) mod->orb_symmetries[i] = 1.0;
    for (index_t s=0; s<mod->n_sym; ++s) {
        Map<CMat3d> symmat((double*)&(mod->rs_symmetries[s]));
        switch (s) {
            case 0: symmat = Mat3d::Identity(); break;
            case 1: symmat = Mat3d::Identity(); symmat.block<2,2>(0,0) = Rot2d(M_PI).toRotationMatrix(); break;
            case 2: symmat = Mat3d::Identity(); symmat.block<2,2>(0,0) = Rot2d(0.5*M_PI).toRotationMatrix(); break;
            case 3: symmat = Mat3d::Identity(); symmat.block<2,2>(0,0) = Rot2d(1.5*M_PI).toRotationMatrix(); break;
            case 4: symmat = Mat3d::Identity(); symmat(0,0) = -1; break;
            case 5: symmat = Mat3d::Identity(); symmat(1,1) = -1; break;
            case 6: symmat = Mat3d::Identity(); symmat(0,0) = -1;
                    symmat.block<2,2>(0,0) = Rot2d(0.25*M_PI)*symmat.block<2,2>(0,0)*Rot2d(-0.25*M_PI); break;
            case 7: symmat = Mat3d::Identity(); symmat(1,1) = -1;
                    symmat.block<2,2>(0,0) = Rot2d(0.25*M_PI)*symmat.block<2,2>(0,0)*Rot2d(-0.25*M_PI); break;
            default: break;
        }
    }

    mod->n_vert = 1;
    mod->vert = (rs_vertex_t*)malloc(sizeof(rs_vertex_t));
    mod->vert[0] = {'D', {0,0,0}, 0,0,0,0,0,0, 3};

    if (diverge_model_validate( mod ))
        diverge_mpi_exit(EXIT_FAILURE);

    mod->n_ibz_path = 4;
    mod->ibz_path[1][0] = 0.5;
    mod->ibz_path[2][0] = 0.5;
    mod->ibz_path[2][1] = 0.5;

    diverge_model_internals_common( mod );
    if (grid) {
        diverge_model_internals_grid( mod );
    } else {
        vector<index_t> patches = diverge_patching_find_fs_pts( mod, NULL, 1, 4, 100 );
        // adding some high symmetry points
        patches.push_back(0);
        patches.push_back(mod->nk[0]/2*mod->nk[1]+mod->nk[1]/2);
        patches.push_back(mod->nk[0]/2*mod->nk[1]);
        patches.push_back(mod->nk[1]/2);
        mod->patching = diverge_patching_from_indices( mod, patches.data(), patches.size() );
        diverge_patching_autofine( mod, mod->patching, NULL, 1, 40, 0.1, 2.0, 0.5 );
        diverge_patching_symmetrize_refinement( mod, mod->patching );
        diverge_model_internals_patch( mod, -1 );
    }
    return mod;
}

static greensfunc_op_t patch_gpu_gf(const diverge_model_t* model,
        complex128_t Lambda, gf_complex_t* buf) {
    (void)model;
    (void)Lambda;
    (void)buf;
    return greensfunc_op_gpu;
}

diverge_model_t* model_examples_kagome( double U, double V ) {
    diverge_model_t* mod = diverge_model_init();

    index_t nk_flat = 2400;
    index_t np_ibz = 6;
    sprintf(mod->name, "kagome_patch");
    mod->nk[0] = mod->nk[1] = nk_flat;
    mod->nkf[0] = mod->nkf[1] = 1;

    Map<Mat3d> lat((double*)mod->lattice);
    lat.col(0) = Vec3d(1,0,0);
    lat.col(1) = Vec3d(cos(M_PI/3.), sin(M_PI/3.), 0);
    lat.col(2) = Vec3d(0,0,1);
    mod->n_orb = 3;

    Map<Mat3d> pos((double*)mod->positions);
    pos.col(0) = Vec3d(0,0,0);
    pos.col(1) = 0.5*lat.col(0);
    pos.col(2) = 0.5*lat.col(1);
    Vec3d shift = pos.col(1) + pos.col(2);
    for (int i=0; i<3; ++i)
        pos.col(i) -= shift; // shift to center!

    mod->SU2 = true;
    mod->n_spin = 1;
    mod->n_hop = 0;
    mod->hop = (rs_hopping_t*)malloc(sizeof(rs_hopping_t)*15);
    mod->hop[mod->n_hop++] = {{0,0,0}, 0,0, 0,0, 0.02};
    mod->hop[mod->n_hop++] = {{0,0,0}, 1,1, 0,0, 0.02};
    mod->hop[mod->n_hop++] = {{0,0,0}, 2,2, 0,0, 0.02};
    mod->hop[mod->n_hop++] = {{ 0, 0,0}, 0,1, 0,0, -1};
    mod->hop[mod->n_hop++] = {{ 0, 0,0}, 0,2, 0,0, -1};
    mod->hop[mod->n_hop++] = {{-1, 0,0}, 0,1, 0,0, -1};
    mod->hop[mod->n_hop++] = {{ 0,-1,0}, 0,2, 0,0, -1};
    mod->hop[mod->n_hop++] = {{ 0, 0,0}, 1,0, 0,0, -1};
    mod->hop[mod->n_hop++] = {{ 0, 0,0}, 1,2, 0,0, -1};
    mod->hop[mod->n_hop++] = {{ 1, 0,0}, 1,0, 0,0, -1};
    mod->hop[mod->n_hop++] = {{ 1,-1,0}, 1,2, 0,0, -1};
    mod->hop[mod->n_hop++] = {{ 0, 0,0}, 2,0, 0,0, -1};
    mod->hop[mod->n_hop++] = {{ 0, 0,0}, 2,1, 0,0, -1};
    mod->hop[mod->n_hop++] = {{ 0, 1,0}, 2,0, 0,0, -1};
    mod->hop[mod->n_hop++] = {{-1, 1,0}, 2,1, 0,0, -1};

    mod->n_sym = 12;
    mod->orb_symmetries = (complex128_t*)malloc(sizeof(complex128_t)*12*3*3);
    for (index_t i=0; i<12; ++i)
        Map<Mat3cd>((complex128_t*)mod->orb_symmetries+9*i) = Mat3cd::Identity();
    for (index_t s=0; s<mod->n_sym; ++s) {
        Map<CMat3d> symmat((double*)&(mod->rs_symmetries[s]));
        symmat = Mat3d::Identity();
        if (s<6)
            symmat.block<2,2>(0,0) = Rot2d(2.0*M_PI/6.0 * (double)s).toRotationMatrix();
        else {
            Mat2d mirror = Mat2d::Identity();
            if (s<9) mirror(0,0) = -1;
            else     mirror(1,1) = -1;
            double theta = 2.0*M_PI/3.0*(s%3);
            symmat.block<2,2>(0,0) = Rot2d(theta) * mirror * Rot2d(-theta);
        }
    }

    mod->n_vert = 0;
    mod->vert = (rs_vertex_t*)malloc(sizeof(rs_vertex_t)*15);
    mod->vert[mod->n_vert++] = {'D', {0,0,0}, 0,0, -1,0,0,0, U};
    mod->vert[mod->n_vert++] = {'D', {0,0,0}, 1,1, -1,0,0,0, U};
    mod->vert[mod->n_vert++] = {'D', {0,0,0}, 2,2, -1,0,0,0, U};
    mod->vert[mod->n_vert++] = {'D', { 0, 0,0}, 0,1, -1,0,0,0, V};
    mod->vert[mod->n_vert++] = {'D', { 0, 0,0}, 0,2, -1,0,0,0, V};
    mod->vert[mod->n_vert++] = {'D', {-1, 0,0}, 0,1, -1,0,0,0, V};
    mod->vert[mod->n_vert++] = {'D', { 0,-1,0}, 0,2, -1,0,0,0, V};
    mod->vert[mod->n_vert++] = {'D', { 0, 0,0}, 1,0, -1,0,0,0, V};
    mod->vert[mod->n_vert++] = {'D', { 0, 0,0}, 1,2, -1,0,0,0, V};
    mod->vert[mod->n_vert++] = {'D', { 1, 0,0}, 1,0, -1,0,0,0, V};
    mod->vert[mod->n_vert++] = {'D', { 1,-1,0}, 1,2, -1,0,0,0, V};
    mod->vert[mod->n_vert++] = {'D', { 0, 0,0}, 2,0, -1,0,0,0, V};
    mod->vert[mod->n_vert++] = {'D', { 0, 0,0}, 2,1, -1,0,0,0, V};
    mod->vert[mod->n_vert++] = {'D', { 0, 1,0}, 2,0, -1,0,0,0, V};
    mod->vert[mod->n_vert++] = {'D', {-1, 1,0}, 2,1, -1,0,0,0, V};

    mod->n_ibz_path = 4;
    mod->ibz_path[1][0] = 0.5;
    mod->ibz_path[2][0] = 2./3.;
    mod->ibz_path[2][1] = 1./3.;

    mod->gfill = &patch_gpu_gf;

    if (diverge_model_validate( mod ))
        diverge_mpi_exit(EXIT_FAILURE);

    diverge_model_internals_common( mod );
    vector<index_t> patches = diverge_patching_find_fs_pts( mod, NULL, mod->n_orb, np_ibz, 100 );
    mod->patching = diverge_patching_from_indices( mod, patches.data(), patches.size() );
    diverge_patching_autofine( mod, mod->patching, NULL, mod->n_orb, 40, 0.3, 5.0, 0.5 );
    diverge_patching_symmetrize_refinement( mod, mod->patching );
    diverge_model_internals_patch( mod, -1 );
    return mod;
}

diverge_model_t* model_examples_graphene( void ) {
    diverge_model_t* mod = diverge_model_init();

    index_t nk_flat = 6,
            //nk_flat_refine = 25;
            nk_flat_refine = 1;
    sprintf(mod->name, "graphene");
    mod->nk[0] = mod->nk[1] = nk_flat;
    mod->nkf[0] = mod->nkf[1] = nk_flat_refine;

    Map<Mat3d> lat((double*)mod->lattice);
    lat.col(0) = Vec3d(1,0,0);
    lat.col(1) = Vec3d(cos(M_PI/3.), sin(M_PI/3.), 0);
    lat.col(2) = Vec3d(0,0,1);

    mod->n_orb = 2;
    Map<Mat3d> pos((double*)mod->positions);
    pos.col(0) = (lat.col(0) + lat.col(1))/3.0;
    pos.col(1) = (lat.col(0) + lat.col(1))*2.0/3.0;

    mod->SU2 = true;
    mod->n_spin = 1;
    mod->n_hop = 0;
    mod->hop = (rs_hopping_t*)malloc(sizeof(rs_hopping_t)*128);
    mod->hop[mod->n_hop++] = {{ 0, 0,0}, 0,1, 0,0, 1.0}; // t
    mod->hop[mod->n_hop++] = {{ 0,-1,0}, 0,1, 0,0, 1.0};
    mod->hop[mod->n_hop++] = {{-1, 0,0}, 0,1, 0,0, 1.0};
    mod->hop[mod->n_hop++] = {{ 0, 0,0}, 1,0, 0,0, 1.0};
    mod->hop[mod->n_hop++] = {{ 0, 1,0}, 1,0, 0,0, 1.0};
    mod->hop[mod->n_hop++] = {{ 1, 0,0}, 1,0, 0,0, 1.0};
    for (int o=0; o<2; ++o) {
        mod->hop[mod->n_hop++] = {{0,0,0}, o,o, 0,0, -0.2}; // mu
        mod->hop[mod->n_hop++] = {{ 1, 0,0}, o,o, 0,0, 0.1}; // tp
        mod->hop[mod->n_hop++] = {{ 0, 1,0}, o,o, 0,0, 0.1};
        mod->hop[mod->n_hop++] = {{-1, 1,0}, o,o, 0,0, 0.1};
        mod->hop[mod->n_hop++] = {{-1, 0,0}, o,o, 0,0, 0.1};
        mod->hop[mod->n_hop++] = {{ 0,-1,0}, o,o, 0,0, 0.1};
        mod->hop[mod->n_hop++] = {{ 1,-1,0}, o,o, 0,0, 0.1};
    }

    mod->n_sym = 12;
    mod->orb_symmetries = (complex128_t*)malloc(sizeof(complex128_t)*12*2*2);
    for (index_t i=0; i<12; ++i)
        Map<Mat2cd>((complex128_t*)mod->orb_symmetries+4*i) = Mat2cd::Identity();
    for (index_t s=0; s<mod->n_sym; ++s) {
        Map<CMat3d> symmat((double*)&(mod->rs_symmetries[s]));
        symmat = Mat3d::Identity();
        if (s<6)
            symmat.block<2,2>(0,0) = Rot2d(2.0*M_PI/6.0 * (double)s).toRotationMatrix();
        else {
            Mat2d mirror = Mat2d::Identity();
            if (s<9) mirror(0,0) = -1;
            else     mirror(1,1) = -1;
            double theta = 2.0*M_PI/3.0*(s%3);
            symmat.block<2,2>(0,0) = Rot2d(theta) * mirror * Rot2d(-theta);
        }
    }

    mod->n_vert = 0;
    mod->vert = (rs_vertex_t*)malloc(sizeof(rs_vertex_t)*15);
    mod->vert[mod->n_vert++] = {'D', {0,0,0}, 0,0, -1,0,0,0, 3.0};
    mod->vert[mod->n_vert++] = {'D', {0,0,0}, 1,1, -1,0,0,0, 3.0};

    mod->n_ibz_path = 4;
    mod->ibz_path[1][0] = 0.5;
    mod->ibz_path[2][0] = 2./3.;
    mod->ibz_path[2][1] = 1./3.;

    if (diverge_model_validate( mod ))
        diverge_mpi_exit(EXIT_FAILURE);

    diverge_model_internals_common( mod );
    diverge_model_internals_grid( mod );
    vector<index_t> patches(mod->nk[0]*mod->nk[1]);
    for (index_t i=0; i<mod->nk[0]*mod->nk[1]; ++i) patches[i] = i;
    mod->patching = diverge_patching_from_indices( mod, patches.data(), patches.size() );
    diverge_model_internals_patch( mod, -1 );
    return mod;
}
