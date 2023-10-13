#include <diverge.h>
#include <diverge_Eigen3.hpp>

#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <iostream>

using std::vector;
using std::array;

diverge_model_t* kagome_model( index_t nk, index_t nkf ,
                            double U , double V  );



int main(int argc, char** argv) {
    diverge_init( &argc, &argv );

    char mode[128] = "tu";
    if (argc > 1) {
        if (!strcmp(argv[1], "-h")) {
            mpi_eprintf("usage: %s [mode]\n"
                        "    mode    default: tu. possible choices: tu, patch, or grid.\n",
                        argv[0]);
            return 0;
        } else {
            strcpy( mode, argv[1] );
        }
    } else {
        mpi_eprintf("print help with %s -h\n", argv[0]);
    }

    diverge_compilation_status();

    // misc
    bool patch_mode = !strcmp(mode,"patch"),
         tu_mode = !strcmp(mode,"tu");

    int nkc = 24,
        nkf = 25;
    double U = 1.1;
    double V = 1.1;
    if(patch_mode) {
        nkc = 24*25;
        nkf = 1;
    }
    diverge_model_t* m = kagome_model( nkc, nkf ,  U , V );

    char outfile[1024];
    sprintf( outfile, "%s_mod.dvg", __FILE__ );
    diverge_model_to_file( m, outfile );

    if(tu_mode) {
        // number = maximal allowed distance
        diverge_model_internals_tu( m, 1.01 );
    }else{
        // number = #patches in IBZ
        diverge_model_internals_patch( m, 2 );
    }

    diverge_flow_step_t* s = diverge_flow_step_init( m, mode, "PCD" );

    sprintf( outfile, "%s_flow.dat", __FILE__ );
    diverge_euler_t eu = diverge_euler_defaults_CPP();

    diverge_flow_step_euler( s, eu.Lambda, eu.dLambda );

    char line[2048];
    char* l = line;
    index_t t;
    for (t=0; t<2; ++t) {
        sprintf( l, "%s:%.2fs, ", diverge_flow_step_timing_descr(s,t), diverge_flow_step_timing(s,t) );
        l += strlen(l);
    }
    if (t != 0) l -= 2;
    sprintf(l, "\n");
    mpi_tim_printf( line );

    diverge_flow_step_free( s );
    diverge_model_free( m );
    diverge_finalize();
}


diverge_model_t* kagome_model( index_t nk, index_t nkf , double U , double V) {
    diverge_model_t* mod = diverge_model_init();
    sprintf(mod->name, "Kagome mod");

    // initialize lattice and number of momentum points - is 2d
    mod->nk[0] = mod->nk[1] = nk;
    mod->nkf[0] = mod->nkf[1] = nkf;
    mod->lattice[0][0] = 0.5*sqrt(3.);
    mod->lattice[0][1] = -0.5;
    mod->lattice[1][0] = 0.5*sqrt(3.);
    mod->lattice[1][1] = 0.5;
    mod->lattice[2][2] = 1.0;


    // initialize positions and number of orbitals
    mod->n_orb = 3;
    mod->SU2 = true;
    mod->n_spin = 1;
    mod->n_hop = 0;
    mod->positions[0][0] = 0.5*sqrt(3.);
    mod->positions[0][1] = 0.0;
    mod->positions[1][0] = 0.5*sqrt(3.)-mod->lattice[0][0]*0.5;
    mod->positions[1][1] = -mod->lattice[0][1]*0.5;
    mod->positions[2][0] = 0.5*sqrt(3.)+mod->lattice[1][0]*0.5;
    mod->positions[2][1] = +mod->lattice[1][1]*0.5;

    // generate nearest neighbor hoppings
    vector<rs_hopping_t> hop;
    rs_hopping_t helper;
    helper.s1 = 0, helper.s2 = 0;
    helper.t = 1.0;
    Map<Vec3d> r1 (mod->lattice[0], 3);
    Map<Vec3d> r2 (mod->lattice[1], 3);

    // distance definition is  c^{\dagger}_{o_2+R}c_{o_1}
    for(int rx = -3; rx < 4; ++rx)
    for(int ry = -3; ry < 4; ++ry) {
        for(int o1 = 0; o1 < mod->n_orb; ++ o1)
        for(int o2 = 0; o2 < mod->n_orb; ++ o2) {
            Map<Vec3d> p1 (mod->positions[o1], 3);
            Map<Vec3d> p2 (mod->positions[o2], 3);
            if( std::abs((p2-p1+rx*r1+ry*r2).norm() - 0.5) < 1e-7 ) {
                helper.o1 = o1, helper.o2 = o2, helper.R[0]=rx,helper.R[1]=ry;
                hop.push_back(helper);
            }
        }
    }
    mod->n_hop = hop.size();
    mod->hop = (rs_hopping_t*)malloc(mod->n_hop*sizeof(rs_hopping_t));
    memcpy(mod->hop,hop.data(),mod->n_hop*sizeof(rs_hopping_t));
    mpi_usr_printf( "using %li hopping elements\n", mod->n_hop );

    //generate interaction vertex - density density type
    vector<rs_vertex_t> Rvert;
    Rvert.reserve(12*10);
    rs_vertex_t helperv;
    helperv.chan = 'D';
    // s1 == -1 means that the interaction can be constructed by using the
    // crossing relations
    helperv.s1 = -1, helperv.R[0] = 0, helperv.R[1] = 0, helperv.R[2] = 0;
    for(int rx = -2; rx < 3; ++rx)
    for(int ry = -2; ry < 3; ++ry) {
        for(int o1 = 0; o1 < mod->n_orb; ++ o1)
        for(int o2 = 0; o2 < mod->n_orb; ++ o2) {
            Map<Vec3d> p1 (mod->positions[o1], 3);
            Map<Vec3d> p2 (mod->positions[o2], 3);
            if( std::abs((p2-p1+rx*r1+ry*r2).norm() - 0.5) < 1e-7 ) {
                helperv.o1 = o1, helperv.o2 = o2,helperv.R[0]=rx,helperv.R[1]=ry;
                helperv.V = V;
                Rvert.push_back(helperv);
            }
            if( (p2-p1+rx*r1+ry*r2).norm() < 1e-7 ) {
                helperv.o1 = o1;
                helperv.o2 = o2;
                helperv.V = U;
                Rvert.push_back(helperv);
            }
        }
    }

    mod->n_vert = Rvert.size();
    mpi_usr_printf( "using %li vertex elements\n", mod->n_vert );
    mod->vert = (rs_vertex_t*)calloc(mod->n_vert,sizeof(rs_vertex_t));
    memcpy( mod->vert, Rvert.data(), sizeof(rs_vertex_t)*mod->n_vert );

    mod->n_ibz_path = 4;
    mod->ibz_path[1][0] = 0.5;
    mod->ibz_path[2][0] = 0.5;
    mod->ibz_path[2][1] = 0.5;

    site_descr_t sites[3];
    sites[0].n_functions = 1;
    sites[0].amplitude[0] = 1.;
    sites[0].function[0] = orb_s;
    sites[1].n_functions = 1;
    sites[1].amplitude[0] = 1.;
    sites[1].function[0] = orb_s;
    sites[2].n_functions = 1;
    sites[2].amplitude[0] = 1.;
    sites[2].function[0] = orb_s;

    mod->orb_symmetries = (complex128_t*)calloc(20
                            *POW2(mod->n_orb*mod->n_spin),sizeof(complex128_t));
    index_t symsize = POW2(mod->n_orb*mod->n_spin);
    sym_op_t curr_symm[3];

    curr_symm[0].type = 'R'; curr_symm[0].normal_vector[0] = 0.;
    curr_symm[0].normal_vector[1] = 0.;curr_symm[0].normal_vector[2] = 1.;
    curr_symm[1].type = 'M'; curr_symm[1].normal_vector[0] = 1.;    sites[2].amplitude[0] = 1.;
    sites[2].function[0] = orb_s;

    mod->orb_symmetries = (complex128_t*)calloc(20
                            *POW2(mod->n_orb*mod->n_spin),sizeof(complex128_t));

    curr_symm[0].type = 'R'; curr_symm[0].normal_vector[0] = 0.;
    curr_symm[0].normal_vector[1] = 0.;curr_symm[0].normal_vector[2] = 1.;
    curr_symm[1].type = 'M'; curr_symm[1].normal_vector[0] = 1.;

    curr_symm[1].normal_vector[1] = 0.;curr_symm[1].normal_vector[2] = 0.;
    curr_symm[2].type = 'R'; curr_symm[2].normal_vector[0] = 0.;
    curr_symm[2].normal_vector[1] = 0.;curr_symm[2].normal_vector[2] = 1.;
    int cnt;
    // C6
    for(cnt = 0; cnt < 6; ++cnt) {
        curr_symm[0].angle = 60*cnt;
        diverge_generate_symm_trafo(mod->n_spin,sites, mod->n_orb,
                curr_symm, 1,&(mod->rs_symmetries[cnt][0][0]),mod->orb_symmetries+cnt*symsize);
    }

    // mirror planes
    for(cnt = 6; cnt < 9; ++cnt) {
        curr_symm[0].angle = -120*(cnt-6);
        curr_symm[2].angle = 120*(cnt-6);
        diverge_generate_symm_trafo(mod->n_spin,sites, mod->n_orb,
                curr_symm, 3,&(mod->rs_symmetries[cnt][0][0]),mod->orb_symmetries+cnt*symsize);
    }
    curr_symm[1].type = 'M'; curr_symm[1].normal_vector[0] = 0.;
    curr_symm[1].normal_vector[1] = 1.;
    for(cnt = 9; cnt < 12; ++cnt) {
        curr_symm[0].angle = -120*(cnt-9);
        curr_symm[2].angle = 120*(cnt-9);
        diverge_generate_symm_trafo(mod->n_spin,sites, mod->n_orb,
                curr_symm, 3,&(mod->rs_symmetries[cnt][0][0]),mod->orb_symmetries+cnt*symsize);
    }
    mod->n_sym = cnt;

    // check that mod is set up properly
    if (diverge_model_validate( mod ))
        diverge_mpi_exit( EXIT_FAILURE );
    // initialize internals
    diverge_model_internals_common( mod );

    double err = diverge_symmetrize_2pt_fine( mod, diverge_model_internals_get_H(mod), nullptr);
    mpi_usr_printf( "symmetry error %2.5f\n", err );

    return mod;
}




