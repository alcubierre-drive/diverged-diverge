#include <diverge.h>
#include <diverge_Eigen3.hpp>

#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <iostream>

using std::vector;
using std::array;

diverge_model_t* SRO_model( index_t nk, index_t nkf , std::string path ,
                            double U , double J ,
                            index_t n_orb = 6, index_t n_spin = 2 );

int main(int argc, char** argv) {
    diverge_compilation_status();

    int nkc = 16,
        nkf = 1,
        n_orb = 6,
        n_spin = 2;
    double U = 2.3;
    double J = 0.174 * U;
    std::string path = "/home/jonas/Desktop/projects/W90files/SROtwi/Sr2RuO4_";
    std::string prefix[2] {"0deg_soc"};
    for(int nid = 0; nid < 1 ; ++nid) {
        std::string name = prefix[nid];
        n_orb = 6;

        std::string fpath = path + name + ".dat";
        diverge_mpi_init( &argc, &argv );
        diverge_model_t* m = SRO_model( nkc, nkf , fpath , U , J , n_orb , n_spin );

        char outfile[1024];
        sprintf( outfile, "%smod.dvg", name.c_str() );
        diverge_model_to_file( m, outfile, true );
        /*
        diverge_model_internals_tu( m, 1.1 );

        diverge_flow_step_t* s = diverge_flow_step_init( m, "tu", "PCD" );

        sprintf( outfile, "%sflow.dat", name.c_str() );
        FILE* f = fopen( outfile, "w" );
        double vertmax = 0;
        diverge_euler_t eu = diverge_euler_defaults_CPP();
        do {
            diverge_flow_step_euler( s, eu.Lambda, eu.dLambda );
            diverge_flow_step_vertmax( s, &vertmax );
            double loopmax[2]; diverge_flow_step_loopmax( s, loopmax );
            double chanmax[3]; diverge_flow_step_chanmax( s, chanmax );
            char line[2048];
            sprintf( line, "%.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e\n",
                    eu.Lambda, eu.dLambda, loopmax[0], loopmax[1], chanmax[0], chanmax[1], chanmax[2], vertmax );
            index_t n_timings = diverge_flow_step_ntimings( s );
            for(int i = 0; i < n_timings; ++i) {
                double timing = diverge_flow_step_timing( s, i );
                const char* ename = diverge_flow_step_timing_descr( s, i );
                std::cout <<  ename << "       " << timing << "s     ";
            }
            std::cout << "\n ";
            mpi_eprintf( "%s", line );
            fprintf( f, "%s", line );
            fflush( stderr );
        } while (diverge_euler_next( &eu, vertmax ));
        fclose( f );

        sprintf( outfile, "%sout.dvg", name.c_str() );
        diverge_postprocess_and_write( s, NULL, outfile );
        diverge_flow_step_free( s );
        */
        diverge_model_free( m );

        diverge_mpi_finalize();
    }
}


diverge_model_t* SRO_model( index_t nk, index_t nkf ,
                            std::string path , double U , double J ,
                            index_t n_orb , index_t n_spin ) {

    diverge_model_t* mod = diverge_model_init();
    sprintf(mod->name, "SRO mod");
    mod->nk[0] = mod->nk[1] = nk;
    mod->nkf[0] = mod->nkf[1] = nkf;


    mod->n_orb = n_orb;
    mod->SU2 = n_spin == 1 ? true : false;
    mod->n_spin = n_spin;
    mod->n_hop = 0;
    mod->lattice[0][0] = 1.0;
    mod->lattice[1][1] = 1.0;
    mod->lattice[2][2] = 1.0;
    if(n_orb == 3) {
        mod->positions[0][0] = 0.0;
        mod->positions[0][1] = 0.0;
        mod->positions[1][0] = 0.0;
        mod->positions[1][1] = 0.0;
        mod->positions[2][0] = 0.0;
        mod->positions[2][1] = 0.0;
    }else{
        mod->positions[0][0] = 0.25;
        mod->positions[0][1] = 0.75;
        mod->positions[1][0] = 0.25;
        mod->positions[1][1] = 0.75;
        mod->positions[2][0] = 0.25;
        mod->positions[2][1] = 0.75;

        mod->positions[3][0] = 0.75;
        mod->positions[3][1] = 0.25;
        mod->positions[4][0] = 0.75;
        mod->positions[4][1] = 0.25;
        mod->positions[5][0] = 0.75;
        mod->positions[5][1] = 0.25;
    }
    mod->hop = diverge_read_W90_C( path.c_str(), n_spin, &(mod->n_hop) );
    mpi_usr_printf( "using %li hopping elements\n", mod->n_hop );


    vector<rs_vertex_t> Rvert;
    Rvert.reserve(12*10);
    rs_vertex_t helper;
    helper.s1 = -1, helper.R[0] = 0, helper.R[1] = 0, helper.R[2] = 0;
    for(index_t o1 = 0; o1 < n_orb; ++o1)
    for(index_t o2 = 0; o2 < n_orb; ++o2) {
        if((o1 < 3 && o2 < 3) || (o1>2 && o2>2)) {
            helper.o1 = o1;
            helper.o2 = o2;
            if( o1 == o2 ) {
                helper.V = U;
                helper.chan = 'D';
                Rvert.push_back(helper);
            }else{
                // P channel
                helper.V = J;
                helper.chan = 'P';
                Rvert.push_back(helper);
                helper.chan = 'C';
                Rvert.push_back(helper);
                helper.V = U - 2.* J;
                helper.chan = 'D';
                Rvert.push_back(helper);
            }
        }
    }

    mod->n_vert = Rvert.size();
    mpi_usr_printf( "using %li vertex elements\n", mod->n_vert );
    mod->vert = (rs_vertex_t*)calloc(mod->n_vert,sizeof(rs_vertex_t));
    memcpy( mod->vert, Rvert.data(), sizeof(rs_vertex_t)*mod->n_vert );

    site_descr_t* sites = (site_descr_t*)calloc(6, sizeof(site_descr_t));

    sites[0].n_functions = 1;
    sites[0].amplitude[0] = 1.;
    sites[0].function[0] = orb_dxz;

    sites[1].n_functions = 1;
    sites[1].amplitude[0] = 1.;
    sites[1].function[0] = orb_dyz;

    sites[2].n_functions = 1;
    sites[2].amplitude[0] = 1.;
    sites[2].function[0] = orb_dxy;

    sites[3].n_functions = 1;
    sites[3].amplitude[0] = 1.;
    sites[3].function[0] = orb_dxz;

    sites[4].n_functions = 1;
    sites[4].amplitude[0] = 1.;
    sites[4].function[0] = orb_dyz;

    sites[5].n_functions = 1;
    sites[5].amplitude[0] = 1.;
    sites[5].function[0] = orb_dxy;

    if(mod->n_orb == 6) {
        sites[0].xaxis[0][0] = 1.;
        sites[0].xaxis[0][1] = -1.;
        sites[0].xaxis[0][2] = 0.;

        sites[1].xaxis[0][0] = 1.;
        sites[1].xaxis[0][1] = -1.;
        sites[1].xaxis[0][2] = 0.;

        sites[2].xaxis[0][0] = 1.;
        sites[2].xaxis[0][1] = -1.;
        sites[2].xaxis[0][2] = 0.;

        sites[3].xaxis[0][0] = 1.;
        sites[3].xaxis[0][1] = -1.;
        sites[3].xaxis[0][2] = 0.;

        sites[4].xaxis[0][0] = 1.;
        sites[4].xaxis[0][1] = -1.;
        sites[4].xaxis[0][2] = 0.;

        sites[5].xaxis[0][0] = 1.;
        sites[5].xaxis[0][1] = -1.;
        sites[5].xaxis[0][2] = 0.;
    }

    index_t symsize = POW2(mod->n_orb*mod->n_spin);
    mod->orb_symmetries = (complex128_t*)calloc(20*symsize,sizeof(complex128_t));
    sym_op_t curr_symm;
    curr_symm.type = 'R'; curr_symm.normal_vector[0] = 0.;
    curr_symm.normal_vector[1] = 0.; curr_symm.normal_vector[2] = 1.;
    curr_symm.angle = 0;
    diverge_generate_symm_trafo(mod->n_spin,sites, mod->n_orb,
                &curr_symm, 1,&(mod->rs_symmetries[mod->n_sym][0][0]),
                                    mod->orb_symmetries+mod->n_sym*symsize);
    mod->n_sym++;

    curr_symm.angle = 180;
    diverge_generate_symm_trafo(mod->n_spin,sites, mod->n_orb,
                &curr_symm, 1,&(mod->rs_symmetries[mod->n_sym][0][0]),
                                    mod->orb_symmetries+mod->n_sym*symsize);
    mod->n_sym++;

    curr_symm.angle = -180;
    diverge_generate_symm_trafo(mod->n_spin,sites, mod->n_orb,
                &curr_symm, 1,&(mod->rs_symmetries[mod->n_sym][0][0]),
                                    mod->orb_symmetries+mod->n_sym*symsize);
    mod->n_sym++;
    /*
    curr_symm.angle = 90;
    diverge_generate_symm_trafo(mod->n_spin,sites, mod->n_orb,
                &curr_symm, 1,&(mod->rs_symmetries[mod->n_sym][0][0]),
                                    mod->orb_symmetries+mod->n_sym*symsize);
    mod->n_sym++;

    curr_symm.angle = -90;
    diverge_generate_symm_trafo(mod->n_spin,sites, mod->n_orb,
                &curr_symm, 1,&(mod->rs_symmetries[mod->n_sym][0][0]),
                                    mod->orb_symmetries+mod->n_sym*symsize);
    mod->n_sym++;
    */
    mod->n_ibz_path = 4;
    mod->ibz_path[1][0] = 0.5;
    mod->ibz_path[2][0] = 0.5;
    mod->ibz_path[2][1] = 0.5;

    if (diverge_model_validate( mod ))
        diverge_mpi_exit(EXIT_FAILURE);

    diverge_model_internals_common( mod );
    double err = diverge_symmetrize_2pt_fine( mod, diverge_model_internals_get_H(mod), nullptr);
    mpi_usr_printf( "error %2.5f\n", err );
    free(sites);
    return mod;
}

