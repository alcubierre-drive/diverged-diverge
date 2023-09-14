#include <diverge.h>

// Example for a multi orbital system with Hubbard
// Kanamori interactions in 1D

static const double U = 2;
static const double J = 0.15;

int main( int argc, char** argv ) {
    // init
    diverge_init( &argc, &argv );

    // other init
    diverge_compilation_status();
    diverge_model_t* m = diverge_model_init();
    strcpy(m->name, __FILE__ );
    m->nk[0] = 100;
    m->nkf[0] = 15;
    m->lattice[0][0] = m->lattice[1][1] = m->lattice[2][2] = 1.0;
    m->n_orb = 3;
    m->SU2 = m->n_spin = 1;
    m->ibz_path[1][0] = 1.0;
    m->n_ibz_path = 4;

    m->hop = diverge_mem_alloc_rs_hopping_t(128);
    m->hop[m->n_hop++] = (rs_hopping_t){.R={ 1, 0,0}, .o1=0, .o2=0, .t=1.2};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={-1, 0,0}, .o1=0, .o2=0, .t=1.2};

    m->hop[m->n_hop++] = (rs_hopping_t){.R={ 1, 0,0}, .o1=1, .o2=1, .t=1};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={-1, 0,0}, .o1=1, .o2=1, .t=1};

    m->hop[m->n_hop++] = (rs_hopping_t){.R={ 1, 0,0}, .o1=2, .o2=2, .t=1};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={-1, 0,0}, .o1=2, .o2=2, .t=1};

    m->vert = diverge_mem_alloc_rs_vertex_t(128);
    for (index_t o1 = 0; o1 < 3; ++o1)
    for (index_t o2 = 0; o2 < 3; ++o2) {
        if(o1 == o2) {
            m->vert[m->n_vert++] = (rs_vertex_t){.chan='D', .R={0,0,0}, .o1=o1, .o2=o2, .V=U};
        }else{
            m->vert[m->n_vert++] = (rs_vertex_t){.chan='D', .R={0,0,0}, .o1=o1, .o2=o2, .V=U-2*J};
            m->vert[m->n_vert++] = (rs_vertex_t){.chan='C', .R={0,0,0}, .o1=o1, .o2=o2, .V=J};
            m->vert[m->n_vert++] = (rs_vertex_t){.chan='P', .R={0,0,0}, .o1=o1, .o2=o2, .V=J};
        }
    }

    // generate symmetries
    site_descr_t* sites = (site_descr_t*)calloc(3, sizeof(site_descr_t));

    sites[0].n_functions = 1;
    sites[0].amplitude[0] = 1.;
    sites[0].function[0] = orb_dxy;

    sites[1].n_functions = 1;
    sites[1].amplitude[0] = 1.;
    sites[1].function[0] = orb_dxz;

    sites[2].n_functions = 1;
    sites[2].amplitude[0] = 1.;
    sites[2].function[0] = orb_dyz;
    index_t symsize = POW2(m->n_orb*m->n_spin);
    m->orb_symmetries = (complex128_t*)calloc(20*symsize,sizeof(complex128_t));
    sym_op_t curr_symm;
    curr_symm.type = 'R'; curr_symm.normal_vector[0] = 0.;
    curr_symm.normal_vector[1] = 0.; curr_symm.normal_vector[2] = 1.;

    curr_symm.angle = 0;
    diverge_generate_symm_trafo(m->n_spin,sites, m->n_orb,
                &curr_symm, 1,&(m->rs_symmetries[0][0][0]),
                                    m->orb_symmetries);
    m->n_sym++;

    curr_symm.angle = 180;
    diverge_generate_symm_trafo(m->n_spin,sites, m->n_orb,
                &curr_symm, 1,&(m->rs_symmetries[1][0][0]),
                                    m->orb_symmetries+symsize);
    m->n_sym++;
    m->n_sym*= -1;
    diverge_model_internals_common( m );
    free(sites);
    diverge_model_internals_tu(m, 2.01);

    // output
    mpi_usr_printf( "model@%s\n", diverge_model_to_file(m, __FILE__ "_mod.dvg") );

    // flow step & integrate
    diverge_flow_step_t* f = diverge_flow_step_init(m, "tu", "PCD");
    diverge_euler_t eu = diverge_euler_defaults;
    double vmax = 0.0;
    do {
        diverge_flow_step_euler( f, eu.Lambda, eu.dLambda );
        diverge_flow_step_vertmax( f, &vmax );
        mpi_printf( "%.5e %.5e\n", eu.Lambda, vmax );
    } while (diverge_euler_next(&eu, vmax));

    // post-process
    diverge_postprocess_config_t pc = diverge_postprocess_config_default;
    diverge_postprocess_and_write( f, &pc, __FILE__ "_out.dvg" );

    // cleanup
    diverge_flow_step_free(f);
    diverge_model_free(m);
    diverge_finalize();

    return 0;
}




