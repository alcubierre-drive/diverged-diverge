#include <diverge.h>
#include <diverge_misc.h>
#include <diverge_phonons.h>
LINALG_IMPLEMENT

#include <dirent.h>
#include <errno.h>

static void add_symmetries( diverge_model_t* m ) {
    sym_op_t sym = { .type='R', .normal_vector={0,0,1} };
    site_descr_t p_orbs[6] = {{.function={orb_px,}},
                              {.function={orb_py,}},
                              {.function={orb_pz,}},
                              {.function={orb_px,}},
                              {.function={orb_py,}},
                              {.function={orb_pz,}}};
    site_descr_t s_orbs[2] = {{.function={orb_s,}},
                              {.function={orb_s,}}};
    site_descr_t* orbs = m->n_orb == 2 ? s_orbs : p_orbs;
    for (int i=0; i<m->n_orb; ++i) {
        orbs[i].amplitude[0] = 1.0;
        orbs[i].n_functions = 1;
    }
    m->orb_symmetries = calloc( 6*POW2(m->n_orb), sizeof*m->orb_symmetries );
    for (int a=0; a<6; ++a) {
        sym.angle = a*60;
        diverge_generate_symm_trafo(1, orbs, m->n_orb, &sym, 1,
                m->rs_symmetries[a][0], &m->orb_symmetries[a*POW2(m->n_orb)]);
        m->n_sym++;
    }
    m->n_sym *= -1;
}

// TODO setenv is only necessary because conventions are a mess (and not sorted
// out properly yet)
int setenv(const char *name, const char *value, int overwrite);

int main(int argc, char** argv) {

    // TODO setenv is only necessary because conventions are a mess (and not
    // sorted out properly yet)
    // setenv( "PHONON_CFG", "4", 1 );

    diverge_init(&argc, &argv);

    double U = 0.0;
    index_t nk = 24,
            nkf = 1;
    index_t maxiter = 10;

    DIR* dir = opendir("phrenorm_graphene");
    if (dir) closedir(dir);
    else if (ENOENT == errno) {
        mpi_err_printf( "directory 'phrenorm_graphene/' does not exist.\n" );
        mpi_err_printf( "please create it and run the corresponding elphmod example in there.\n" );
        diverge_mpi_exit( EXIT_FAILURE );
    }

    // read the electron model
    diverge_model_t* e = diverge_read_W90_model( "phrenorm_graphene/graphene_hr.dat",
            "phrenorm_graphene/graphene.wout", "phrenorm_graphene/graphene_wsvec.dat", 0 );

    diverge_wigner_config_t wc = diverge_wigner_config_defaults;
    wc.apply_asr = DIVERGE_WIGNER_CONFIG_APPLY_ASR_HOPPINGS | DIVERGE_WIGNER_CONFIG_APPLY_ASR_VERTICES;
    // read the phonon model (+elph)
    diverge_model_t* p = diverge_phonons_read_ifc_and_elph( "phrenorm_graphene/dfpt.ifc",
            "phrenorm_graphene/wigner.fmt", "phrenorm_graphene/dfpt.epmatwp", 5.e-2, &wc );

    // scale the interaction
    // ph_vertex_t* pv = (ph_vertex_t*)p->vert; for (index_t i=0; i<p->n_vert; ++i) { pv[i].g  *= 3.0; }

    // add some symmetries
    diverge_generate_symm_maps_set_precision( 1.e-2, 1.e-2, 1.e-2, 0.99, 1.e-2, 1.e-2 );
    add_symmetries( e );
    add_symmetries( p );

    // kpt setup
    e->nk[0] = e->nk[1] = nk;
    e->nkf[0] = e->nkf[1] = nkf;
    e->ibz_path[1][0] = 0.5;
    e->ibz_path[2][0] = 1./3.; e->ibz_path[2][1] = 1./3.;
    e->n_ibz_path = 4;
    memcpy( p->nk, e->nk, sizeof(e->nk) );
    memcpy( p->nkf, e->nkf, sizeof(e->nkf) );
    memcpy( p->ibz_path, e->ibz_path, sizeof(e->ibz_path) );
    p->n_ibz_path = e->n_ibz_path;

    // validation (does not do anything :)
    if (diverge_model_validate( e ) || diverge_model_validate( p ))
        diverge_mpi_exit(EXIT_FAILURE);

    // common internals for electronic model, including...
    diverge_model_internals_common( e );
    diverge_model_set_filling( e, NULL, -1, 0.564231 );
    // ... a DOS!
    index_t n_w = 1024;
    double* nu = diverge_linspace( 0, 1, n_w );
    double* w = diverge_filling_to_energy( e, NULL, -1, nu, n_w );
    e->data = diverge_model_dos( e, NULL, -1, w, n_w, -1.e-2 );
    e->nbytes_data = sizeof(double) * n_w;
    free( nu );
    free( w );

    // we reset the electronic symmetries; that way, we have symmetrization but
    // do *not* symmetry-enforce vertices
    e->n_sym = 0;
    // diverge_model_internals_common( p );
    // diverge_symmetrize_hoppings( p, NULL, NULL, 1.e-4 );
    // diverge_phonons_apply_asr( p );
    // diverge_model_internals_reset( p );

    // common internals for phononic model, and the sqrt() for the dispersion
    diverge_model_internals_common( p );
    diverge_phonons_apply_sgnsqrt( p );

    // write phonons and electrons to file
    diverge_model_output_conf_t oc = diverge_model_output_conf_defaults;
    oc.kf_ibz_path = 1;
    oc.kc_ibz_path = 1;
    oc.npath = -1;
    oc.kf = 1;
    oc.kc = 1;
    oc.U = oc.E = 1;

    // add some electronic vertices
    e->vert = calloc(1024, sizeof*e->vert);
    e->vert[e->n_vert++] = (rs_vertex_t){ .chan='C', .o1=0, .o2=0, .V=U };
    e->vert[e->n_vert++] = (rs_vertex_t){ .chan='C', .o1=1, .o2=1, .V=U };

    // set up the phonon TUFRG with 1.51 lattice vectors FF distance
    diverge_model_t* pp = diverge_model_internals_tu_phonons( e,
            1.51*claV3d_norm(claV3d_map(e->lattice[0])),
            p->hop, p->n_hop, (ph_vertex_t*)p->vert, p->n_vert );

    // save models only after tu internals are set to get the formfactors
    diverge_model_to_file_fg( e, __FILE__ "_e.dvg", &oc );
    diverge_model_to_file_fg( p, __FILE__ "_p.dvg", &oc );

    // attach elph to the model and save it
    pp->data = diverge_phonons_get_g( e );
    pp->nbytes_data = sizeof(complex128_t) * kdim(e->nk) * pp->n_orb * e->n_tu_ff;
    diverge_model_to_file_fg( pp, __FILE__ "_pp.dvg", &oc );
    pp->nbytes_data = 0;
    pp->data = NULL;

    // and do the flow+postproc
    diverge_flow_step_t* st = diverge_flow_step_init_any( e, "PCD" );

    diverge_euler_t eu = diverge_euler_defaults;
    eu.maxiter = maxiter;
    double maxvert = 0;
    do {
        diverge_flow_step_euler( st, eu.Lambda, eu.dLambda );
        double Pmax = diverge_flow_step_eigchan( st, 'p', 0 ),
               Cmax = diverge_flow_step_eigchan( st, 'm', 0 ),
               Dmax = diverge_flow_step_eigchan( st, 'h', 0 );
        maxvert = MAX(MAX(Pmax,Cmax),Dmax);
        printf( "%.5e %.5e %.5e %.5e\n", eu.Lambda, Pmax, Cmax, Dmax );
    } while (diverge_euler_next( &eu, maxvert ));
    diverge_postprocess_conf_t pc = diverge_postprocess_conf_defaults;
    pc.tu_storing_relative = 1;
    pc.tu_storing_threshold = 0.9;
    pc.tu_symmetry_maps = 1;
    pc.tu_which_solver_mode = 'B';
    pc.tu_lingap = 0;
    pc.tu_channel_calc_project = -1;
    diverge_postprocess_and_write_fg( st, __FILE__ "_post.dvg", &pc );
    diverge_flow_step_free( st );

    // cleanup
    diverge_model_free( e );
    diverge_model_free( p );
    diverge_finalize();
}
