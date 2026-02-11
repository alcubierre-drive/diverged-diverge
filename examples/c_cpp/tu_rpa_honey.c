#include <diverge.h>
#include <diverge_misc.h>
LINALG_IMPLEMENT

int main(int argc, char** argv) {
    diverge_init( &argc, &argv );
    mpi_loglevel_set( 5 );
    diverge_compilation_status();

    const index_t nk = 48;
    const double t = 1.0, tp = 0.1, U = 3.6;

    diverge_model_t* m = diverge_model_init();
    m->lattice[0][0] = m->lattice[1][0] = sin(M_PI/3.);
    m->lattice[0][1] = m->lattice[1][1] = cos(M_PI/3.);
    m->lattice[0][1] *= -1.0;
    m->lattice[2][2] = 1.0;

    claV3d_t a1 = claV3d_map(m->lattice[0]), a2 = claV3d_map(m->lattice[1]);
    claV3d_rmap(claV3d_scale(claV3d_add(a1,a2),-1./3.), m->positions[0]);
    claV3d_rmap(claV3d_scale(claV3d_add(a1,a2), 1./3.), m->positions[1]);
    m->n_orb = 2;
    m->n_spin = m->SU2 = 1;
    m->nk[0] = m->nk[1] = nk;
    m->nkf[0] = m->nkf[1] = 1;
    m->n_ibz_path = 4;
    m->ibz_path[1][0] = 0.5;
    m->ibz_path[2][0] = 2./3.;
    m->ibz_path[2][1] = 1./3.;
    strcpy( m->name, __FILE__ );

    double t_norm = claV3d_norm(claV3d_map(m->positions[0])), tp_norm = 1.0;
    m->hop = calloc(32, sizeof(rs_hopping_t));
    m->vert = calloc(16, sizeof(rs_vertex_t));
    m->vert[m->n_vert++] = (rs_vertex_t){ .chan='D', .o1=0,.o2=0, .s1=-1, .V=U };
    m->vert[m->n_vert++] = (rs_vertex_t){ .chan='D', .o1=1,.o2=1, .s1=-1, .V=U };
    for (int Rx=-5; Rx<=5; ++Rx)
    for (int Ry=-5; Ry<=5; ++Ry)
    for (int o=0; o<2; ++o)
    for (int p=0; p<2; ++p) {
        claV3d_t R = claV3d_add(claV3d_scale(a1,Rx),claV3d_scale(a2,Ry));
        claV3d_t P = claV3d_sub(claV3d_map(m->positions[p]),claV3d_map(m->positions[o]));
        claV3d_t D = claV3d_add(R,P);
        double dist = claV3d_norm(D);
        if (fabs(dist- t_norm) < 1e-5) m->hop[m->n_hop++] = (rs_hopping_t){ .R={Rx,Ry}, .o1=o,.o2=p, .t=t };
        if (fabs(dist-tp_norm) < 1e-5) m->hop[m->n_hop++] = (rs_hopping_t){ .R={Rx,Ry}, .o1=o,.o2=p, .t=tp };
    }

    if (diverge_model_validate(m)) diverge_mpi_exit(EXIT_FAILURE);

    diverge_model_internals_common(m);
    diverge_model_set_chempot( m, NULL, -1, 0.8 );
    diverge_model_internals_tu( m, 0.01 );
    diverge_model_output_conf_t cfg = diverge_model_output_conf_defaults;
    cfg.npath = -1;
    cfg.kf = cfg.kf_ibz_path = 1;
    diverge_model_to_file_finegrained(m, __FILE__ ".mod.dvg", &cfg);

    VECTOR_INIT( index_t, kidx );
    const double* E = diverge_model_internals_get_E(m);
    for (index_t k=0; k<kdim(m->nk); ++k) {
        double Eabs_min = MIN( fabs(E[IDX2(k,0,2)]), fabs(E[IDX2(k,1,2)]) );
        if (Eabs_min < 1.0) VECTOR_PUSH_BACK( kidx, k );
    }
    mpi_usr_printf( "found %li kidx\n", kidx_sz );

    diverge_flow_step_t* s = diverge_flow_step_init_any( m, "C" );
    diverge_flow_step_attach_rpa( s, kidx, kidx_sz );
    free( kidx );
    diverge_euler_t eu = diverge_euler_defaults;
    double maxvert = 0;
    do {
        diverge_flow_step_euler( s, eu.Lambda, eu.dLambda );
        diverge_flow_step_vertmax( s, &maxvert );
        printf( "%.5e %.5e\n", eu.Lambda, maxvert );
    } while (diverge_euler_next( &eu, maxvert ));

    diverge_postprocess_conf_t pc = diverge_postprocess_conf_defaults;
    diverge_postprocess_and_write_fg( s, __FILE__ ".out.dvg", &pc );

    diverge_flow_step_free( s );
    diverge_model_free( m );
    diverge_finalize();
}
