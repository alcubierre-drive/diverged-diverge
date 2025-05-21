#include <diverge.h>
#include <diverge_misc.h>
LINALG_IMPLEMENT

const index_t nk = 30;
const double t = 1.0;
const double tp = 0.1;
const double U = 3.6;
const double V = 2.0;
const double V2 = 0.5;
const double nu = 0.6;

#define ZERO(x) (fabs(x) < 1e-5)

int main(int argc, char** argv) {
    // init
    diverge_init( &argc, &argv );
    mpi_loglevel_set( 5 );
    diverge_compilation_status();
    diverge_model_t* m = diverge_model_init();

    // set up honeycomb lattice
    m->lattice[2][2] = m->lattice[0][0] = 1.0;
    m->lattice[1][0] = cos(2.*M_PI/3.);
    m->lattice[1][1] = sin(2.*M_PI/3.);
    claV3d_t a1 = claV3d_map(m->lattice[0]),
             a2 = claV3d_map(m->lattice[1]);
    claV3d_rmap(claV3d_add(claV3d_scale(a1,2./3.),claV3d_scale(a2,1./3.)), m->positions[0]);
    claV3d_rmap(claV3d_add(claV3d_scale(a1,1./3.),claV3d_scale(a2,2./3.)), m->positions[1]);

    m->n_orb = 2;
    m->n_spin = 1;
    m->SU2 = 1;
    m->nk[0] = m->nk[1] = nk;
    m->nkf[0] = m->nkf[1] = 1;
    m->n_ibz_path = 4;
    m->ibz_path[1][0] = 0.5;
    m->ibz_path[2][0] = 1./3.;
    m->ibz_path[2][1] = 1./3.;
    strcpy( m->name, __FILE__ );

    // find hopping parameters
    double* udist = diverge_model_unique_distances(m, 5, 2);
    m->hop = calloc(1024, sizeof(rs_hopping_t));
    m->vert = calloc(1024, sizeof(rs_vertex_t));
    m->vert[m->n_vert++] = (rs_vertex_t){ 'D', {0,0,0}, 0,0, -1,0,0,0, U };
    m->vert[m->n_vert++] = (rs_vertex_t){ 'D', {0,0,0}, 1,1, -1,0,0,0, U };
    for (int Rx=-5; Rx<=5; ++Rx)
    for (int Ry=-5; Ry<=5; ++Ry)
    for (int o=0; o<2; ++o)
    for (int p=0; p<2; ++p) {
        claV3d_t D = claV3d_add(claV3d_scale(a1,Rx),claV3d_scale(a2,Ry));
        D = claV3d_add(D,claV3d_sub(claV3d_map(m->positions[p]),claV3d_map(m->positions[o])));
        double dist = claV3d_norm(D);
        if (ZERO(dist-udist[1])) {
            m->hop[m->n_hop++] = (rs_hopping_t){ {Rx,Ry,0}, o,p, 0,0, t };
            m->vert[m->n_vert++] = (rs_vertex_t){ 'D', {Rx,Ry,0}, o,p, -1,0,0,0, V };
        }
        if (ZERO(dist-udist[2])) {
            m->hop[m->n_hop++] = (rs_hopping_t){ {Rx,Ry,0}, o,p, 0,0, tp };
            m->vert[m->n_vert++] = (rs_vertex_t){ 'D', {Rx,Ry,0}, o,p, -1,0,0,0, V2 };
        }
    }
    free( udist );

    // set up symmetries
    m->orb_symmetries = calloc(12*4, sizeof(*m->orb_symmetries));
    site_descr_t sites[2];
    for (int s=0; s<2; ++s) {
        sites[s].amplitude[0] = 1.0;
        sites[s].function[0] = orb_s;
        sites[s].n_functions = 1;
    }
    sym_op_t sym = {.type='R', .normal_vector={0,0,1}};
    for (int i=0; i<6; ++i) {
        sym.angle = 60*i;
        diverge_generate_symm_trafo( 1, sites, 2, &sym, 1, m->rs_symmetries[m->n_sym][0],
                m->orb_symmetries+m->n_sym*4 );
        m->n_sym++;
    }
    sym.type = 'M';
    for (int i=0; i<6; ++i) {
        claM3d_t R = {{{ cos(M_PI/6.*i), sin(M_PI/6.*i), 0},
                       {-sin(M_PI/6.*i), cos(M_PI/6.*i), 0},
                       {              0,              0, 1}}};
        claV3d_rmap(claM3d_matvec(R, (claV3d_t){{1,0,0}}), sym.normal_vector);
        diverge_generate_symm_trafo( 1, sites, 2, &sym, 1, m->rs_symmetries[m->n_sym][0],
                m->orb_symmetries+m->n_sym*4 );
        m->n_sym++;
    }
    /* m->n_sym = 0; // TODO */

    if (diverge_model_validate(m)) diverge_mpi_exit(EXIT_FAILURE);

    diverge_model_internals_common(m);
    diverge_model_set_filling(m, NULL, -1, nu);
    diverge_model_internals_tu(m, diverge_max_dist_iobi());
    diverge_model_hack(m, "tu_katanin", "0");

    /*
    diverge_model_output_conf_t cfg = diverge_model_output_conf_defaults_CPP();
    cfg.npath = -1;
    cfg.kf_ibz_path = 1;
    diverge_model_to_file_finegrained(m, __FILE__ ".mod.dvg", &cfg);
    */

    // flow
    diverge_flow_step_t* s = diverge_flow_step_init_any( m, "PCDS" );
    diverge_euler_t eu = diverge_euler_defaults_CPP();
    complex128_t* sigma_step = diverge_flow_step_vertex( s, 'S' ).ary;

    double maxvert = 0;
    int iter = 0;
    do {
        diverge_flow_step_euler( s, eu.Lambda, eu.dLambda );
        diverge_flow_step_vertmax( s, &maxvert );
        if (++iter == 100) break;
    } while (diverge_euler_next( &eu, maxvert ));

    FILE* f = diverge_mpi_comm_rank() == 0 ? fopen("sigma.bin", "wb") : NULL;
    if (f) {
        fwrite( sigma_step, kdimtot(m->nk,m->nkf)*POW2(m->n_orb*m->n_spin), sizeof*sigma_step, f );
        fclose( f );
    }

    // post processing
    diverge_postprocess_and_write( s, __FILE__ ".out.dvg" );

    // cleanup
    diverge_flow_step_free( s );
    diverge_model_free( m );
    diverge_finalize();
}
