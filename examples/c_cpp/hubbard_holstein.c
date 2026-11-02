#include <diverge.h>
#include <diverge_phonons.h>

static const char model_file[] = "hubbard_holstein_el.dvg";
static const char post_file[] = "hubbard_holstein_post.dvg";
static const char vert_file[] = "hubbard_holstein_vert.bin";

int main( int argc, char** argv ) {
    diverge_init( &argc, &argv );
    atexit( &diverge_finalize );
    diverge_compilation_status();

    int nk = 12;
    int nkf = 1;
    int SU2 = argc > 1 ? atoi(argv[1]) : 0;
    float ffdist = 1.01;
    double omega_2 = 1.0,
           U = 1.0;
    double g = 2.0;

    diverge_model_t* m = diverge_model_init();
    m->nk[0] = m->nk[1] = nk;
    m->nkf[0] = m->nkf[1] = nkf;
    m->lattice[0][0] = m->lattice[1][1] = m->lattice[2][2] = 1.0;
    m->n_orb = 1;
    m->SU2 = SU2;
    m->n_spin = m->SU2 ? 1 : 2;

    m->hop = calloc(4, sizeof*m->hop);
    m->hop[m->n_hop++] = (rs_hopping_t){ .R={ 1,0,0}, .s1=-1, .t=1 };
    m->hop[m->n_hop++] = (rs_hopping_t){ .R={-1,0,0}, .s1=-1, .t=1 };
    m->hop[m->n_hop++] = (rs_hopping_t){ .R={0, 1,0}, .s1=-1, .t=1 };
    m->hop[m->n_hop++] = (rs_hopping_t){ .R={0,-1,0}, .s1=-1, .t=1 };
    m->vert = calloc(1, sizeof*m->vert);
    m->vert[m->n_vert++] = (rs_vertex_t){ .chan='D', .s1=-1, .V=U };

    diverge_model_internals_common(m);
    rs_hopping_t phonon_modes[] = { { .t=omega_2 }, };
    ph_vertex_t phonon_vertex[] = { { .s1=-1, .g=g }, };
    diverge_model_internals_tu_phonons(m, ffdist, phonon_modes, 1, phonon_vertex, 1);

    m->data = diverge_phonons_get_g(m, 0);
    m->nbytes_data = sizeof(complex128_t) * kdim(m->nk) * POW2(m->n_spin) * m->n_tu_ff;
    diverge_model_output_conf_t oc = diverge_model_output_conf_defaults;
    oc.E = 1;
    diverge_model_to_file_fg(m, model_file, &oc);
    m->data = NULL;
    m->nbytes_data = 0;

    diverge_flow_step_t* s = diverge_flow_step_init_any(m, "PCD");
    double chmax[3] = {0};
    // this is a terrbile way to integrate the flow equations; but a very
    // effective one to do debug things!
    double Lambda = 10.0,
           dLambda = -0.5;
    for (int i=0; i<10; ++i) {
        diverge_flow_step_euler( s, Lambda, dLambda );
        Lambda += dLambda;
        if (SU2) {
            chmax[0] = diverge_flow_step_eigchan(s,'p',0);
            chmax[1] = diverge_flow_step_eigchan(s,'m',0);
            chmax[2] = diverge_flow_step_eigchan(s,'h',0);
        } else {
            chmax[0] = 0.5 * diverge_flow_step_eigchan(s,'p',0);
            chmax[1] = diverge_flow_step_eigchan(s,'c',0);
            chmax[2] = diverge_flow_step_eigchan(s,'d',0);
        }
        mpi_printf( "%.5e %.5e %.5e %.5e\n", Lambda, chmax[0], chmax[1], chmax[2] );
    }

    diverge_flow_step_vertex_t v = diverge_flow_step_vertex(s, 'p');
    FILE* vf = fopen(vert_file, "wb");
    fwrite( v.ary, sizeof*v.ary, POW2(v.n_orbff * POW2(v.n_spin)) * (v.q_1 - v.q_0), vf );
    fclose( vf );

    diverge_postprocess_conf_t pc = diverge_postprocess_conf_defaults;
    pc.tu_channels = 1;
    pc.tu_channel_calc_project = 1;
    pc.tu_storing_relative = 1;
    pc.tu_storing_threshold = 0;
    diverge_postprocess_and_write_fg(s, post_file, &pc);

    diverge_flow_step_free(s);
    diverge_model_free(m);
}
