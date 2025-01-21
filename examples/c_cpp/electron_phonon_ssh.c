#include <diverge.h>
#include <tu/phonons.h>
#include <diverge_misc.h>

FILE* vertex_file = NULL;
const char vertex_file_name[] = __FILE__ "_eig.bin";

static inline void append_vertex_eigs_to_file( diverge_flow_step_vertex_t v ) {
    if (!vertex_file) {
        vertex_file = fopen(vertex_file_name, "wb");
        mpi_usr_printf("opening %s for vertices\n", vertex_file_name);
    }
    index_t sz = v.n_orbff, num = v.q_1 - v.q_0;
    double* vals = calloc(num*sz, sizeof*vals);
    complex128_t* vecs = calloc(num*POW2(sz), sizeof*vecs);
    complex128_t* tmp = calloc(num*POW2(sz), sizeof*tmp);
    for (index_t q=0; q<num; ++q) {
        single_svd( v.ary + POW2(sz)*q, tmp + POW2(sz)*q, vecs + POW2(sz)*q, vals + sz*q, sz );
        single_eigen_sort( tmp, vals, sz, 'M' );
    }
    free( vecs );
    free( tmp );
    fwrite( vals, sizeof*vals, sz*num, vertex_file );
    free( vals );
}

int main( int argc, char** argv ) {
    diverge_init( &argc, &argv );

    double t = 1.0, U = 0.0, V = 0.0, max_dist = 1.01;

    int nk = 24, nkf = 15;
    double omega = 10.0, lambda = 0.1;

    if (argc > 1) lambda = atof(argv[1]);
    if (argc > 2) omega = atof(argv[2]);

    double g = sqrt(lambda * (8.*t) * omega);

    diverge_model_t* m = diverge_model_init();
    m->nk[0] = m->nk[1] = nk;
    m->nkf[0] = m->nkf[1] = nkf;
    m->lattice[0][0] = m->lattice[1][1] = m->lattice[2][2] = 1.0;
    m->hop = calloc(1024, sizeof*m->hop);
    m->hop[m->n_hop++] = (rs_hopping_t){ .R={ 1, 0,0}, .t=t };
    m->hop[m->n_hop++] = (rs_hopping_t){ .R={-1, 0,0}, .t=t };
    m->hop[m->n_hop++] = (rs_hopping_t){ .R={ 0, 1,0}, .t=t };
    m->hop[m->n_hop++] = (rs_hopping_t){ .R={ 0,-1,0}, .t=t };
    m->vert = calloc(1024, sizeof*m->vert);
    m->vert[m->n_vert++] = (rs_vertex_t){ .chan='C', .s1=-1, .V=U };
    // we add some P and D channel interactions, just to simulate how the elph
    // interaction would have to look
    m->vert[m->n_vert++] = (rs_vertex_t){ .chan='P', .R={ 1, 0,0}, .s1=-1, .V=V };
    m->vert[m->n_vert++] = (rs_vertex_t){ .chan='P', .R={-1, 0,0}, .s1=-1, .V=V };
    m->vert[m->n_vert++] = (rs_vertex_t){ .chan='P', .R={ 0, 1,0}, .s1=-1, .V=V };
    m->vert[m->n_vert++] = (rs_vertex_t){ .chan='P', .R={ 0,-1,0}, .s1=-1, .V=V };
    m->vert[m->n_vert++] = (rs_vertex_t){ .chan='D', .R={ 1, 0,0}, .s1=-1, .V=V };
    m->vert[m->n_vert++] = (rs_vertex_t){ .chan='D', .R={-1, 0,0}, .s1=-1, .V=V };
    m->vert[m->n_vert++] = (rs_vertex_t){ .chan='D', .R={ 0, 1,0}, .s1=-1, .V=V };
    m->vert[m->n_vert++] = (rs_vertex_t){ .chan='D', .R={ 0,-1,0}, .s1=-1, .V=V };
    m->n_spin = m->n_orb = m->SU2 = 1;

    diverge_model_internals_common(m);
    diverge_model_to_file(m, __FILE__ "_mod.dvg" );
    diverge_model_set_filling(m, NULL, -1, 0.5);

    rs_hopping_t phop[] = {
        { .o1=0, .o2=0, .t=omega },
        { .o1=1, .o2=1, .t=omega },
    };
    ph_vertex_t pvert[] = {
        { .R12={ 1, 0,0}, .O1=0, .g=g },
        { .R12={-1, 0,0}, .O1=0, .g=g },
        { .R12={ 0, 1,0}, .O1=1, .g=g },
        { .R12={ 0,-1,0}, .O1=1, .g=g },
    };
    diverge_model_internals_tu_phonons(m, max_dist, phop, sizeof(phop)/sizeof(phop[0]), (rs_vertex_t*)pvert, 4);

    diverge_flow_step_t* st = diverge_flow_step_init_any(m, "PCD");
    diverge_euler_t eu = diverge_euler_defaults;
    double cmax[3] = {0};
    double vmax;
    do {
        diverge_flow_step_euler( st, eu.Lambda, eu.dLambda );
        diverge_flow_step_chanmax( st, cmax );
        diverge_flow_step_vertmax( st, &vmax );
        append_vertex_eigs_to_file( diverge_flow_step_vertex( st, 'p' ) );
        append_vertex_eigs_to_file( diverge_flow_step_vertex( st, 'm' ) );
        append_vertex_eigs_to_file( diverge_flow_step_vertex( st, 'h' ) );
        mpi_printf( "%.5e %.5e %.5e %.5e\n", eu.Lambda, cmax[0], cmax[1], cmax[2] );
    } while (diverge_euler_next(&eu, vmax));
    if (vertex_file) fclose( vertex_file );

    diverge_postprocess_conf_t pc = diverge_postprocess_conf_defaults;
    pc.tu_storing_relative = 1;
    pc.tu_storing_threshold = 0.8;
    pc.tu_susceptibilities_ff = true;
    diverge_postprocess_and_write_finegrained( st, __FILE__ "_out.dvg", &pc );
    diverge_flow_step_free( st );
    diverge_model_free( m );
    diverge_finalize();
}
