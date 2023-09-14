#include <diverge.h>

static const double tp = -0.2;

// helper functions and structures
typedef struct {
    double M[3][3];
} matrix_t;
static matrix_t rotation( double theta );
static matrix_t reflection( int axis );
static matrix_t matmul( matrix_t A, matrix_t B );
static int sgn2( int x );

int main( int argc, char** argv ) {
    // init
    diverge_init( &argc, &argv );

    // command line arguments
    char mode[128] = "tu";
    if (argc > 1) {
        if (!strcmp(argv[1], "-h")) {
            mpi_eprintf("usage: %s [mode]\n", argv[0]);
            mpi_eprintf("    mode    default: tu. possible choices: tu, patch, or grid.\n");
            return 0;
        } else {
            strcpy( mode, argv[1] );
        }
    } else {
        mpi_eprintf("print help with %s -h\n", argv[0]);
    }

    // other init
    diverge_compilation_status();
    diverge_model_t* m = diverge_model_init();

    // TODO remove the hack...
    diverge_model_hack( NULL, "npatch_vertex_resym", "0" );

    // misc
    bool patch_mode = !strcmp(mode,"patch"),
         tu_mode = !strcmp(mode,"tu"),
         grid_mode = !strcmp(mode,"grid");
    strcpy(m->name, __FILE__ );
    m->nk[0] = m->nk[1] = patch_mode ? 200 : 20;
    m->nkf[0] = m->nkf[1] = patch_mode ? 1 : 15;
    m->lattice[0][0] = m->lattice[1][1] = m->lattice[2][2] = 1.0;
    m->n_orb = m->SU2 = m->n_spin = 1;
    m->ibz_path[1][0] = m->ibz_path[2][0] = m->ibz_path[2][1] = 0.5;
    m->n_ibz_path = 4;

    // hopping
    m->hop = diverge_mem_alloc_rs_hopping_t(128);
    m->hop[m->n_hop++] = (rs_hopping_t){.R={ 1, 0,0}, .t=1};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={-1, 0,0}, .t=1};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={ 0, 1,0}, .t=1};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={ 0,-1,0}, .t=1};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={ 1, 1,0}, .t=tp};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={-1, 1,0}, .t=tp};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={ 1,-1,0}, .t=tp};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={-1,-1,0}, .t=tp};

    // vertex
    m->vert = diverge_mem_alloc_rs_vertex_t(128);
    m->vert[m->n_vert++] = (rs_vertex_t){.chan='D', .R={0,0,0}, .V=3};

    // symmetries
    m->n_sym = 8;
    m->orb_symmetries = (complex128_t*)diverge_mem_alloc_complex128_t(m->n_sym);
    for (int i=0; i<m->n_sym; ++i) {
        m->orb_symmetries[i] = 1;
        matrix_t S;
        switch (i) {
            case 0: case 1: case 2: case 3: S = rotation(i*90); break;
            case 4: case 5: S = reflection(i%2); break;
            case 6: case 7: S = matmul(matmul(rotation(sgn2(i)*45),reflection(1)),
                                       rotation(-sgn2(i)*45)); break;
            default: break;
        }
        memcpy( m->rs_symmetries[i][0], S.M[0], sizeof(double)*9 );
    }

    // validate
    if (diverge_model_validate(m)) {
        mpi_usr_printf("invalid model...\n");
        return 1;
    }

    // common internals, chempot
    diverge_model_internals_common(m);
    diverge_model_set_chempot(m, NULL, -1, -4.*tp);

    // backend selection
    if (patch_mode) diverge_model_internals_patch(m, 4);
    if (tu_mode) diverge_model_internals_tu(m, 2.01);
    if (grid_mode) diverge_model_internals_grid(m);

    // output
    mpi_usr_printf( "model@%s\n", diverge_model_to_file(m, __FILE__ "_mod.dvg") );

    // flow step & integrate
    diverge_flow_step_t* f = diverge_flow_step_init(m, mode, "PCD");
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

// some helper functions because we're using C
static matrix_t rotation( double theta ) {
    double theta_rad = M_PI/180.*theta;
    matrix_t r = {0};
    r.M[2][2] = 1;
    r.M[0][0] = r.M[1][1] = cos(theta_rad);
    r.M[0][1] = sin(theta_rad);
    r.M[1][0] = -r.M[0][1];
    return r;
}
static matrix_t reflection( int axis ) {
    matrix_t r = {0};
    r.M[0][0] = r.M[1][1] = r.M[2][2] = 1.0;
    r.M[axis][axis] *= -1;
    return r;
}
static matrix_t matmul( matrix_t A, matrix_t B ) {
    matrix_t r = {0};
    for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j)
    for (int k=0; k<3; ++k)
        r.M[i][j] += A.M[i][k] * B.M[k][j];
    return r;
}
/* static void matrix_print( matrix_t M ) { */
/*     mpi_usr_printf( "[[%5.2f %5.2f %5.2f]" */
/*                     " [%5.2f %5.2f %5.2f]" */
/*                     " [%5.2f %5.2f %5.2f]]\n", */
/*             M.M[0][0], M.M[0][1], M.M[0][2], */
/*             M.M[1][0], M.M[1][1], M.M[1][2], */
/*             M.M[2][0], M.M[2][1], M.M[2][2] ); */
/* } */
static int sgn2( int x ) {
    if (x < 0) x *= -1;
    return -1 + (x % 2)*2;
}
