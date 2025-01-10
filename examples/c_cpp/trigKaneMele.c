#include <diverge.h>

#define LINALG_NO_LAPACKE
#define LINALG_EXPORT static inline
#include <misc/linalg.h>
LINALG_IMPLEMENT

// helper functions and structures
#define POW2(x) ((x)*(x))

static claM3d_t rotation( double theta );
static claM3d_t reflection( int axis );
static int sgn2( int x );
static double claV3d_angle2d( claV3d_t a );

static const double phi = M_PI/6.0;
static const double U = 6.0;

#define ISCLOSE_ZERO 1e-3
#define ISCLOSE( a, b ) (fabs((a)-(b)) < ISCLOSE_ZERO)

int main( int argc, char** argv ) {
    diverge_init( &argc, &argv );
    diverge_compilation_status();
    mpi_loglevel_set( 5 );
    diverge_model_t* m = diverge_model_init();

    strcpy(m->name, __FILE__ );
    m->nk[0] = m->nk[1] = argc > 1 ? atoi(argv[1]) : 20;
    m->nkf[0] = m->nkf[1] = argc > 2 ? atoi(argv[2]) : 15;
    m->lattice[0][0] = m->lattice[2][2] = 1.0;
    m->lattice[1][0] = cos(M_PI/3.0);
    m->lattice[1][1] = sin(M_PI/3.0);
    m->n_orb = 1;
    m->SU2 = 0;
    m->n_spin = 2;

    m->ibz_path[1][0] = 0.5;
    m->ibz_path[2][0] = 2./3.;
    m->ibz_path[2][1] = 1./3.;
    m->n_ibz_path = 4;

    m->hop = diverge_mem_alloc_rs_hopping_t(128);
    for (int Rx=-4; Rx<=4; ++Rx)
    for (int Ry=-4; Ry<=4; ++Ry) {
        claV3d_t bond = claV3d_add(claV3d_scale(claV3d_map(m->lattice[0]),Rx),
                                   claV3d_scale(claV3d_map(m->lattice[1]),Ry));
        if (ISCLOSE(claV3d_norm(bond),1.0)) {
            double angle_deg = 180./M_PI*claV3d_angle2d(bond);
            if (ISCLOSE(angle_deg, 0) || ISCLOSE(angle_deg, 120) || ISCLOSE(angle_deg, -120)) {
                for (int s=0; s<2; ++s) {
                    m->hop[m->n_hop++] = (rs_hopping_t){.R={ Rx, Ry}, .t=cexp( I*phi*sgn2(s)), .s1=s,.s2=s};
                    m->hop[m->n_hop++] = (rs_hopping_t){.R={-Rx,-Ry}, .t=cexp(-I*phi*sgn2(s)), .s1=s,.s2=s};
                }
            }
        }
    }

    m->vert = diverge_mem_alloc_rs_vertex_t(128);
    m->vert[m->n_vert++] = (rs_vertex_t){.chan='D', .R={0,0,0}, .V=U, .s1=-1};

    // symmetries (only use C3v here)
    m->n_sym = 6;
    m->orb_symmetries = (complex128_t*)diverge_mem_alloc_complex128_t(m->n_sym*4);
    for (int i=0; i<m->n_sym; ++i) {
        m->orb_symmetries[i*4+0] = m->orb_symmetries[i*4+3] = 1;
        m->orb_symmetries[i*4+1] = m->orb_symmetries[i*4+2] = 0;
        claM3d_t S;
        switch (i) {
            case 0: case 1: case 2: S = rotation(i*120); break;
            case 3: case 4: case 5: S = claM3d_matmul(rotation((i-3)*60), reflection(0)); break;
            default: break;
        }
        memcpy( m->rs_symmetries[i][0], S.m[0], sizeof(double)*9 );
    }

    // validate
    if (diverge_model_validate(m)) {
        mpi_usr_printf("invalid model...\n");
        return 1;
    }
    mpi_usr_printf( "model@%s\n", diverge_model_to_file(m, __FILE__ "_mod.dvg") );

    diverge_model_internals_common( m );
    diverge_model_internals_any( m, "tu", 2.01 );
    // some algorithm testing
    diverge_model_hack( m, "tu_extra_propagator_timings", "1" );

    diverge_flow_step_t* st = diverge_flow_step_init_any(m, "PCD");
    diverge_flow_step_euler( st, 10, -1 );
    for (int t=0; t<diverge_flow_step_ntimings(st); ++t)
        mpi_tim_printf( "used %.2fs for '%s'\n", diverge_flow_step_timing(st, t),
                diverge_flow_step_timing_descr(st, t) );
    diverge_postprocess_and_write( st, __FILE__ "_out.dvg" );

    diverge_flow_step_free( st );
    diverge_model_free(m);
    diverge_finalize();
    return 0;
}

static double claV3d_angle2d( claV3d_t a ) {
    return atan2(a.v[1],a.v[0]);
}

// some helper functions because we're using C
static claM3d_t rotation( double theta ) {
    double theta_rad = M_PI/180.*theta;
    claM3d_t r = {0};
    r.m[2][2] = 1;
    r.m[0][0] = r.m[1][1] = cos(theta_rad);
    r.m[0][1] = sin(theta_rad);
    r.m[1][0] = -r.m[0][1];
    return r;
}
static claM3d_t reflection( int axis ) {
    claM3d_t r = {0};
    r.m[0][0] = r.m[1][1] = r.m[2][2] = 1.0;
    r.m[axis][axis] *= -1;
    return r;
}
static int sgn2( int x ) {
    if (x < 0) x *= -1;
    return -1 + (x % 2)*2;
}
