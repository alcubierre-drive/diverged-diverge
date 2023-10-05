#include <diverge.h>

// helper functions and structures
#define POW2(x) ((x)*(x))

typedef struct {
    double M[3][3];
} matrix_t;
static matrix_t rotation( double theta );
static matrix_t reflection( int axis );
static matrix_t matmul( matrix_t A, matrix_t B );
static int sgn2( int x );

typedef struct {
    double V[3];
} vector_t;

static vector_t vector_map( const double* data );
static vector_t vector_add( vector_t a, vector_t b );
static vector_t vector_scale( vector_t a, double b );
static double vector_norm( vector_t v );

static double vector_angle2d( vector_t a );

static const double phi = M_PI/6.0;
static const double U = 6.0;

#define ISCLOSE_ZERO 1e-3
#define ISCLOSE( a, b ) (fabs((a)-(b)) < ISCLOSE_ZERO)

int main( int argc, char** argv ) {
    diverge_init( &argc, &argv );
    diverge_compilation_status();
    diverge_model_t* m = diverge_model_init();

    strcpy(m->name, __FILE__ );
    m->nk[0] = m->nk[1] = 20;
    m->nkf[0] = m->nkf[1] = 15;
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
        vector_t bond = vector_add(vector_scale(vector_map(m->lattice[0]),Rx),
                                   vector_scale(vector_map(m->lattice[1]),Ry));
        if (ISCLOSE(vector_norm(bond),1.0)) {
            double angle_deg = 180./M_PI*vector_angle2d(bond);
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
        matrix_t S;
        switch (i) {
            case 0: case 1: case 2: S = rotation(i*120); break;
            case 3: case 4: case 5: S = matmul(rotation((i-3)*60), reflection(0)); break;
            default: break;
        }
        memcpy( m->rs_symmetries[i][0], S.M[0], sizeof(double)*9 );
    }

    // validate
    if (diverge_model_validate(m)) {
        mpi_usr_printf("invalid model...\n");
        return 1;
    }
    mpi_usr_printf( "model@%s\n", diverge_model_to_file(m, __FILE__ "_mod.dvg") );

    diverge_model_internals_common(m);
    diverge_model_internals_tu(m, 2.01);

    diverge_model_free(m);
    diverge_finalize();
    return 0;
}

static vector_t vector_map( const double* data ) {
    vector_t r;
    for (int i=0; i<3; ++i) { r.V[i] = data[i]; }
    return r;
}
static vector_t vector_add( vector_t a, vector_t b ) {
    vector_t r;
    for (int i=0; i<3; ++i) { r.V[i] = a.V[i]+b.V[i]; }
    return r;
}
static vector_t vector_scale( vector_t a, double b ) {
    vector_t r;
    for (int i=0; i<3; ++i) { r.V[i] = a.V[i]*b; }
    return r;
}
static double vector_norm( vector_t v ) {
    double d = 0.0;
    for (int i=0; i<3; ++i) { d += v.V[i]*v.V[i]; }
    return sqrt(d);
}
static double vector_angle2d( vector_t a ) {
    return atan2(a.V[1],a.V[0]);
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
static int sgn2( int x ) {
    if (x < 0) x *= -1;
    return -1 + (x % 2)*2;
}
