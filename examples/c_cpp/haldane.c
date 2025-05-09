#include <diverge.h>
#include <diverge_misc.h>

LINALG_IMPLEMENT
#define ZERO( x ) (fabs(x) < 1e-6)

static diverge_model_t* haldane_model( int nk, int nkf, double M, double U );

int main( int argc, char** argv ) {
    diverge_init( &argc, &argv );
    diverge_compilation_status();
    mpi_loglevel_set(6);
    atexit( &diverge_finalize );

    int nk = 24, nkf = 15;
    double M = 0.1, U = 3.0;
    diverge_model_t* m = haldane_model( nk, nkf, M, U );

    // we explore the topology features of divERGe here
    // of course, the model should be diagonalized first
    diverge_model_internals_common( m );

    // this k point array is used to take into account nontrivial phase factors
    // stemming from the proper gauge used in divERGe (effectively pinning all
    // sublattices to the origin)
    index_t nks[] = {-m->nk[0]*m->nkf[0], m->nk[1]*m->nkf[1], m->nk[2]*m->nkf[2]};

    // let's start with the Berry curvature and Chern number
    double* berry = diverge_fukui( m, NULL, -1, nks );
    double chern = 0.0;
    for (index_t i=0; i<kdimtot(m->nk,m->nkf); ++i) chern += berry[IDX3(i,0,2, m->n_orb,3)];
    free( berry );

    mpi_scc_printf( "found chern number ::: C=%.2f\n", chern );

    // how about calculating the full quantum geometric tensor (over the first
    // band)? its integral over the right components should also yield the chern
    // number…
    index_t bands[] = {0};
    complex128_t* qgt = diverge_qgt_tensor( m, NULL, -1, nks, bands, 1, 1 );
    chern = 0.0;
    for (index_t i=0; i<kdimtot(m->nk,m->nkf); ++i) chern += cimag(qgt[IDX3(i,0,1, 3,3)] - qgt[IDX3(i,1,0, 3,3)]);
    chern /= (double)kdimtot(m->nk,m->nkf);
    chern *= -2.*M_PI / claV3d_norm(claV3d_cross(claV3d_map(m->lattice[0]), claV3d_map(m->lattice[1])));
    free( qgt );

    mpi_scc_printf( "chern number via QGT ::: C=%.2f\n", chern );

    // let's have a look at the non-abelian versions. To do that map the model
    // to a supercell, so reset internals and let's go. We need to use the right
    // convention here...
    diverge_model_internals_reset( m );
    int nr[] = {2,2,1};
    diverge_supercell_set_sign(-1);
    diverge_model_to_supercell( m, nr );
    diverge_model_internals_common( m );

    index_t bands_s[] = {0,1,2,3};
    berry = diverge_fukui_non_abelian( m, NULL, -1, nks, bands_s, 4 );
    chern = 0.0;
    for (index_t i=0; i<kdimtot(m->nk,m->nkf); ++i) chern += berry[IDX2(i,2, 3)];
    free( berry );

    mpi_scc_printf( "non-abelian fukui method over lower band(s) ::: C=%.2f\n", chern );

    // and now export the model to a file, appending the full quantum geometric
    // tensor (in its supercell basis) to the data field. That way you can e.g.
    // check for how its symmetries look like in momentum space.
    m->data = diverge_qgt_tensor( m, NULL, -1, nks, bands_s, 4, 1 );
    m->nbytes_data = sizeof(complex128_t) * kdimtot(m->nk,m->nkf) * 9;
    diverge_model_output_conf_t oc = diverge_model_output_conf_defaults;
    oc.npath = -1;
    oc.kf_ibz_path = 1;
    oc.kf = 1;
    oc.E = 1;
    diverge_model_to_file_fg( m, __FILE__ "_mod.dvg", &oc );

    diverge_model_free( m );
}

static diverge_model_t* haldane_model( int nk, int nkf, double M, double U ) {
    diverge_model_t* m = diverge_model_init();
    m->nk[0] = m->nk[1] = nk;
    m->nkf[0] = m->nkf[1] = nkf;
    m->SU2 = m->n_spin = 1;
    m->n_orb = 2;

    m->lattice[0][0] = m->lattice[2][2] = 1.0;
    m->lattice[1][0] = cos(2.*M_PI/3.);
    m->lattice[1][1] = sin(2.*M_PI/3.);

    claV3d_t a1 = claV3d_map( m->lattice[0] ),
             a2 = claV3d_map( m->lattice[1] );

    claV3d_t o1 = claV3d_add(claV3d_scale(a1,1./3.), claV3d_scale(a2,2./3.)),
             o2 = claV3d_add(claV3d_scale(a1,2./3.), claV3d_scale(a2,1./3.));

    claV3d_rmap( o1, m->positions[0] );
    claV3d_rmap( o2, m->positions[1] );

    double* dists = diverge_model_unique_distances( m, 4, 2 );
    VECTOR_INIT( rs_hopping_t, hop );

    for (int i=-2; i<=2; ++i)
    for (int j=-2; j<=2; ++j)
    for (int o=0; o<2; ++o)
    for (int p=0; p<2; ++p) {
        claV3d_t O = claV3d_sub( claV3d_map(m->positions[p]), claV3d_map(m->positions[o]) );
        claV3d_t D = claV3d_add( claV3d_add( claV3d_scale(a1,i), claV3d_scale(a2,j) ), O );

        double dist = claV3d_norm( D );
        if ZERO(dist - dists[1]) {
            VECTOR_PUSH_BACK( hop, (rs_hopping_t){ .R={i,j}, .o1=o, .o2=p, .t=1 } );
        } else if ZERO(dist - dists[2]) {
            index_t phi = lroundl( fmod( carg( D.v[0] + I * D.v[1] ) / (2.*M_PI) + 1., 1. ) * 6. );
            VECTOR_PUSH_BACK( hop, (rs_hopping_t){ .R={i,j}, .o1=o, .o2=p,
                    .t=(o?1.:-1.) * (phi%2?1:-1.) * I * M } );
        }
    }
    free( dists );

    m->hop = hop;
    m->n_hop = hop_sz;

    m->vert = calloc(2, sizeof*m->vert);
    for (int o=0; o<2; ++o)
        m->vert[o] = (rs_vertex_t){ .chan='D', .o1=o, .o2=o, .V=U };
    m->n_vert = 2;

    m->ibz_path[1][0] = 0.5;
    m->ibz_path[2][0] = m->ibz_path[2][1] = 1./3.;
    m->n_ibz_path = 4;

    return m;
}

