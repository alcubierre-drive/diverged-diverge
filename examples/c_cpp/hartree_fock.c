#include <diverge.h>
#include <diverge_misc.h>
LINALG_IMPLEMENT

diverge_model_t* s2xs2_hubbard( int nk, int nkf, double ffcut, double U );

int main( int argc, char** argv ) {

    int nk = 24, nkf = 1;
    double ffcut = 0.1;

    double nu = 0.5;
    double U = 3.0;
    double T = 1.e-2;

    double amp = 1e-5;
    index_t maxiter = 200;
    double mindiff = 1e-10;
    double alpha = 0.7;

    diverge_init( &argc, &argv );

    // initialize the model
    diverge_model_t* m = s2xs2_hubbard( nk, nkf, ffcut, U );
    index_t nk_nb2 = kdimtot(m->nk,m->nkf)*POW2(m->n_orb*m->n_spin);

    // hartree fock backend
    diverge_hartree_fock_t* h = diverge_hartree_fock_init( m, nu, "PCD" );

    // self-energy arrays
    complex128_t* sigma = diverge_hartree_fock_get_sigma( h );
    complex128_t* sigma_prev = diverge_hartree_fock_get_sigma_prev( h );

    // initial seed (to get the AFM)
    sigma[IDX4(0,1,0,1,2,2,2)] = +amp;
    sigma[IDX4(1,1,1,1,2,2,2)] = -amp;
    sigma[IDX4(0,0,0,0,2,2,2)] = -amp;
    sigma[IDX4(1,0,1,0,2,2,2)] = +amp;

    mpi_printf( "# iter absdiff maxval\n" );
    // self-consistency loop
    for (int it=0; it<maxiter; ++it) {
        diverge_hartree_fock_step( h, T );
        diverge_hartree_fock_normalize_sigma( h );

        // linear mixing
        for (index_t i=0; i<nk_nb2; ++i)
            sigma[i] = alpha * sigma[i] + (1.-alpha) * sigma_prev[i];

        // print out the difference and the maximum value of sigma
        double err = 0.0;
        double smax = 0.0;
        for (index_t i=0; i<nk_nb2; ++i) {
            err += cabs(sigma_prev[i]-sigma[i]);
            smax = MAX(smax, cabs(sigma[i]));
        }
        err /= (double)nk_nb2;

        // output
        mpi_printf( "%i %.5e %.5e\n", it, err, smax );

        if (err < mindiff && it > 10) break;
    }

    // attach the self-energy to the model
    m->data = calloc( nk_nb2, sizeof*sigma );
    m->nbytes_data = nk_nb2 * sizeof*sigma;
    memcpy( m->data, sigma, m->nbytes_data );

    // and output the model (which includes the self-energy)
    diverge_model_output_conf_t oc = diverge_model_output_conf_defaults;
    oc.npath = -1;
    oc.kf_ibz_path = 1;
    oc.E = oc.U = oc.H = 1;
    diverge_model_to_file_finegrained( m, __FILE__ "_mod.dvg", &oc );

    // and free resources
    diverge_hartree_fock_free( h );
    diverge_model_free( m );

    diverge_finalize();
}

diverge_model_t* s2xs2_hubbard( int nk, int nkf, double ffcut, double U ) {
    diverge_model_t* m = diverge_model_init();
    m->lattice[0][0] = m->lattice[1][1] = m->lattice[2][2] = 1.0;

    m->n_orb = 1;
    m->n_spin = 2; m->SU2 = 0;
    m->nk[0] = m->nk[1] = nk;
    m->nkf[0] = m->nkf[1] = nkf;

    m->n_ibz_path = 4;
    m->ibz_path[1][0] = m->ibz_path[2][0] = m->ibz_path[2][1] = 0.5;

    m->hop = calloc(128, sizeof*m->hop);
    m->vert = calloc(128, sizeof*m->vert);

    for (int s=0; s<2; ++s) {
        m->hop[m->n_hop++] = (rs_hopping_t){ .R={ 1,0}, .s1=s, .s2=s, .t=1 };
        m->hop[m->n_hop++] = (rs_hopping_t){ .R={-1,0}, .s1=s, .s2=s, .t=1 };
        m->hop[m->n_hop++] = (rs_hopping_t){ .R={0, 1}, .s1=s, .s2=s, .t=1 };
        m->hop[m->n_hop++] = (rs_hopping_t){ .R={0,-1}, .s1=s, .s2=s, .t=1 };
    }

    int NR1[3] = {1,-1,0};
    int NR2[3] = {1,1,0};
    int NR3[3] = {0,0,1};
    rs_hopping_to_fractcell( m, NR1, NR2, NR3 );

    for (int o=0; o<m->n_orb; ++o)
        m->vert[m->n_vert++] = (rs_vertex_t){ .chan='D', .o1=o, .o2=o, .s1=-1, .V=U };

    if (diverge_model_validate( m )) diverge_mpi_exit( EXIT_FAILURE );
    diverge_model_internals_common( m );
    diverge_model_internals_tu( m, ffcut );
    return m;
}
