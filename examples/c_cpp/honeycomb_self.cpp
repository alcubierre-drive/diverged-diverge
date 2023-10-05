#include <diverge.h>
#include <diverge_Eigen3.hpp>

#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <sstream>

#include <unistd.h>

using std::cout;
using std::endl;
using std::vector;
using std::array;
using std::string;

typedef Eigen::Array<complex128_t,-1,-1> AryXXcd;

static bool suppress_omega_config = false;

diverge_model_t* honeycomb( index_t nk, index_t nkf, double t, double tp, double U, double V, double V2 );

struct omega_config_t {
    double w0 = -1.5, w1 = 1.5;
    int nw = 300;
    double eta = 1.e-2;
    int use = 0;
    omega_config_t() {}
    omega_config_t( const char* str );
    void tostr( char* buf ) {
        sprintf( buf, "w0:%.5e,w1:%.5e,nw:%i,eta:%.5e,use:%i", w0, w1, nw, eta, use );
    }
};
MatXd calc_spectral_function( omega_config_t& wc, diverge_model_t* m );

int main(int argc, char** argv) {

    diverge_init( &argc, &argv );
    diverge_compilation_status();

    // parameters
    int nkc = 6,
        nkf = 1;
    bool flow = 0,
         self = 0,
         refill = 0,
         ldos_flow = 0;
    double ffdist = 1.1,
           t = 1.0,
           tp = 0.0,
           U = 5.0,
           V = 0.0,
           V2 = 0.0,
           nu = 0.7;
    char prefix[1024] = __FILE__ "_";
    char omega_config[1024] = "";

    // argument parser
    char pad[512] = "";
    sprintf( pad, "%*s", (int)strlen(argv[0]), " " );
    int opt;
    while ((opt = getopt(argc, argv, "K:k:FSRLf:T:t:U:v:V:n:p:w:h")) != -1) {
        switch (opt) {
            case 'K': nkc = atoi(optarg); break;
            case 'k': nkf = atoi(optarg); break;
            case 'F': flow = 1; break;
            case 'S': self = 1; break;
            case 'R': refill = 1; break;
            case 'L': ldos_flow = 1; break;
            case 'f': ffdist = atof(optarg); break;
            case 'T': t = atof(optarg); break;
            case 't': tp = atof(optarg); break;
            case 'U': U = atof(optarg); break;
            case 'V': V = atof(optarg); break;
            case 'v': V2 = atof(optarg); break;
            case 'n': nu = atof(optarg); break;
            case 'p': strcpy(prefix, optarg); break;
            case 'w': strcpy(omega_config, optarg); break;
            case 'h': mpi_eprintf( "Usage: %s [-K nk_coarse !U] [-k nk_fine !U] [-F:flow] [-S:self] [-R:refill] [-L:ldos_flow]\n"
                                   "       %s [-f ffdist] [-T t !U] [-t tp !U] [-U U] [-V V] [-v V2] [-n nu !U] [-p prefix !U]\n"
                                   "       %s [-w omega_config !U]\n\n"
                                   "All paramaters with !U affect the non-interacting model. Others only relevant for\n"
                                   "interacting model.\n",
                                   argv[0], pad, pad );
                      diverge_mpi_exit(0); break;
            case '?':
            default:
                      mpi_eprintf( "'%s -h' for help.\n", argv[0] );
                      diverge_mpi_exit(1); break;
        }
    }
    argc -= optind;
    argv += optind;

    // model
    diverge_model_t* m = honeycomb( nkc, nkf, t, tp, U, V, V2 );
    diverge_model_set_filling( m, NULL, -1, nu );

    // calc DOS?
    omega_config_t wc(omega_config);
    if (wc.use) {
        VecXcd omega = VecXcd::LinSpaced(wc.nw, wc.w0-I128*wc.eta, wc.w1-I128*wc.eta);
        std::ofstream Afile( string(prefix) + "spec.dat" );
        Afile << omega.transpose().real() << endl << omega.transpose().imag()
              << endl << calc_spectral_function( wc, m ) << endl;
    }

    // output model to file
    diverge_model_output_set_npath( -1 );
    char outfile[2048] = "";
    sprintf( outfile, "%smod.dvg", prefix );
    diverge_model_to_file( m, outfile );

    if (flow) {
        // only init tu internals if flow
        diverge_model_internals_tu( m, ffdist );
        // and the flow step
        diverge_flow_step_t* s = diverge_flow_step_init( m, "tu", self ? "PCDS" : "PCD\0" );

        // for refilling: some sizes
        index_t nk = m->nk[0] * m->nk[1] * m->nk[2] * m->nkf[0] * m->nkf[1] * m->nkf[2],
                nb = m->n_orb * m->n_spin;
        // if using selfenergy allocate the workspace for refill. The code can
        // do without, but if we want access to the effective band structure at
        // each flow step this is the way to go.
        void* workspace = self ? calloc( nk*nb*nb*2 + nk*nb, sizeof(double) ) : NULL;
        // the first part of the workspace is used for the orbital to band
        // matrices. the second part for the energies.
        double* E_work = (double*)workspace + nk*nb*nb*2;
        // we need the *actual* filling
        double nu_real = diverge_model_get_filling( m, NULL, -1 );
        // array to save the flowing band structure in
        double *flowbands = NULL;
        index_t n_flowbands = 0;
        // arrays to save the kmesh indices to obtain the flowing band structure
        index_t *pts, n_pts;
        index_t *n_per_segment = diverge_kmesh_to_bands( m, &pts, &n_pts );

        // the flowing DOS file
        sprintf( outfile, "%sflowdos.dat", prefix );
        FILE* dos_file = fopen(outfile, "w");

        // the flowing LDOS file
        FILE* ldos_file = NULL;
        if (ldos_flow) {
            sprintf( outfile, "%sflow_ldos.bin", prefix );
            ldos_file = fopen( outfile, "w" );
        }

        // the flow file
        sprintf( outfile, "%sflow.dat", prefix );
        FILE* flow_file = fopen( outfile, "w" );

        // setup the integrator
        double vertmax = 0;
        diverge_euler_t eu = diverge_euler_defaults_CPP();
        eu.dLambda_fac = 0.1;
        eu.Lambda = 100;
        eu.dLambda = -10;
        eu.consider_maxvert_iter_start = 50;
        eu.consider_maxvert_lambda = 10;
        eu.maxvert = 30.0;
        do {
            // perform flow step
            diverge_flow_step_euler( s, eu.Lambda, eu.dLambda );

            // output the flow data
            diverge_flow_step_vertmax( s, &vertmax );
            double loopmax[2]; diverge_flow_step_loopmax( s, loopmax );
            double chanmax[3]; diverge_flow_step_chanmax( s, chanmax );
            char line[2048];
            sprintf( line, "%.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e\n",
                    eu.Lambda, eu.dLambda, loopmax[0], loopmax[1], chanmax[0], chanmax[1], chanmax[2], vertmax );
            mpi_eprintf( "%s", line );
            fflush( stderr );
            fprintf( flow_file, "%s", line );
            fflush( flow_file );

            if (self) {
                index_t nb = m->n_orb;
                // only refill the system if asked to do so, otherwise just
                // calculate the self-energy effective Hamiltonian and
                // diagonalize it
                if (refill)
                    diverge_flow_step_refill( s, nu_real, workspace );
                else
                    diverge_flow_step_refill_Hself( s, nu_real, workspace );

                // save the flowing band structure
                flowbands = (double*)realloc( flowbands, sizeof(double)*(n_flowbands + n_pts*nb) );
                for (index_t p=0; p<n_pts; ++p)
                    for (int b=0; b<nb; ++b)
                        flowbands[n_flowbands+p*nb+b] = E_work[pts[p]*nb+b];
                n_flowbands += n_pts*nb;

                // save the flowing LDOS/DOS
                if (wc.use) {
                    suppress_omega_config = true;
                    MatXd Amat = calc_spectral_function( wc, m );
                    VecXd A = Amat.colwise().sum();
                    for (double d: A) fprintf(dos_file, "%.5e ", d);
                    fprintf( dos_file, "\n" );
                    fflush( dos_file );

                    if (ldos_flow) {
                        fwrite( Amat.data(), sizeof(double), Amat.size(), ldos_file );
                        fflush( ldos_file );
                    }
                }
            }
        } while (diverge_euler_next( &eu, vertmax ));

        // clean up
        fclose( flow_file );
        if (workspace) free(workspace);
        fclose(dos_file);
        free( pts );
        free( n_per_segment );

        if (flowbands) {
            // write flowbands to file
            sprintf( outfile, "%sflowbands.bin", prefix );
            FILE* fb_file = fopen(outfile, "w");
            fwrite( flowbands, sizeof(double), n_flowbands, fb_file );
            fclose( fb_file );
            free(flowbands);
        }

        // ldos output
        if (ldos_flow) fclose( ldos_file );

        // postprocessing
        sprintf( outfile, "%sout.dvg", prefix );
        diverge_postprocess_conf_t pc = diverge_postprocess_conf_defaults_CPP();
        pc.tu_storing_threshold = 0.8;
        pc.tu_storing_relative = true;
        pc.tu_selfenergy = true;
        diverge_postprocess_and_write_finegrained( s, outfile, &pc );

        // flow step cleanup
        diverge_flow_step_free( s );
    }

    // model cleanup
    diverge_model_free( m );

    // library cleanup
    diverge_finalize();
}

diverge_model_t* honeycomb( index_t nk, index_t nkf, double t, double tp, double U, double V, double V2 ) {
    diverge_model_t* m = diverge_model_init();

    m->lattice[0][0] = m->lattice[1][0] = sin(M_PI/3.);
    m->lattice[0][1] = m->lattice[1][1] = cos(M_PI/3.);
    m->lattice[0][1] *= -1.0;
    m->lattice[2][2] = 1.0;
    Map<Mat3d> pos(m->positions[0]);
    Map<Mat3d> lat(m->lattice[0]);
    pos.col(0) = -1./3. * (lat.col(0) + lat.col(1));
    pos.col(1) =  1./3. * (lat.col(0) + lat.col(1));
    m->n_orb = 2;
    m->n_spin = 1;
    m->SU2 = 1;
    m->nk[0] = m->nk[1] = nk;
    m->nkf[0] = m->nkf[1] = nkf;
    m->n_ibz_path = 4;
    m->ibz_path[1][0] = 0.5;
    m->ibz_path[2][0] = 2./3.;
    m->ibz_path[2][1] = 1./3.;
    strcpy( m->name, __FILE__ );

    double t_norm = pos.col(0).norm();
    double tp_norm = 1.0;

    m->hop = (rs_hopping_t*)calloc(sizeof(rs_hopping_t), 1024);
    m->vert = (rs_vertex_t*)calloc(sizeof(rs_vertex_t), 1024);
    m->vert[m->n_vert++] = rs_vertex_t{ 'D', {0,0,0}, 0,0, -1,0,0,0, U };
    m->vert[m->n_vert++] = rs_vertex_t{ 'D', {0,0,0}, 1,1, -1,0,0,0, U };
    for (int Rx=-5; Rx<=5; ++Rx)
    for (int Ry=-5; Ry<=5; ++Ry)
    for (int o=0; o<2; ++o)
    for (int p=0; p<2; ++p) {
        double dist = (lat.col(0) * Rx + lat.col(1) * Ry + pos.col(p) - pos.col(o)).norm();
        if (std::abs(dist - t_norm) < 1e-5) {
            m->hop[m->n_hop++] = rs_hopping_t{ {Rx,Ry,0}, o,p, 0,0, t };
            m->vert[m->n_vert++] = rs_vertex_t{ 'D', {Rx,Ry,0}, o,p, -1,0,0,0, V };
        }
        if (std::abs(dist - tp_norm) < 1e-5) {
            m->hop[m->n_hop++] = rs_hopping_t{ {Rx,Ry,0}, o,p, 0,0, tp };
            m->vert[m->n_vert++] = rs_vertex_t{ 'D', {Rx,Ry,0}, o,p, -1,0,0,0, V2 };
        }
    }

    m->orb_symmetries = (complex128_t*)malloc(sizeof(complex128_t)*12*4);
    site_descr_t sites[2];
    for (int s=0; s<2; ++s) {
        sites[s].amplitude[0] = 1.0;
        sites[s].function[0] = orb_s;
        sites[s].n_functions = 1;
    }
    sym_op_t sym;
    Map<Vec3d> sym_nvec(sym.normal_vector);
    sym_nvec = Vec3d(0,0,1);
    sym.type = 'R';
    for (int i=0; i<6; ++i) {
        sym.angle = 60*i;
        diverge_generate_symm_trafo( 1, sites, 2, &sym, 1, m->rs_symmetries[m->n_sym][0],
                m->orb_symmetries+m->n_sym*4 );
        m->n_sym++;
    }
    sym.type = 'M';
    sym_nvec *= 0.0;
    for (int i=0; i<6; ++i) {
        sym_nvec.head(2) = Rot2d(M_PI/6.0 * i).toRotationMatrix() * Vec2d(1,0);
        diverge_generate_symm_trafo( 1, sites, 2, &sym, 1, m->rs_symmetries[m->n_sym][0],
                m->orb_symmetries+m->n_sym*4 );
        m->n_sym++;
    }

    if (diverge_model_validate( m ))
        diverge_mpi_exit(EXIT_FAILURE);

    diverge_model_internals_common( m );
    return m;
}

omega_config_t::omega_config_t( const char* str_ ) {
    vector<string> single_configs;
    char *token, *str, *tofree;
    tofree = str = strdup(str_);
    while ((token = strsep(&str, ","))) {
        single_configs.push_back(token);
    }
    free( tofree );
    for (const string& s_: single_configs) {
        char* s = strdup(s_.c_str());
        #define find_pattern( pattern_name, conversion ) { \
            char* s_##pattern_name = strstr( s, #pattern_name ":" ); \
            if (s_##pattern_name) pattern_name = conversion( s + strlen(#pattern_name ":") ); \
        }
        find_pattern( w0, atof );
        find_pattern( w1, atof );
        find_pattern( nw, atoi );
        find_pattern( eta, atof );
        find_pattern( use, atoi );
        free(s);
    }
}

MatXd calc_spectral_function( omega_config_t& wc, diverge_model_t* m ) {
    char sbuf[512];
    wc.tostr( sbuf );
    if (!suppress_omega_config) mpi_usr_printf( "%s\n", sbuf );
    VecXcd omega = VecXcd::LinSpaced(wc.nw, wc.w0-I128*wc.eta, wc.w1-I128*wc.eta);
    index_t no = m->n_orb,
            nk = m->nk[0] * m->nk[1] * m->nk[2] *
                 m->nkf[0] * m->nkf[1] * m->nkf[2];

    double* E_ = diverge_model_internals_get_E( m );
    complex128_t* U_ = diverge_model_internals_get_U( m );

    MatXcd A = MatXcd::Zero( wc.nw, no );

    AryXXcd E_W = AryXXcd::Zero( wc.nw, no );
    #pragma omp parallel for
    for (index_t k=0; k<nk; ++k) {
        Map<MatXcd> U(U_ + k*no*no, no, no);
        Map<VecXd> E(E_ + k*no, no);
        for (index_t b=0; b<no; ++b)
        for (index_t w=0; w<wc.nw; ++w)
            E_W(w,b) = 1./(E(b) - omega(w));
        MatXcd A_add = E_W.matrix() * U.array().abs().pow(2).matrix().transpose();
        #pragma omp critical
        A += A_add;
    }

    return -A.imag().transpose();
}

