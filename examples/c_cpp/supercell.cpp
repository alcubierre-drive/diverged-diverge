#include <diverge.h>
#include <diverge_Eigen3.hpp>

#include <vector>
#include <array>
#include <fstream>

using std::vector;
using std::array;

diverge_model_t* supercell_model( index_t nk, index_t nkf );

int main(int argc, char** argv) {
    diverge_init( &argc, &argv );
    diverge_compilation_status();

    int nkc = 6,
        nkf = 1;
    char prefix[] = __FILE__ "_";

    diverge_model_t* m = supercell_model( nkc, nkf );

    char outfile[1024];
    diverge_model_output_set_npath( 50 );
    sprintf( outfile, "%smod.dvg", prefix );
    diverge_model_to_file( m, outfile, true );

    diverge_model_internals_tu( m, 1.01 );

    diverge_flow_step_t* s = diverge_flow_step_init( m, "tu", "PCD" );

    sprintf( outfile, "%sflow.dat", prefix );
    FILE* f = fopen( outfile, "w" );
    double vertmax = 0;
    diverge_euler_t eu = diverge_euler_defaults_CPP();
    do {
        diverge_flow_step_euler( s, eu.Lambda, eu.dLambda );
        diverge_flow_step_vertmax( s, &vertmax );
        double loopmax[2]; diverge_flow_step_loopmax( s, loopmax );
        double chanmax[3]; diverge_flow_step_chanmax( s, chanmax );
        char line[2048];
        sprintf( line, "%.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e\n",
                eu.Lambda, eu.dLambda, loopmax[0], loopmax[1], chanmax[0], chanmax[1], chanmax[2], vertmax );
        mpi_eprintf( "%s", line );
        fprintf( f, "%s", line );
        fflush( stderr );
    } while (diverge_euler_next( &eu, vertmax ));
    fclose( f );

    sprintf( outfile, "%sout.dvg", prefix );
    diverge_postprocess_and_write( s, NULL, outfile );
    diverge_flow_step_free( s );

    diverge_model_free( m );

    diverge_finalize();
}

// some variables and helper functions
static const double eps=1.e-7;
inline double get_U_Ohno( double r, double U, double a, double d ) {
    return U * a * std::exp(-r/d) / std::sqrt(a*a + r*r);
}
inline double fermi( double x ) {
    return 1./(std::exp(x)+1);
}

diverge_model_t* supercell_model( index_t nk, index_t nkf ) {

    diverge_model_t* mod = diverge_model_init();

    sprintf(mod->name, "supercell_model");
    mod->nk[0] = mod->nk[1] = nk;
    mod->nkf[0] = mod->nkf[1] = nkf;

    Vec2d g_a0(1,0);
    Vec2d g_a1 = Rot2d(M_PI/3)*g_a0;
    Mat2d g_latt; g_latt.col(0) = g_a0; g_latt.col(1) = g_a1;

    Vec2d g_o0(0,0);
    Vec2d g_o1 = (g_a0 + g_a1)/3.0;
    array<Vec2d,2> g_orb = {g_o0, g_o1};

    Mat2d to_super; to_super << 2,4,-4,6;
    Mat2d s_latt = (to_super * g_latt.transpose()).transpose();

    Mat2d s_inv = s_latt.inverse();

    Mat2d super_g = s_inv.transpose() * 2. * M_PI;

    vector<Vec2d> supercell;
    vector<int> sublattice;
    int search=20;
    for (int i=-search; i<=search; ++i)
    for (int j=-search; j<=search; ++j) {
        Vec2d g_vec = g_latt.col(0) * i + g_latt.col(1) * j;
        Vec2d X = (1.-eps)*s_inv * g_vec;
        if ((X(0) > 0.0+eps && X(0) < 1.0) && (X(1) > 0.0+eps && X(1) < 1.0)) {
            for (int o=0; o<2; ++o) {
                supercell.push_back(g_orb[o] + g_vec);
                sublattice.push_back(o);
            }
        }
    }
    int N = supercell.size();
    // save sublattice info
    mod->nbytes_data = sizeof(int)*N;
    mod->data = (char*)malloc(mod->nbytes_data);
    memcpy( mod->data, sublattice.data(), mod->nbytes_data );

    Map<Mat3d> lat((double*)mod->lattice);
    lat.block<2,2>(0,0) = s_latt;
    lat(2,2) = 1.0;

    mod->n_orb = N;
    for (int i=0; i<N; ++i) {
        Map<Vec3d> P((double*)mod->positions + 3*i);
        P.head(2) = supercell[i];
    }

    mod->SU2 = true;
    mod->n_spin = 1;
    mod->n_hop = 0;
    mod->hop = (rs_hopping_t*)calloc(1024*128,sizeof(rs_hopping_t));

    array<Vec2d,3> g_star = { Rot2d(0) * super_g.col(0),
                              Rot2d(2.*M_PI/3.) * super_g.col(0),
                              Rot2d(4.*M_PI/3.) * super_g.col(0) };
    auto hopparams = [g_o1]( double dist )->double{
        double a = g_o1.norm();
        double V_pp_pi = -2.7 * std::exp(3.14*(1.-dist/a));
        if (dist/a>5 || dist/a<eps || dist<eps)
            return 0;
        else
            return V_pp_pi;
    };

    for (int Rx=-2; Rx<=2; ++Rx)
    for (int Ry=-2; Ry<=2; ++Ry)
    for (int i=0; i<N; ++i)
    for (int j=0; j<N; ++j) {
        Vec2d R = Rx*s_latt.col(0) + Ry*s_latt.col(1);
        Vec2d d = supercell[i] - supercell[j] + R;
        double hop = hopparams(d.norm());
        if (std::abs(hop) > eps)
            mod->hop[mod->n_hop++] = {{Rx, Ry, 0}, i,j, 0,0, hop};
    }
    for (int i=0; i<N; ++i) {
        double Vpot = 0.0;
        for (Vec2d g: g_star)
             Vpot += 1.0 * std::cos( g.dot(supercell[i]) );
        mod->hop[mod->n_hop++] = {{0,0,0}, i,i, 0,0, Vpot};
    }

    vector<rs_vertex_t> Rvert;
    Rvert.reserve( 1024*1024 );
    #pragma omp parallel
    {
        vector<rs_vertex_t> Rvert_p;
        Rvert_p.reserve( 1024*1024 );
        #pragma omp for collapse(4)
        for (int Rx=-200; Rx<=200; ++Rx)
        for (int Ry=-200; Ry<=200; ++Ry)
        for (int j=0; j<N; ++j)
        for (int i=0; i<N; ++i) {
            Vec2d R = Rx*s_latt.col(0) + Ry*s_latt.col(1) + supercell[i] - supercell[j];
            double interaction = get_U_Ohno( R.norm(), 8.0, 0.3, 100.0 ) * fermi( (R.norm() - 10.0) * 5.0 );
            if (std::abs(interaction) > std::pow(eps,1./3.)) {
                rs_vertex_t V {'D',
                    {(Rx+100*mod->nk[0])%mod->nk[0],(Ry+100*mod->nk[1])%mod->nk[1],0},
                    i,j, -1,0,0,0, interaction };
                Rvert_p.push_back( V );
            }
        }
        #pragma omp critical
        {
            for (const rs_vertex_t& V: Rvert_p)
                Rvert.push_back( V );
        }
    }
    mod->n_vert = Rvert.size();
    mpi_usr_printf( "using %li vertex elements\n", mod->n_vert );
    mod->vert = (rs_vertex_t*)calloc(mod->n_vert,sizeof(rs_vertex_t));
    memcpy( mod->vert, Rvert.data(), sizeof(rs_vertex_t)*Rvert.size() );

    mod->n_ibz_path = 4;
    mod->ibz_path[1][0] = 0.5;
    mod->ibz_path[2][0] = 2./3.;
    mod->ibz_path[2][1] = 1./3.;

    if (diverge_model_validate( mod ))
        diverge_mpi_exit(EXIT_FAILURE);

    diverge_model_internals_common( mod );
    return mod;
}

