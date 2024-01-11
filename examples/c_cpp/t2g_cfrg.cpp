#include <diverge.h>
#include <diverge_Eigen3.hpp>

// parameters
const double Ud = 3.0, Up = 1.0,
             tdd = 0.2, tpp = 0.3, tdp = 1.0;

// some parameters
const index_t nk = 24, nkf = 5;
const double filling = 0.8,
             tu_distance = 1.51;

// some uninteresting functions…
static double t2g_phase( index_t o1, index_t o2, Vec3d po1, Vec3d po2 );
static diverge_model_t* t2g_model_gen( index_t nk, index_t nkf );

// here's where we generate the subspace Green's function
static greensfunc_op_t subspace_gf( const diverge_model_t* model, complex128_t Lambda, gf_complex_t* buf ) {

    // orbitals that should be contained in the subspace
    const index_t orbsub[] = {0};
    const index_t n_orbsub = sizeof(orbsub)/sizeof(orbsub[0]);

    double* E = diverge_model_internals_get_E( model );
    complex128_t* U = diverge_model_internals_get_U( model );

    const index_t nkf = model->nk[0] * model->nk[1] * model->nk[2] *
                        model->nkf[0] * model->nkf[1] * model->nkf[2];
    const index_t nb = model->n_spin * model->n_orb;

    // set everything to zero
    memset( (void*)buf, 0, sizeof(gf_complex_t) * nkf * nb*nb );

    for (int sign=0; sign<2; ++sign) {
        complex128_t L = sign ? Lambda : std::conj(Lambda);
        #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
        for (index_t k=0; k<nkf; ++k)
        // only iterate over the subspace
        for (index_t oo=0; oo<n_orbsub; ++oo)
        for (index_t pp=0; pp<n_orbsub; ++pp) {
            // and find the right indices
            index_t o = orbsub[oo], p = orbsub[pp];
            // now fill the GF buffer only for those indices.
            complex128_t tmp = 0.0;
            for (index_t b=0; b<nb; ++b)
                tmp += U[IDX3(k,b,o,nb,nb)] * std::conj(U[IDX3(k,b,p,nb,nb)]) / (L - E[IDX2(k,b,nb)]);
            buf[IDX4(sign, k, o, p, nkf, nb, nb)] = tmp;
        }
    }
    return greensfunc_op_cpu;
}

int main( int argc, char** argv ) {
    // init
    diverge_init( &argc, &argv );
    diverge_compilation_status();

    // gen the kinetics
    diverge_model_t* model = t2g_model_gen( nk, nkf );
    diverge_model_set_filling( model, NULL, -1, filling );

    // set some vertex elements
    model->vert = (rs_vertex_t*)calloc( 128, sizeof(rs_vertex_t) );
    model->vert[model->n_vert++] = rs_vertex_t{ 'D', {0,0,0}, 0,0, -1,0,0,0, Ud };
    model->vert[model->n_vert++] = rs_vertex_t{ 'D', {0,0,0}, 1,1, -1,0,0,0, Up };
    model->vert[model->n_vert++] = rs_vertex_t{ 'D', {0,0,0}, 2,2, -1,0,0,0, Up };

    // set the cfrg GF
    model->gproj = &subspace_gf;

    // and be sure there's no problem
    if (diverge_model_validate( model ))
        diverge_mpi_exit( EXIT_FAILURE );
    diverge_model_internals_common( model );
    diverge_model_internals_tu( model, tu_distance );

    // flow step & flow
    diverge_flow_step_t* step = diverge_flow_step_init_any( model, "PCD" );
    diverge_euler_t eu = diverge_euler_defaults_CPP();
    double cmax[3] = {0}, vmax = 0;
    mpi_printf( "# Lambda Pmax Cmax Dmax\n" );
    do {
        diverge_flow_step_euler( step, eu.Lambda, eu.dLambda );
        diverge_flow_step_vertmax( step, &vmax );
        diverge_flow_step_chanmax( step, cmax );
        mpi_printf( "%.5e %.5e %.5e %.5e\n", eu.Lambda, cmax[0], cmax[1], cmax[2] );
    } while (diverge_euler_next( &eu, vmax ));
    diverge_flow_step_free( step );

    // cleanup
    diverge_model_free( model );
    diverge_finalize();
}

// some uninteresting functions…
static double t2g_phase( index_t o1, index_t o2, Vec3d po1, Vec3d po2 ) {
    if (o1 == o2 && o1 == 0) {
        return 1.0;
    } else if (o1 == 0 && o2 == 1) {
        return po2[0] > 0 ? -1.0 : 1.0;
    } else if (o1 == 0 && o2 == 2) {
        return po2[1] > 0 ? 1.0 : -1.0;
    } else if (o1 == 1 && o2 == 0) {
        Vec3d dist = po1-po2;
        return dist[0] > 0 ? -1.0 : 1.0;
    } else if (o1 == 2 && o2 == 0) {
        Vec3d dist = po1-po2;
        return dist[1] > 0 ? 1.0 : -1.0;
    } else {
        Vec3d dist = po2-po1;
        if (dist[0] > 0)
            return dist[1] > 0 ? -1.0 : 1.0;
        else
            return dist[1] < 0 ? -1.0 : 1.0;
    }
}

// some uninteresting functions…
static diverge_model_t* t2g_model_gen( index_t nk, index_t nkf ) {
    diverge_model_t* t2g_model = diverge_model_init();

    t2g_model->nk[0] = t2g_model->nk[1] = nk;
    t2g_model->nkf[0] = t2g_model->nkf[1] = nkf;

    t2g_model->SU2 = t2g_model->n_spin = 1;
    for (int i=0; i<3; ++i) t2g_model->lattice[i][i] = 1.0;
    t2g_model->positions[1][0] = t2g_model->positions[2][1] = 0.5;

    t2g_model->n_orb = 3;

    Map<Mat3d> lattice(t2g_model->lattice[0]),
               positions(t2g_model->positions[0]);
    t2g_model->hop = (rs_hopping_t*)calloc( 1024, sizeof(rs_hopping_t) );
    double tdd_norm = 1.0,
           tdp_norm = 0.5,
           tpp_norm = 0.5*sqrt(2.);
    for (int Rx=-2; Rx<=2; ++Rx) for (int Ry=-2; Ry<=2; ++Ry)
    for (int o1=0; o1<3; ++o1) for (int o2=0; o2<3; ++o2) {
        double dist = ( lattice.col(0)*Rx + lattice.col(1)*Ry + positions.col(o2) - positions.col(o1) ).norm();
        double phase = t2g_phase( o1, o2, positions.col(o1), lattice.col(0)*Rx + lattice.col(1)*Ry + positions.col(o2) );
        if (fabs( dist-tdd_norm ) < 1e-5 && o1 == 0 && o2 == 0)
            t2g_model->hop[t2g_model->n_hop++] = rs_hopping_t{ {Rx, Ry, 0}, o1, o2, 0,0, tdd*phase };
        if (fabs( dist-tpp_norm ) < 1e-5 && o1 != 0 && o2 != 0)
            t2g_model->hop[t2g_model->n_hop++] = rs_hopping_t{ {Rx, Ry, 0}, o1, o2, 0,0, tpp*phase };
        if (fabs( dist-tdp_norm ) < 1e-5)
            t2g_model->hop[t2g_model->n_hop++] = rs_hopping_t{ {Rx, Ry, 0}, o1, o2, 0,0, tdp*phase };
    }

    t2g_model->ibz_path[1][1] = t2g_model->ibz_path[2][0] = t2g_model->ibz_path[2][1] = 0.5;
    t2g_model->n_ibz_path = 4;

    if (diverge_model_validate( t2g_model ))
        diverge_mpi_exit( EXIT_FAILURE );

    diverge_model_internals_common( t2g_model );
    return t2g_model;
}

