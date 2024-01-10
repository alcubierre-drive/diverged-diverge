#include <diverge.h>
#include <diverge_Eigen3.hpp>

// parameters
const double U = 3.0,
             tdd = 0.2, tpp = 0.3, tdp = 1.0;

// orbitals in the 'main' model
const index_t omap[] = {1, 2};

// some parameters
const index_t nk = 24, nkf = 15;
const double filling = 0.8,
             tu_distance = 1.51;

// some uninteresting functions…
static double t2g_phase( index_t o1, index_t o2, Vec3d po1, Vec3d po2 );
static diverge_model_t* t2g_model_gen( index_t nk, index_t nkf );

// this function is passed to the 'main' model in order to destroy the attached
// one ('helper' model) upon destruction
static void data_destructor( char* addr ) {
    diverge_model_t* t2g_model = (diverge_model_t*)addr;
    diverge_model_free( t2g_model );
}

// here's where the magic happens. We generate the GF of the 'helper' model by
// interpreting the attached data ptr of the 'main' model as a diverge_model_t
// (i.e. the 'helper' model) and then calling the GF generator of the 'helper'
// model first. After that, we only copy over the needed entries.
static greensfunc_op_t subspace_gf( const diverge_model_t* model, complex128_t Lambda, gf_complex_t* buf ) {
    diverge_model_t* t2g_model = (diverge_model_t*)model->data;
    if (t2g_model) {
        // calc the full Green's function ('helper')
        gf_complex_t* buf_full = diverge_model_internals_get_greens( t2g_model );
        const index_t nb_full = t2g_model->n_spin * t2g_model->n_orb;
        t2g_model->gfill( t2g_model, Lambda, buf_full );

        // and then only look at the required subspace ('main')
        const index_t nkf = model->nk[0] * model->nk[1] * model->nk[2] *
                            model->nkf[0] * model->nkf[1] * model->nkf[2];
        const index_t nb = model->n_spin * model->n_orb;

        // shared_exclusive_enter() and shared_exclusive_wait() not needed for
        // standard compilation flags. ensures that only one process per node
        // fills the buffer in case shared mem GFs are activated (not default!)
        if (shared_exclusive_enter(buf)) {
            #pragma omp parallel for collapse(4) num_threads(diverge_omp_num_threads())
            for (int sign=0; sign<2; ++sign)
            for (index_t k=0; k<nkf; ++k)
            for (index_t o=0; o<nb; ++o) for (index_t p=0; p<nb; ++p)
                    buf[IDX4(sign, k, o, p, nkf, nb, nb)] = buf_full[IDX4(sign, k, omap[o], omap[p], nkf, nb_full, nb_full)];
        }
        shared_exclusive_wait(buf);
    }
    return greensfunc_op_cpu;
}

// we attach a custom Hamiltonian generator to the 'main' model because the
// notion of a 'Hamiltonian' is useless when only looking at a subspace.
static void subspace_ham( const diverge_model_t* model, complex128_t* buf ) {
    (void)model;
    (void)buf;
    return;
}

int main( int argc, char** argv ) {
    diverge_init( &argc, &argv );
    diverge_compilation_status();

    // alloc the 'main' model
    diverge_model_t* model = diverge_model_init();

    // gen the 'helper' model (in standard fashion: common internals & validate)
    diverge_model_t* t2g_model = t2g_model_gen( nk, nkf );
    diverge_model_set_filling( t2g_model, NULL, -1, filling );

    // set the data ptr of the 'main' model to the 'helper' model
    model->data = (char*)t2g_model;
    model->nbytes_data = sizeof(diverge_model_t);
    model->data_destructor = &data_destructor;

    // set parameters of the 'main' model (steal some from the 'helper')
    memcpy( model->lattice[0], t2g_model->lattice[0], sizeof(double)*9 );
    memcpy( model->positions[0], t2g_model->positions[1], sizeof(double)*6 );
    memcpy( model->nk, t2g_model->nk, sizeof(index_t)*2 );
    memcpy( model->nkf, t2g_model->nkf, sizeof(index_t)*2 );
    model->n_orb = 2;
    model->n_spin = model->SU2 = 1;
    model->vert = (rs_vertex_t*)calloc( 128, sizeof(rs_vertex_t) );
    model->vert[model->n_vert++] = rs_vertex_t{ 'D', {0,0,0}, 0,0, -1,0,0,0, U };
    model->vert[model->n_vert++] = rs_vertex_t{ 'D', {0,0,0}, 1,1, -1,0,0,0, U };

    // set the Hamiltonian & GF generators of the 'main' model
    model->hfill = &subspace_ham;
    model->gfill = &subspace_gf;

    // and be sure there's no problem
    if (diverge_model_validate( model ))
        diverge_mpi_exit( EXIT_FAILURE );
    diverge_model_internals_common( model );
    diverge_model_internals_tu( model, tu_distance );

    // flow step & flow ('main' only!)
    diverge_flow_step_t* step = diverge_flow_step_init_any( model, "PCD" );
    diverge_euler_t eu = diverge_euler_defaults_CPP();
    double cmax[3] = {0}, vmax = 0;
    do {
        diverge_flow_step_euler( step, eu.Lambda, eu.dLambda );
        diverge_flow_step_vertmax( step, &vmax );
        diverge_flow_step_chanmax( step, cmax );
        mpi_printf( "%.5e %.5e %.5e %.5e\n", eu.Lambda, cmax[0], cmax[1], cmax[2] );
    } while (diverge_euler_next( &eu, vmax ));
    diverge_flow_step_free( step );

    // free the 'main' model -> auto-destroy the 'helper' model
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

