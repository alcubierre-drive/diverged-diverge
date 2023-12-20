#include <diverge.h>
#include <fftw3-mpi.h>
#include <fftw3.h>
#define LINALG_NO_LAPACKE
#include "linalg.h/linalg.h"
LINALG_IMPLEMENT

static const double tdd = 1.0,
                    tdp = 0.1,
                    tpp = 0.2,
                    U = 3.0,
                    J = 0.2,
                    ffdist = 1.2;

double hopphase( int o1, int o2, claV2d_t p1, claV2d_t p2 ) {
    claV2d_t dist = claV2d_sub( p1, p2 );
    if ((o1 == o2) && (o1 == 0))
        return 1.0;
    else if ((o1 == 0) && (o2 == 1))
        return p2.v[0] > 0 ? -1.0 : 1.0;
    else if ((o1 == 0) && (o2 == 2))
        return p2.v[1] > 0 ? 1.0 : -1.0;
    else if ((o1 == 1) && (o2 == 0))
        return dist.v[0] > 0 ? -1.0 : 1.0;
    else if ((o1 == 2) && (o2 == 0))
        return dist.v[1] > 0 ? 1.0 : -1.0;
    else if (dist.v[0] < 0)
        return dist.v[1] < 0 ? -1.0 : 1.0;
    else
        return dist.v[1] > 0 ? -1.0 : 1.0;
    return 0.0;
}

int main( int argc, char** argv ) {
    // init
    diverge_init( &argc, &argv );

    if (diverge_mpi_comm_rank() == 0) fftw_import_wisdom_from_filename("t2g.wisdom");
    if (diverge_mpi_get_comm() != NULL)
        fftw_mpi_broadcast_wisdom( diverge_mpi_get_comm() );

    mpi_loglevel_set( 5 );
    diverge_compilation_status();
    int nk = argc > 1 ? atoi(argv[1]) : 10,
        nkf = argc > 2 ? atoi(argv[2]) : 15;
    mpi_usr_printf( "using nk=%i and nkf=%i\n", nk, nkf );
    diverge_model_t* m = diverge_model_init();
    strcpy(m->name, __FILE__ );
    m->nk[0] = m->nk[1] = nk;
    m->nkf[0] = m->nkf[1] = nkf;
    m->lattice[0][0] = m->lattice[1][1] = m->lattice[2][2] = 1.0;
    m->n_orb = 3;
    m->SU2 = m->n_spin = 1;
    m->ibz_path[1][0] = m->ibz_path[2][0] = m->ibz_path[2][1] = 0.5;
    m->n_ibz_path = 4;
    m->positions[1][0] = m->positions[2][1] = 0.5;
    double tdd_norm = 1.0, tdp_norm = 0.5, tpp_norm = 0.5*sqrt(2.);
    m->hop = diverge_mem_alloc_rs_hopping_t(128);
    for (int Rx=-2; Rx<=2; ++Rx)
    for (int Ry=-2; Ry<=2; ++Ry)
    for (int o1=0; o1<3; ++o1)
    for (int o2=0; o2<3; ++o2) {
        claV2d_t p2 = claV2d_add(
                claV2d_add( claV2d_scale( claV2d_map(m->lattice[0]), Rx ),
                            claV2d_scale( claV2d_map(m->lattice[1]), Ry ) ),
                claV2d_map( m->positions[o2] ) );
        claV2d_t p1 = claV2d_map(m->positions[o1]);
        double phase = hopphase( o1, o2, p1, p2 ),
               dist = claV2d_norm( claV2d_sub( p2, p1 ) );
        if (fabs(dist-tdd_norm) < 1e-6 && o1 == 0 && o2 == 0)
            m->hop[m->n_hop++] = (rs_hopping_t){ .R={Rx,Ry,0}, .o1=o1, .o2=o2, .t=tdd*phase };
        if (fabs(dist-tpp_norm) < 1e-6 && o1 != 0 && o2 != 0)
            m->hop[m->n_hop++] = (rs_hopping_t){ .R={Rx,Ry,0}, .o1=o1, .o2=o2, .t=tpp*phase };
        if (fabs(dist-tdp_norm) < 1e-6)
            m->hop[m->n_hop++] = (rs_hopping_t){ .R={Rx,Ry,0}, .o1=o1, .o2=o2, .t=tdp*phase };
    }
    m->vert = diverge_mem_alloc_rs_vertex_t(128);
    for (int o1=0; o1<3; ++o1)
    for (int o2=0; o2<3; ++o2) {
        if (o1 == o2) {
            m->vert[m->n_vert++] = (rs_vertex_t){.chan='D', .R={0,0,0}, .o1=o1, .o2=o2, .V=U};
        } else {
            m->vert[m->n_vert++] = (rs_vertex_t){.chan='D', .R={0,0,0}, .o1=o1, .o2=o2, .V=U-2*J};
            m->vert[m->n_vert++] = (rs_vertex_t){.chan='C', .R={0,0,0}, .o1=o1, .o2=o2, .V=J};
            m->vert[m->n_vert++] = (rs_vertex_t){.chan='P', .R={0,0,0}, .o1=o1, .o2=o2, .V=J};
        }
    }
    if (diverge_model_validate( m ))
        mpi_usr_printf("warning: model not well behaved\n");
    diverge_model_internals_common( m );
    diverge_model_internals_any( m, "tu", ffdist );

    char chnk_[] = "8";
    char* chnk = chnk_;
    diverge_model_hack( m, "tu_mpifft_chunksize", argc > 3 ? argv[3] : chnk );
    diverge_model_hack( m, "tu_extra_propagator_timings", "1" );
    diverge_model_hack( m, "model_shared_gf", "1" );
    diverge_model_hack( m, "model_free_hamiltonian", "1" );

    // flow step & integrate
    diverge_flow_step_t* f = diverge_flow_step_init_any(m, "PCD");
    double Lambda = 11.0;

    double tick = diverge_mpi_wtime();
    for (int i=0; i<2; ++i) {
        mpi_usr_printf( "step %i…\n", i );
        diverge_flow_step_euler( f, Lambda--, -1.0 );
    }
    double tock = diverge_mpi_wtime();

    for (int i=0; i<diverge_flow_step_ntimings(f); ++i) {
        mpi_tim_printf( "%.2fs spent in %s\n",
                diverge_flow_step_timing(f, i),
                diverge_flow_step_timing_descr(f, i) );
    }

    if (diverge_mpi_comm_rank() == 0)
        printf("%.2fs\n", tock-tick);

    // cleanup
    diverge_flow_step_free(f);
    diverge_model_free(m);

    if (diverge_mpi_get_comm() != NULL)
        fftw_mpi_gather_wisdom( diverge_mpi_get_comm() );
    if (diverge_mpi_comm_rank() == 0) {
        // some proper import/export business
        fftw_import_wisdom_from_filename("t2g.wisdom");
        fftw_export_wisdom_to_filename("t2g.wisdom");
    }

    diverge_finalize();
    return EXIT_SUCCESS;
}

