#define CT_SUITE_NAME phonons
#include "ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../src/diverge.h"
#include "../src/diverge_phonons.h"

static diverge_model_t* hubbard_ssh_model( int SU2, int nk ) {
    double omega=1.0;
    double g=2.0;
    double t=1.0;
    double max_dist=1.01;
    index_t nkf=5;

    diverge_model_t* m = diverge_model_init();
    strcpy(m->name, "hubbard ssh");
    m->nk[0] = m->nk[1] = nk;
    m->nkf[0] = m->nkf[1] = nkf;
    m->lattice[0][0] = m->lattice[1][1] = m->lattice[2][2] = 1.0;
    m->n_spin = (m->SU2=SU2) ? 1 : 2;
    m->n_orb = 1;

    m->hop = calloc(64, sizeof*m->hop);
    for (int i=0; i<4; ++i) {
        m->hop[m->n_hop].R[i/2] = i%2 ? 1:-1;
        m->hop[m->n_hop].s1 = -1; // auto SU2
        m->hop[m->n_hop++].t = t;
    }
    m->vert = calloc(1, sizeof*m->vert);

    diverge_model_internals_common(m);

    rs_hopping_t phop[] = {
        { .o1=0, .o2=0, .t=omega },
        { .o1=1, .o2=1, .t=omega },
    };
    ph_vertex_t pvert[] = {
        {.mode='x', .R12={ 1, 0, 0}, .s1=-1, .OA=0, .RA_pos={+0.5, 0, 0 }, .g=g },
        {.mode='x', .R12={-1, 0, 0}, .s1=-1, .OA=0, .RA_pos={-0.5, 0, 0 }, .g=g },
        {.mode='y', .R12={ 0, 1, 0}, .s1=-1, .OA=1, .RA_pos={ 0,+0.5, 0 }, .g=g },
        {.mode='y', .R12={ 0,-1, 0}, .s1=-1, .OA=1, .RA_pos={ 0,-0.5, 0 }, .g=g },
    };
    diverge_model_internals_tu_phonons(m, max_dist, phop, sizeof(phop)/sizeof(phop[0]),
            pvert, sizeof(pvert)/sizeof(pvert[0]));
    return m;
}

CT_TEST_NAME(hubbard_SSH_model, "hubbard-ssh,SU2") {
    diverge_model_t* m = hubbard_ssh_model(1, 12);
    diverge_flow_step_t* st = diverge_flow_step_init_any(m, "PCD");
    diverge_euler_t eu = diverge_euler_defaults_CPP();
    double cmax[3], vmax;
    VECTOR_INIT(double, PCD_max);
    do {
        diverge_flow_step_euler( st, eu.Lambda, eu.dLambda );
        diverge_flow_step_chanmax( st, cmax );
        diverge_flow_step_vertmax( st, &vmax );
        double Pmax = diverge_flow_step_eigchan( st, 'p', 0 ),
               Cmax = diverge_flow_step_eigchan( st, 'm', 0 ),
               Dmax = diverge_flow_step_eigchan( st, 'h', 0 );
        VECTOR_PUSH_BACK(PCD_max, Pmax);
        VECTOR_PUSH_BACK(PCD_max, Cmax);
        VECTOR_PUSH_BACK(PCD_max, Dmax);
        mpi_log_printf("%s: %5.2f  %5.2f %5.2f %5.2f…", m->name, eu.Lambda, Pmax, Cmax, Dmax);
        if (mpi_loglevel_get() >= 4) mpi_eprintf( "\n%c[1A", 27 ), fflush(stderr);
    } while (diverge_euler_next(&eu, vmax));
    if (mpi_loglevel_get() >= 4) mpi_eprintf( "\n" ), fflush(stderr);

    diverge_flow_step_free( st );
    diverge_model_free( m );

    for (index_t i=0; i<PCD_max_sz/3; ++i) {
        double PC = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,1,3)]),
               PD = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,2,3)]);
        CT_CHECK( PC < 1e-4 );
        CT_CHECK( PD < 1e-4 );
    }
    VECTOR_FREE(PCD_max);
}

CT_TEST_NAME(hubbard_SSH_model_nonSU2, "hubbard-ssh,non-SU2") {
    diverge_model_t* m = hubbard_ssh_model(0, 8);
    diverge_flow_step_t* st = diverge_flow_step_init_any(m, "PCD");
    diverge_euler_t eu = diverge_euler_defaults_CPP();
    double cmax[3], vmax;
    VECTOR_INIT(double, PCD_max);
    do {
        diverge_flow_step_euler( st, eu.Lambda, eu.dLambda );
        diverge_flow_step_chanmax( st, cmax );
        diverge_flow_step_vertmax( st, &vmax );
        double Pmax = diverge_flow_step_eigchan( st, 'p', 0 ),
               Cmax = diverge_flow_step_eigchan( st, 'c', 0 ),
               Dmax = diverge_flow_step_eigchan( st, 'd', 0 );
        VECTOR_PUSH_BACK(PCD_max, 0.5*Pmax);
        VECTOR_PUSH_BACK(PCD_max, Cmax);
        VECTOR_PUSH_BACK(PCD_max, Dmax);
        mpi_log_printf("%s: %5.2f  %5.2f %5.2f %5.2f…", m->name, eu.Lambda, Pmax, Cmax, Dmax);
        if (mpi_loglevel_get() >= 4) mpi_eprintf( "\n%c[1A", 27 ), fflush(stderr);
    } while (diverge_euler_next(&eu, vmax));
    if (mpi_loglevel_get() >= 4) mpi_eprintf( "\n" ), fflush(stderr);

    diverge_flow_step_free( st );
    diverge_model_free( m );

    for (index_t i=0; i<PCD_max_sz/3; ++i) {
        double PC = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,1,3)]),
               PD = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,2,3)]);
        CT_CHECK( PC < 1e-4 );
        CT_CHECK( PD < 1e-4 );
    }
    VECTOR_FREE(PCD_max);
}

CT_TEST_NAME(hubbard_SSH_model_self, "hubbard-ssh,SU2,self") {
    diverge_model_t* m = hubbard_ssh_model(1, 12);
    double nu = diverge_model_get_filling(m, NULL, -1);
    diverge_flow_step_t* st = diverge_flow_step_init_any(m, "PCDS");
    diverge_euler_t eu = diverge_euler_defaults_CPP();
    double cmax[3], vmax;
    VECTOR_INIT(double, PCD_max);
    int ll = mpi_loglevel_get(); mpi_loglevel_set(4);
    do {
        diverge_flow_step_euler( st, eu.Lambda, eu.dLambda );
        diverge_flow_step_refill( st, nu, NULL );
        diverge_flow_step_chanmax( st, cmax );
        diverge_flow_step_vertmax( st, &vmax );
        double Pmax = diverge_flow_step_eigchan( st, 'p', 0 ),
               Cmax = diverge_flow_step_eigchan( st, 'm', 0 ),
               Dmax = diverge_flow_step_eigchan( st, 'h', 0 );
        VECTOR_PUSH_BACK(PCD_max, Pmax);
        VECTOR_PUSH_BACK(PCD_max, Cmax);
        VECTOR_PUSH_BACK(PCD_max, Dmax);
        mpi_log_printf("%s: %5.2f  %5.2f %5.2f %5.2f…", m->name, eu.Lambda, Pmax, Cmax, Dmax);
        if (mpi_loglevel_get() >= 4) mpi_eprintf( "\n%c[1A", 27 ), fflush(stderr);
    } while (diverge_euler_next(&eu, vmax));
    mpi_loglevel_set(ll);
    if (mpi_loglevel_get() >= 4) mpi_eprintf( "\n" ), fflush(stderr);

    diverge_flow_step_free( st );
    diverge_model_free( m );

    for (index_t i=0; i<PCD_max_sz/3; ++i) {
        double PC = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,1,3)]),
               PD = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,2,3)]);
        CT_CHECK( PC < 1e-4 );
        CT_CHECK( PD < 1e-4 );
    }
    VECTOR_FREE(PCD_max);
}

CT_TEST_NAME(hubbard_SSH_model_nonSU2_self, "hubbard-ssh,non-SU2,self") {
    diverge_model_t* m = hubbard_ssh_model(0, 8);
    double nu = diverge_model_get_filling(m, NULL, -1);
    diverge_flow_step_t* st = diverge_flow_step_init_any(m, "PCDS");
    diverge_euler_t eu = diverge_euler_defaults_CPP();
    double cmax[3], vmax;
    VECTOR_INIT(double, PCD_max);
    int ll = mpi_loglevel_get(); mpi_loglevel_set(4);
    do {
        diverge_flow_step_euler( st, eu.Lambda, eu.dLambda );
        diverge_flow_step_refill( st, nu, NULL );
        diverge_flow_step_chanmax( st, cmax );
        diverge_flow_step_vertmax( st, &vmax );
        double Pmax = diverge_flow_step_eigchan( st, 'p', 0 ),
               Cmax = diverge_flow_step_eigchan( st, 'c', 0 ),
               Dmax = diverge_flow_step_eigchan( st, 'd', 0 );
        VECTOR_PUSH_BACK(PCD_max, 0.5*Pmax);
        VECTOR_PUSH_BACK(PCD_max, Cmax);
        VECTOR_PUSH_BACK(PCD_max, Dmax);
        mpi_log_printf("%s: %5.2f  %5.2f %5.2f %5.2f…", m->name, eu.Lambda, Pmax, Cmax, Dmax);
        if (mpi_loglevel_get() >= 4) mpi_eprintf( "\n%c[1A", 27 ), fflush(stderr);
    } while (diverge_euler_next(&eu, vmax));
    mpi_loglevel_set(ll);
    if (mpi_loglevel_get() >= 4) mpi_eprintf( "\n" ), fflush(stderr);

    diverge_flow_step_free( st );
    diverge_model_free( m );

    for (index_t i=0; i<PCD_max_sz/3; ++i) {
        double PC = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,1,3)]),
               PD = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,2,3)]);
        CT_CHECK( PC < 1e-4 );
        CT_CHECK( PD < 1e-4 );
    }
    VECTOR_FREE(PCD_max);
}

#endif
