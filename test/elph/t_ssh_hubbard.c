#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge.h"
#include "../../src/diverge_phonons.h"

#define CT_SUITE_NAME phonons
#include "../ctest.h"

static diverge_model_t* hubbard_ssh_model() {
    double omega=1.0;
    double lambda=0.1;
    double t=1.0;
    double U=0.0;
    double VP=0.0;
    double VD=0.0;
    double max_dist=1.01;
    index_t nk=12;
    index_t nkf=5;

    double g = sqrt(lambda * (8.*t) * omega);
    diverge_model_t* m = diverge_model_init();
    strcpy(m->name, "hubbard ssh");
    m->nk[0] = m->nk[1] = nk;
    m->nkf[0] = m->nkf[1] = nkf;
    m->lattice[0][0] = m->lattice[1][1] = m->lattice[2][2] = 1.0;
    m->hop = (rs_hopping_t*)calloc(4, sizeof*m->hop);
    for (int i=0; i<4; ++i) {
        m->hop[m->n_hop].R[i/2] = i%2 ? 1:-1;
        m->hop[m->n_hop++].t = t;
    }
    m->vert = (rs_vertex_t*)calloc(9, sizeof*m->vert);
    m->vert[m->n_vert].chan = 'C'; m->vert[m->n_vert].s1 = -1; m->vert[m->n_vert++].V = U;
    for (int c=0; c<2; ++c)
    for (int i=0; i<4; ++i) {
        m->vert[m->n_vert].chan = c?'P':'D';
        m->vert[m->n_vert].R[i/2] = i%2 ? 1:-1;
        m->vert[m->n_vert].s1 = -1;
        m->vert[m->n_vert++].V = c?VP:VD;
    }
    m->n_spin = m->n_orb = m->SU2 = 1;
    m->n_ibz_path = 4;
    m->ibz_path[1][0] = m->ibz_path[2][0] = m->ibz_path[2][1] = 0.5;
    diverge_model_internals_common(m);

    rs_hopping_t phop[] = {
        { {0,0,0}, 0,0, 0,0, omega },
        { {0,0,0}, 1,1, 0,0, omega },
    };
    ph_vertex_t pvert[] = {
        {'x', { 1, 0, 0 }, 0,0, 0,0, 0, { 0, 0, 0 },  g },
        {'x', {-1, 0, 0 }, 0,0, 0,0, 0, { 0, 0, 0 },  g },
        {'y', { 0, 1, 0 }, 0,0, 0,0, 1, { 0, 0, 0 },  g },
        {'y', { 0,-1, 0 }, 0,0, 0,0, 1, { 0, 0, 0 },  g },
    };
    // TODO do we need these?!
    // pvert.push_back( {'x', { 1, 0, 0 }, 0,0, 0,0, 0, { 1, 0, 0 }, -g } );
    // pvert.push_back( {'x', {-1, 0, 0 }, 0,0, 0,0, 0, {-1, 0, 0 }, -g } );
    // pvert.push_back( {'y', { 0, 1, 0 }, 0,0, 0,0, 1, { 0, 1, 0 }, -g } );
    // pvert.push_back( {'y', { 0,-1, 0 }, 0,0, 0,0, 1, { 0,-1, 0 }, -g } );
    diverge_model_internals_tu_phonons(m, max_dist, phop, sizeof(phop)/sizeof(phop[0]),
            pvert, sizeof(pvert)/sizeof(pvert[0]));
    return m;
}

CT_TEST_NAME(hubbard_SSH_model, "hubbard-ssh") {
    diverge_model_t* m = hubbard_ssh_model();
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
        mpi_eprintf( "\n%c[1A", 27 );
        fflush(stderr);
    } while (diverge_euler_next(&eu, vmax));
    mpi_eprintf( "\n" ); fflush(stderr);

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
