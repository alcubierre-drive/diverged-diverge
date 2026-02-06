#define CT_SUITE_NAME tu_reco
#include "ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../src/diverge_model.h"
#include "../src/diverge_model_internals.h"
#include "../src/diverge_flow_step.h"
#include "../src/diverge_flow_step_internal.h"
#include "models/all.h"
#include "../src/tu/vertex.h"
#include "../src/tu/vertex_details.h"
#include "helper_functions.h"

CT_TEST(tu_reco_sym) {
    double ffdist = 1.51;
    int nk = 6;

    diverge_model_t* m_sym = KWARGS(gen_honeycomb_lat_hub_sym, .nkx=nk, .nky=nk);
    diverge_model_internals_tu(m_sym, ffdist);
    diverge_flow_step_t* s_sym = diverge_flow_step_init_any(m_sym, "PCD");

    diverge_model_t* m = KWARGS(gen_honeycomb_lat_hub, .nkx=nk, .nky=nk);
    diverge_model_internals_tu(m, ffdist);
    diverge_flow_step_t* s = diverge_flow_step_init_any(m, "PCD");

    double Lambda = 1.0,
           dLambda = -0.1;
    for (int i=0; i<9; ++i) {
        diverge_flow_step_euler(s_sym, Lambda, dLambda);
        diverge_flow_step_euler(s, Lambda, dLambda);
        Lambda += dLambda;
        mpi_log_printf( "reco flow (V=%.2f;%.2f)\n",
                diverge_flow_step_vertmax_r(s_sym,NULL),
                diverge_flow_step_vertmax_r(s,NULL) );
    }

    tu_vertex_t* tu_vertex = s->tu_vertex;
    tu_vertex_t* tu_vertex_sym = s_sym->tu_vertex;

    index_t bufsz = kdim(m->nk)*POW2(tu_vertex->n_orbff*POW2(m->n_spin));
    complex128_t* buf = calloc(bufsz, sizeof*buf);
    complex128_t* buf_sym = calloc(bufsz, sizeof*buf_sym);

    char chans[] = "PCD";
    for (char* chan=chans; *chan; ++chan) {
        diverge_tu_vertex_reco( tu_vertex, *chan, buf, NULL );
        diverge_tu_vertex_reco( tu_vertex_sym, *chan, buf_sym, NULL );
        CT_CHECK(check_equal_cd(buf, buf_sym, bufsz, 1.e-12));
    }

    free(buf_sym);
    diverge_flow_step_free(s_sym);
    diverge_model_free(m_sym);

    free(buf);
    diverge_flow_step_free(s);
    diverge_model_free(m);
}

CT_TEST(tu_reco_q_sym) {
    double ffdist = 1.51;
    int nk = 6;

    diverge_model_t* m_sym = KWARGS(gen_honeycomb_lat_hub_sym, .nkx=nk, .nky=nk);
    diverge_model_internals_tu(m_sym, ffdist);
    diverge_flow_step_t* s_sym = diverge_flow_step_init_any(m_sym, "PCD");

    diverge_model_t* m = KWARGS(gen_honeycomb_lat_hub, .nkx=nk, .nky=nk);
    diverge_model_internals_tu(m, ffdist);
    diverge_flow_step_t* s = diverge_flow_step_init_any(m, "PCD");

    double Lambda = 1.0,
           dLambda = -0.1;
    for (int i=0; i<9; ++i) {
        diverge_flow_step_euler(s_sym, Lambda, dLambda);
        diverge_flow_step_euler(s, Lambda, dLambda);
        Lambda += dLambda;
        mpi_log_printf( "reco flow (V=%.2f;%.2f)\n",
                diverge_flow_step_vertmax_r(s_sym,NULL),
                diverge_flow_step_vertmax_r(s,NULL) );
    }

    tu_vertex_t* tu_vertex = s->tu_vertex;
    tu_vertex_t* tu_vertex_sym = s_sym->tu_vertex;

    index_t bufsz = POW2(tu_vertex->n_orbff*POW2(m->n_spin));
    complex128_t* buf = calloc(bufsz, sizeof*buf);
    complex128_t* buf_sym = calloc(bufsz, sizeof*buf_sym);

    char chans[] = "PCD";
    for (char* chan=chans; *chan; ++chan)
    for (index_t q=0; q<kdim(m->nk); ++q) {
        diverge_tu_vertex_reco_at_q( tu_vertex, *chan, buf, q, NULL );
        diverge_tu_vertex_reco_at_q( tu_vertex_sym, *chan, buf_sym, q, NULL );
        CT_CHECK(check_equal_cd(buf, buf_sym, bufsz, 1.e-12));
    }

    free(buf_sym);
    diverge_flow_step_free(s_sym);
    diverge_model_free(m_sym);

    free(buf);
    diverge_flow_step_free(s);
    diverge_model_free(m);
}

#endif // DIVERGE_SKIP_TESTS
