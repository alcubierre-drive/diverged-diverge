#ifndef DIVERGE_SKIP_TESTS

#define CT_SUITE_NAME npatch_sym
#include "../ctest.h"

#include "../../src/diverge_model.h"
#include "../../src/diverge_flow_step.h"
#include "../../src/diverge_flow_step_internal.h"
#include "../../src/diverge_model_internals.h"
#include "../models/all.h"
#include "../../src/npatch/vertex.h"
#include "../../src/npatch/symmetrize_vertex.h"
#include "../helper_functions.h"


CT_TEST(vertex) {
    diverge_model_t* m = KWARGS(gen_kagome_model,.nkx=12,.nky=12);
    diverge_model_internals_patch( m, 6 );
    diverge_flow_step_t* s = diverge_flow_step_init( m, "patch", "PCD" );
    index_t size = ((npatch_vertex_t*)(m->internals->patch_vertex))->size;

    complex128_t* helper = (complex128_t*)calloc(size,sizeof(complex128_t));
    complex128_t* V = ((npatch_vertex_t*)(m->internals->patch_vertex))->V;
    memcpy( helper, V, sizeof(complex128_t)*size );

    symmetrize_npatch_vertex( m, V, s->patch_loop->dpat->patches, s->patch_loop->np, NULL );
    double max_diff = 0.;
    #pragma omp parallel for reduction(max:max_diff) num_threads(diverge_omp_num_threads())
    for (index_t idx=0; idx<size; ++idx) {
        max_diff = MAX(max_diff, cabs(V[idx]-helper[idx]));
    }
    CT_CHECK(max_diff < 1e-5);

    diverge_flow_step_free( s );
    diverge_model_free( m );
    free(helper);
}

#endif // DIVERGE_SKIP_TESTS
