#ifndef DIVERGE_SKIP_TESTS

#include "../catch.hpp"

#include "../../src/diverge_model.h"
#include "../../src/diverge_flow_step.h"
#include "../../src/diverge_flow_step_internal.hpp"
#include "../model_examples/old.h"
#include "../../src/diverge_model_internals.h"
#include "../model_examples/model_constructor.hpp"
#include "../../src/npatch/vertex.h"
#include "../../src/npatch/symmetrize_vertex.h"
#include "../helper_functions.hpp"


TEST_CASE("npatch vertex symmetrization", "[npatch_symm]") {
    diverge_model_t* m = gen_kagome_model(12,12);
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
        max_diff = MAX(max_diff, std::abs(V[idx]-helper[idx]));
    }
    CHECK(max_diff < 1e-5);

    diverge_flow_step_free( s );
    diverge_model_free( m );
    free(helper);
}

#endif // DIVERGE_SKIP_TESTS
