#ifndef DIVERGE_SKIP_TESTS

#include "../catch.hpp"
#include "../../src/diverge_model.h"
#include "../../src/diverge_flow_step.h"
#include "../../src/diverge_flow_step_internal.h"
#include "../../src/diverge_patching.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/diverge_model_output.h"

#include "../../src/tu/diverge_interface.h"
#include "../../src/tu/diverge_interface_symmstruct.h"
#include "../../src/tu/vertex_details.h"
#include "../../src/tu/projection.h"
#include "../../src/misc/generate_symmetries.h"
#include "../../src/diverge_symmetrize.h"

#include "../model_examples/model_constructor.hpp"
#include "../helper_functions.hpp"


void full_vertex(const diverge_model_t* mod, index_t k1, index_t k2, index_t k3, complex128_t* buf) {
    complex128_t* hbuf = (complex128_t*)mod->data;

    index_t nk = kdim( mod->nk );
    index_t n_os = mod->n_orb * mod->n_spin;
    index_t size_batch = POW4(n_os);
    index_t koff = k3 + nk *(k2 + nk * k1);
    memcpy((void*)buf,(void*)(hbuf + size_batch * koff), sizeof(complex128_t)*size_batch);
}

TEST_CASE("check init_from_full","[tu_from_full]") {

    diverge_model_t* mod1 = gen_honeycomb_lat_hub_sym(6,6);

    diverge_model_internals_tu( mod1, 1.2 );
    diverge_flow_step_t* step1 = diverge_flow_step_init(mod1, "tu", "PCD");
    double dLambda = -0.1;
    double Lambda = 1;

    for (int i=0; i<4; ++i) {
        diverge_flow_step_euler( step1, Lambda, dLambda );
        Lambda += dLambda;
    }
    for(index_t i = 0; i < step1->tu_vertex->full_vert_size;++i) {
        step1->tu_vertex->Pch[i] = 0.0;
        step1->tu_vertex->Cch[i] = 0.0;
    }
    index_t nk = kdim( mod1->nk );
    index_t n_os = mod1->n_orb * mod1->n_spin;
    index_t size = POW4(n_os) * POW3(nk);

    complex128_t* hbuf = (complex128_t*) calloc( size, sizeof(complex128_t) );
    for (index_t k123=0; k123<POW3(nk); ++k123)
        memcpy( hbuf + k123*POW4(mod1->n_orb*mod1->n_spin), mod1->internals->ham, POW4(mod1->n_orb*mod1->n_spin)*sizeof(complex128_t) );
    mod1->data = (char*)hbuf;

    complex128_t* buf = (complex128_t*) calloc( size, sizeof(complex128_t) );
    tu_vertex_project_fullV( step1->tu_vertex, mod1, hbuf );

    mod1->ffill = &full_vertex;
    tu_projection_t* proj = tu_projection_init(mod1);
    tu_prop_t* loop = tu_prop_init(mod1);
    tu_vertex_t* v2 = tu_vertex_init(mod1, proj, loop, 1,1,1, 0);
    tu_vertex_project_fullV( v2, mod1, buf );
    CHECK(check_equal(buf,hbuf,size));

    tu_vertex_free( v2 );
    tu_prop_free( loop );
    free(buf);
    free(hbuf);
    mod1->data = NULL;
    diverge_flow_step_free(step1);
    tu_projection_free( proj );
    diverge_model_free( mod1 );
}

#endif // DIVERGE_SKIP_TESTS
