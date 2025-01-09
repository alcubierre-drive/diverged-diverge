#ifndef DIVERGE_SKIP_TESTS

#include "../catch.hpp"
#include "../../src/diverge_model.h"
#include "../../src/diverge_flow_step.h"
#include "../../src/diverge_flow_step_internal.hpp"
#include "../../src/diverge_patching.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/diverge_model_output.h"

#include "../../src/grid/vertex_memory.hpp"

#include "../../src/tu/diverge_interface.h"
#include "../../src/tu/diverge_interface_symmstruct.h"
#include "../../src/tu/vertex.hpp"
#include "../../src/tu/projection.h"
#include "../../src/tu/flow_stepper.hpp"
#include "../../src/misc/generate_symmetries.h"
#include "../../src/diverge_symmetrize.h"

#include "../model_examples/model_constructor.hpp"
#include "../helper_functions.hpp"
#include "../../src/misc/shared_mem.h"

static double dot(double* A, double* B) {
    return A[0]*B[0]+A[1]*B[1]+A[2]*B[2];
}

double test_symm_4_pnt( diverge_model_t* model, Vertex* vertex ) {
    if (model->n_sym == 0 || model->orb_symmetries == NULL) return -1.0;

    diverge_generate_symm_maps(model);
    index_t nk = kdim( model->nk );
    index_t n_os = model->n_orb * model->n_spin;
    index_t n_spin = model->n_spin;
    index_t n_orb = model->n_orb;
    index_t size = POW4(n_os) * POW3(nk);

    index_t n_sym = model->n_sym;
    double norm = 1./(double) n_sym;

    index_t* symm_map_mom = model->internals->symm_map_mom_crs;
    double* symm_beyond_UC = model->internals->symm_beyond_UC;
    double* kmesh = model->internals->kmesh;
    index_t* symm_map_orb = model->internals->symm_map_orb;
    index_t* symm_orb_off = model->internals->symm_orb_off;
    index_t* symm_orb_len = model->internals->symm_orb_len;
    complex128_t* symm_pref = model->internals->symm_pref;
    complex128_t* aux = (complex128_t*) calloc( size, sizeof(complex128_t) );
    complex128_t* buf = (complex128_t*) calloc( size, sizeof(complex128_t) );
    vertex->project_fullV(model, aux);

   #pragma omp parallel for schedule(static) collapse(11) num_threads(diverge_omp_num_threads())
    for (index_t k1=0; k1<nk; ++k1)
    for (index_t k2=0; k2<nk; ++k2)
    for (index_t k3=0; k3<nk; ++k3)
    for (index_t oo1=0; oo1<n_orb; ++oo1)
    for (index_t ss1=0; ss1<n_spin; ++ss1)
    for (index_t oo2=0; oo2<n_orb; ++oo2)
    for (index_t ss2=0; ss2<n_spin; ++ss2)
    for (index_t oo3=0; oo3<n_orb; ++oo3)
    for (index_t ss3=0; ss3<n_spin; ++ss3)
    for (index_t oo4=0; oo4<n_orb; ++oo4)
    for (index_t ss4=0; ss4<n_spin; ++ss4) {
        index_t o1 = IDX2( ss1, oo1, n_orb ),
                o2 = IDX2( ss2, oo2, n_orb ),
                o3 = IDX2( ss3, oo3, n_orb ),
                o4 = IDX2( ss4, oo4, n_orb );
        index_t k4 = k1m2( k1p2( k1, k2, model->nk ), k3, model->nk );

    for (index_t s=0; s<n_sym; ++s) {
            index_t os1 = IDX2( o1, s, n_sym ),
                    os2 = IDX2( o2, s, n_sym ),
                    os3 = IDX2( o3, s, n_sym ),
                    os4 = IDX2( o4, s, n_sym );
            index_t ks1 = symm_map_mom[IDX2(k1,s,n_sym)],
                    ks2 = symm_map_mom[IDX2(k2,s,n_sym)],
                    ks3 = symm_map_mom[IDX2(k3,s,n_sym)],
                    ks4 = symm_map_mom[IDX2(k4,s,n_sym)];
            for (index_t i1=0; i1<symm_orb_len[os1]; ++i1)
            for (index_t i2=0; i2<symm_orb_len[os2]; ++i2)
            for (index_t i3=0; i3<symm_orb_len[os3]; ++i3)
            for (index_t i4=0; i4<symm_orb_len[os4]; ++i4) {
                index_t xo1 = symm_orb_off[os1] + i1,
                        xo2 = symm_orb_off[os2] + i2,
                        xo3 = symm_orb_off[os3] + i3,
                        xo4 = symm_orb_off[os4] + i4;
                index_t mo1 = symm_map_orb[xo1],
                        mo2 = symm_map_orb[xo2],
                        mo3 = symm_map_orb[xo3],
                        mo4 = symm_map_orb[xo4];
                double kdotR11 = dot(symm_beyond_UC+3*IDX2(oo1,s,n_sym), kmesh+3*ks1);
                double kdotR22 = dot(symm_beyond_UC+3*IDX2(oo2,s,n_sym), kmesh+3*ks2);
                double kdotR33 = dot(symm_beyond_UC+3*IDX2(oo3,s,n_sym), kmesh+3*ks3);
                double kdotR44 = dot(symm_beyond_UC+3*IDX2(oo4,s,n_sym), kmesh+3*ks4);
                complex128_t val = aux[IDX7(ks1,ks2,ks3, mo1,mo2,mo3,mo4, nk,nk, n_os,n_os,n_os,n_os)] *
                            (symm_pref[xo1] * (cos(kdotR11) - I128 * sin(kdotR11))) *
                            (symm_pref[xo2] * (cos(kdotR22) - I128 * sin(kdotR22))) *
                   std::conj(symm_pref[xo3] * (cos(kdotR33) - I128 * sin(kdotR33))) *
                   std::conj(symm_pref[xo4] * (cos(kdotR44) - I128 * sin(kdotR44)));
                buf[IDX7(k1,k2,k3, o1,o2,o3,o4, nk,nk, n_os,n_os,n_os,n_os)] +=
                     val * norm;
            }
        }
    }
    double max_diff = 0.;
    #pragma omp parallel for reduction(max:max_diff) num_threads(diverge_omp_num_threads())
    for (index_t idx=0; idx<size; ++idx)  {
        max_diff = MAX(max_diff, std::abs(buf[idx]-aux[idx]));
    }
    mpi_log_printf("maximal error %2.5f \n", max_diff);
    free(aux);
    free(buf);
    return max_diff;
}

static double shmem_diverge_symmetrize_2pt_fine( diverge_model_t* mod, complex128_t* ptr, complex128_t* aux ) {
    double priv_error = 0.0;
    if (shared_exclusive_enter( ptr ) == 1)
        priv_error += diverge_symmetrize_2pt_fine( mod, ptr, aux );
    shared_exclusive_wait( ptr );
    double all_error = 0.0;
    diverge_mpi_allreduce_double_sum( &priv_error, &all_error, 1 );
    return all_error;
}

TEST_CASE("check symm honey","[symm_2pnt][symm_honey]") {

    diverge_model_t* mod1 = gen_honeycomb_lat_hub_sym(6,6);
    diverge_model_t* mod2 = gen_honeycomb_lat_hub(6,6);

    diverge_model_internals_tu( mod1, 1.0 );
    diverge_model_internals_tu( mod2, 1.0 );
    double err = shmem_diverge_symmetrize_2pt_fine( mod1, mod1->internals->ham, NULL );
    CHECK(err < 1e-12);
    err = shmem_diverge_symmetrize_2pt_fine( mod1, mod2->internals->ham, NULL );
    CHECK(err < 1e-12);
    diverge_flow_step_t* step1 = diverge_flow_step_init(mod1, "tu", "PCD");
    diverge_flow_step_t* step2 = diverge_flow_step_init(mod2, "tu", "PCD");

    complex128_t* reco_vert1 = (complex128_t*)malloc(step1->tu_vertex->nk *
                POW2(step1->tu_vertex->n_orbff*POW2(step1->tu_vertex->n_spin)) * sizeof(complex128_t));
    complex128_t* reco_vert2 = (complex128_t*)malloc(step2->tu_vertex->nk *
                POW2(step2->tu_vertex->n_orbff*POW2(step2->tu_vertex->n_spin)) * sizeof(complex128_t));

    double Lambda = 1.0;
    double dLambda = -0.06;

    for (int i=0; i<4; ++i) {
        diverge_flow_step_euler( step1, Lambda, dLambda );
        diverge_flow_step_euler( step2, Lambda, dLambda );
        Lambda += dLambda;
    }
    double maxdiff = 0.0;
    CHECK( test_symm_4_pnt( mod1, step2->tu_vertex ) < 1e-10 );

    step1->tu_vertex->reco('P', reco_vert1);
    step2->tu_vertex->reco('P', reco_vert2);
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }

    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('C', reco_vert1);
    step2->tu_vertex->reco('C', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('D', reco_vert1);
    step2->tu_vertex->reco('D', reco_vert2);
    maxdiff = 0.0;

    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);

    diverge_flow_step_free(step1);
    diverge_flow_step_free(step2);
    free(reco_vert1);
    free(reco_vert2);
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
}

TEST_CASE("check symm rashba","[symm_2pnt][symm_rashba]") {

    diverge_model_t* mod1 = gen_square_lat_rashba_sym(6,6);
    diverge_model_t* mod2 = gen_square_lat_rashba(6,6);
    double err = shmem_diverge_symmetrize_2pt_fine( mod1, mod1->internals->ham, NULL );
    CHECK(err < 1e-12);
    err = shmem_diverge_symmetrize_2pt_fine( mod1, mod2->internals->ham, NULL );
    CHECK(err < 1e-12);
    diverge_model_internals_tu( mod1, 1.0 );
    diverge_model_internals_tu( mod2, 1.0 );
    diverge_flow_step_t* step1 = diverge_flow_step_init(mod1, "tu", "PCD");
    diverge_flow_step_t* step2 = diverge_flow_step_init(mod2, "tu", "PCD");

    double Lambda = 1.0;
    double dLambda = -0.05;
    complex128_t* reco_vert1 = (complex128_t*)malloc(step1->tu_vertex->nk *
                POW2(step1->tu_vertex->n_orbff*POW2(step1->tu_vertex->n_spin)) * sizeof(complex128_t));
    complex128_t* reco_vert2 = (complex128_t*)malloc(step2->tu_vertex->nk *
                POW2(step2->tu_vertex->n_orbff*POW2(step2->tu_vertex->n_spin)) * sizeof(complex128_t));

    for (int i=0; i<4; ++i) {
        diverge_flow_step_euler( step1, Lambda, dLambda );
        diverge_flow_step_euler( step2, Lambda, dLambda );
        Lambda += dLambda;
    }
    // check P,C,D bubble_helper
    double maxdiff = 0.0;
    for(uint i = 0; i < 4*((tu_data_t*)(mod1->internals->tu_data))->nk; ++i) {
        double d = std::abs(mod1->internals->ham[i]-mod2->internals->ham[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    maxdiff = 0.0;
    step1->tu_vertex->reco('P', reco_vert1);
    step2->tu_vertex->reco('P', reco_vert2);
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('C', reco_vert1);
    step2->tu_vertex->reco('C', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('D', reco_vert1);
    step2->tu_vertex->reco('D', reco_vert2);
    maxdiff = 0.0;

    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }

    CHECK(maxdiff < 1e-8);
    diverge_flow_step_free(step1);
    diverge_flow_step_free(step2);
    free(reco_vert1);
    free(reco_vert2);
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
}

TEST_CASE("check symm hub","[symm_2pnt][symm_hub]") {

    diverge_model_t* mod1 = gen_square_lat_hub_sym(6,6);
    diverge_model_t* mod2 = gen_square_lat_hub(6,6);
    double err = shmem_diverge_symmetrize_2pt_fine( mod1, mod1->internals->ham, NULL );
    CHECK(err < 1e-12);
    err = shmem_diverge_symmetrize_2pt_fine( mod2, mod2->internals->ham, NULL );
    CHECK(err < 1e-12);
    diverge_model_internals_tu( mod1, 1.1 );
    diverge_model_internals_tu( mod2, 1.1 );
    diverge_flow_step_t* step1 = diverge_flow_step_init(mod1, "tu", "PCD");
    diverge_flow_step_t* step2 = diverge_flow_step_init(mod2, "tu", "PCD");
    complex128_t* reco_vert1 = (complex128_t*)malloc(step1->tu_vertex->nk *
                POW2(step1->tu_vertex->n_orbff*POW2(step1->tu_vertex->n_spin)) * sizeof(complex128_t));
    complex128_t* reco_vert2 = (complex128_t*)malloc(step2->tu_vertex->nk *
                POW2(step2->tu_vertex->n_orbff*POW2(step2->tu_vertex->n_spin)) * sizeof(complex128_t));

    double Lambda = 1.0;
    double dLambda = -0.1;
    for (int i=0; i<4; ++i) {
        diverge_flow_step_euler( step1, Lambda, dLambda );
        diverge_flow_step_euler( step2, Lambda, dLambda );
        Lambda += dLambda;
    }
    // check P,C,D bubble_helper
    step1->tu_vertex->reco('P', reco_vert1);
    step2->tu_vertex->reco('P', reco_vert2);
    double maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('C', reco_vert1);
    step2->tu_vertex->reco('C', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('D', reco_vert1);
    step2->tu_vertex->reco('D', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    diverge_flow_step_free(step1);
    diverge_flow_step_free(step2);
    free(reco_vert1);
    free(reco_vert2);
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
}


TEST_CASE("check symm hub ref","[symm_2pnt][symm_hub][symm_ref]") {
    diverge_model_t* mod1 = gen_square_lat_hub_sym(6,6,3,3);
    diverge_model_t* mod2 = gen_square_lat_hub_sym(6,6,3,3);
    mod2->n_sym = 0;
    double err = shmem_diverge_symmetrize_2pt_fine( mod1, mod1->internals->ham, NULL );
    CHECK(err < 1e-12);
    diverge_model_internals_tu( mod1, 2.0 );
    diverge_model_internals_tu( mod2, 2.0 );
    diverge_flow_step_t* step1 = diverge_flow_step_init(mod1, "tu", "PCD");
    diverge_flow_step_t* step2 = diverge_flow_step_init(mod2, "tu", "PCD");
    complex128_t* reco_vert1 = (complex128_t*)malloc(step1->tu_vertex->nk *
                POW2(step1->tu_vertex->n_orbff*POW2(step1->tu_vertex->n_spin)) * sizeof(complex128_t));
    complex128_t* reco_vert2 = (complex128_t*)malloc(step2->tu_vertex->nk *
                POW2(step2->tu_vertex->n_orbff*POW2(step2->tu_vertex->n_spin)) * sizeof(complex128_t));


    double Lambda = 1.0;
    double dLambda = -0.1;
    for (int i=0; i<8; ++i) {
        diverge_flow_step_euler( step1, Lambda, dLambda );
        diverge_flow_step_euler( step2, Lambda, dLambda );
        Lambda += dLambda;
    }
    test_symm_4_pnt( mod1, step1->tu_vertex );
    test_symm_4_pnt( mod1, step2->tu_vertex );
    // check P,C,D bubble_helper
    step1->tu_vertex->reco('P', reco_vert1);
    step2->tu_vertex->reco('P', reco_vert2);
    double maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('C', reco_vert1);
    step2->tu_vertex->reco('C', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('D', reco_vert1);
    step2->tu_vertex->reco('D', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    diverge_flow_step_free(step1);
    diverge_flow_step_free(step2);
    free(reco_vert1);
    free(reco_vert2);
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
}


TEST_CASE("check symm honey ref","[symm_2pnt][symm_honey][symm_ref]") {
    diverge_model_t* mod1 = gen_honeycomb_lat_hub_sym(6,6,3,3);
    diverge_model_t* mod2 = gen_honeycomb_lat_hub_sym(6,6,3,3);
    mod2->n_sym = 0;
    double err = shmem_diverge_symmetrize_2pt_fine( mod1, mod1->internals->ham, NULL );
    CHECK(err < 1e-12);
    diverge_model_internals_tu( mod1, 1.3 );
    diverge_model_internals_tu( mod2, 1.3 );
    diverge_flow_step_t* step1 = diverge_flow_step_init(mod1, "tu", "PCD");
    diverge_flow_step_t* step2 = diverge_flow_step_init(mod2, "tu", "PCD");
    complex128_t* reco_vert1 = (complex128_t*)malloc(step1->tu_vertex->nk *
                POW2(step1->tu_vertex->n_orbff*POW2(step1->tu_vertex->n_spin)) * sizeof(complex128_t));
    complex128_t* reco_vert2 = (complex128_t*)malloc(step2->tu_vertex->nk *
                POW2(step2->tu_vertex->n_orbff*POW2(step2->tu_vertex->n_spin)) * sizeof(complex128_t));


    double Lambda = 1.0;
    double dLambda = -0.1;
    for (int i=0; i<8; ++i) {
        diverge_flow_step_euler( step1, Lambda, dLambda );
        diverge_flow_step_euler( step2, Lambda, dLambda );
        Lambda += dLambda;
    }
    test_symm_4_pnt( mod1, step1->tu_vertex );
    test_symm_4_pnt( mod1, step2->tu_vertex );
    // check P,C,D bubble_helper
    step1->tu_vertex->reco('P', reco_vert1);
    step2->tu_vertex->reco('P', reco_vert2);
    double maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('C', reco_vert1);
    step2->tu_vertex->reco('C', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('D', reco_vert1);
    step2->tu_vertex->reco('D', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    diverge_flow_step_free(step1);
    diverge_flow_step_free(step2);
    free(reco_vert1);
    free(reco_vert2);
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
}


TEST_CASE("check symm rashba ref","[symm_2pnt][symm_rashba][symm_ref]") {
    diverge_model_t* mod1 = gen_square_lat_rashba_sym(6,6,3,3);
    diverge_model_t* mod2 = gen_square_lat_rashba_sym(6,6,3,3);
    mod2->n_sym = 0;
    double err = shmem_diverge_symmetrize_2pt_fine( mod1, mod1->internals->ham, NULL );
    CHECK(err < 1e-12);
    diverge_model_internals_tu( mod1, 1.3 );
    diverge_model_internals_tu( mod2, 1.3 );
    diverge_flow_step_t* step1 = diverge_flow_step_init(mod1, "tu", "PCD");
    diverge_flow_step_t* step2 = diverge_flow_step_init(mod2, "tu", "PCD");
    complex128_t* reco_vert1 = (complex128_t*)malloc(step1->tu_vertex->nk *
                POW2(step1->tu_vertex->n_orbff*POW2(step1->tu_vertex->n_spin)) * sizeof(complex128_t));
    complex128_t* reco_vert2 = (complex128_t*)malloc(step2->tu_vertex->nk *
                POW2(step2->tu_vertex->n_orbff*POW2(step2->tu_vertex->n_spin)) * sizeof(complex128_t));


    double Lambda = 1.0;
    double dLambda = -0.1;
    for (int i=0; i<8; ++i) {
        diverge_flow_step_euler( step1, Lambda, dLambda );
        diverge_flow_step_euler( step2, Lambda, dLambda );
        Lambda += dLambda;
    }
    test_symm_4_pnt( mod1, step1->tu_vertex );
    test_symm_4_pnt( mod1, step2->tu_vertex );
    // check P,C,D bubble_helper
    step1->tu_vertex->reco('P', reco_vert1);
    step2->tu_vertex->reco('P', reco_vert2);
    double maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('C', reco_vert1);
    step2->tu_vertex->reco('C', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('D', reco_vert1);
    step2->tu_vertex->reco('D', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    diverge_flow_step_free(step1);
    diverge_flow_step_free(step2);
    free(reco_vert1);
    free(reco_vert2);
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
}

TEST_CASE("check symm triangular pxpy ref","[symm_2pnt][symm_tri][symm_ref]") {
    diverge_model_t* mod1 = gen_triangular_pxpy_sym(4,false);
    diverge_model_t* mod2 = gen_triangular_pxpy_sym(4,false);
    mod2->n_sym = 0;
    double err = shmem_diverge_symmetrize_2pt_fine( mod1, mod1->internals->ham, NULL );
    CHECK(err < 1e-12);
    diverge_model_internals_tu( mod1, 1.0 );
    diverge_model_internals_tu( mod2, 1.0 );
    diverge_flow_step_t* step1 = diverge_flow_step_init(mod1, "tu", "PCD");
    diverge_flow_step_t* step2 = diverge_flow_step_init(mod2, "tu", "PCD");
    complex128_t* reco_vert1 = (complex128_t*)malloc(step1->tu_vertex->nk *
                POW2(step1->tu_vertex->n_orbff*POW2(step1->tu_vertex->n_spin)) * sizeof(complex128_t));
    complex128_t* reco_vert2 = (complex128_t*)malloc(step2->tu_vertex->nk *
                POW2(step2->tu_vertex->n_orbff*POW2(step2->tu_vertex->n_spin)) * sizeof(complex128_t));

    double Lambda = 1.0;
    double dLambda = -0.1;
    for (int i=0; i<4; ++i) {
        diverge_flow_step_euler( step1, Lambda, dLambda );
        diverge_flow_step_euler( step2, Lambda, dLambda );
        Lambda += dLambda;
    }
    CHECK(test_symm_4_pnt( mod1, step1->tu_vertex ) < 1e-10);
    CHECK(test_symm_4_pnt( mod1, step2->tu_vertex ) < 1e-10);
    // check P,C,D bubble_helper
    step1->tu_vertex->reco('P', reco_vert1);
    step2->tu_vertex->reco('P', reco_vert2);
    double maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('C', reco_vert1);
    step2->tu_vertex->reco('C', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('D', reco_vert1);
    step2->tu_vertex->reco('D', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    diverge_flow_step_free(step1);
    diverge_flow_step_free(step2);
    free(reco_vert1);
    free(reco_vert2);
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
}

TEST_CASE("check symm triangular rashba ref","[symm_2pnt][symm_trirash]") {
    diverge_model_t* mod1 = gen_triangular_lat_rashba(4);
    diverge_model_t* mod2 = gen_triangular_lat_rashba(4);
    mod2->n_sym = 0;
    double err = shmem_diverge_symmetrize_2pt_fine( mod1, mod1->internals->ham, NULL );
    CHECK(err < 1e-12);
    diverge_model_internals_tu( mod1, 1.1 );
    diverge_model_internals_tu( mod2, 1.1 );
    diverge_flow_step_t* step1 = diverge_flow_step_init(mod1, "tu", "PCD");
    diverge_flow_step_t* step2 = diverge_flow_step_init(mod2, "tu", "PCD");
    complex128_t* reco_vert1 = (complex128_t*)malloc(step1->tu_vertex->nk *
                POW2(step1->tu_vertex->n_orbff*POW2(step1->tu_vertex->n_spin)) * sizeof(complex128_t));
    complex128_t* reco_vert2 = (complex128_t*)malloc(step2->tu_vertex->nk *
                POW2(step2->tu_vertex->n_orbff*POW2(step2->tu_vertex->n_spin)) * sizeof(complex128_t));

    double Lambda = 1.0;
    double dLambda = -0.1;
    for (int i=0; i<5; ++i) {
        diverge_flow_step_euler( step1, Lambda, dLambda );
        diverge_flow_step_euler( step2, Lambda, dLambda );
        Lambda += dLambda;
    }
    CHECK(test_symm_4_pnt( mod1, step1->tu_vertex ) < 1e-10);
    CHECK(test_symm_4_pnt( mod1, step2->tu_vertex ) < 1e-10);
    // check P,C,D bubble_helper
    step1->tu_vertex->reco('P', reco_vert1);
    step2->tu_vertex->reco('P', reco_vert2);
    double maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('C', reco_vert1);
    step2->tu_vertex->reco('C', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('D', reco_vert1);
    step2->tu_vertex->reco('D', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    diverge_flow_step_free(step1);
    diverge_flow_step_free(step2);
    free(reco_vert1);
    free(reco_vert2);
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
}

TEST_CASE("check symm triangular p+p- ref","[symm_2pnt][symm_tri][symm_ref]") {
    diverge_model_t* mod1 = gen_triangular_pxpy_sym(6,true);
    diverge_model_t* mod2 = gen_triangular_pxpy_sym(6,true);
    mod2->n_sym = 0;
    double err = shmem_diverge_symmetrize_2pt_fine( mod1, mod1->internals->ham, NULL );
    CHECK(err < 1e-12);

    bool test = check_hermicity(mod1->internals->ham, 2, 36);
    CHECK(test);

    diverge_model_internals_tu( mod1, 0.0 );
    diverge_model_internals_tu( mod2, 0.0 );
    diverge_flow_step_t* step1 = diverge_flow_step_init(mod1, "tu", "PCD");
    diverge_flow_step_t* step2 = diverge_flow_step_init(mod2, "tu", "PCD");
    complex128_t* reco_vert1 = (complex128_t*)malloc(step1->tu_vertex->nk *
                POW2(step1->tu_vertex->n_orbff*POW2(step1->tu_vertex->n_spin)) * sizeof(complex128_t));
    complex128_t* reco_vert2 = (complex128_t*)malloc(step2->tu_vertex->nk *
                POW2(step2->tu_vertex->n_orbff*POW2(step2->tu_vertex->n_spin)) * sizeof(complex128_t));

    double Lambda = 1.0;
    double dLambda = -0.1;
    for (int i=0; i<4; ++i) {
        diverge_flow_step_euler( step1, Lambda, dLambda );
        diverge_flow_step_euler( step2, Lambda, dLambda );
        Lambda += dLambda;
    }
    test_symm_4_pnt( mod1, step1->tu_vertex );
    test_symm_4_pnt( mod1, step2->tu_vertex );
    // check P,C,D bubble_helper
    step1->tu_vertex->reco('P', reco_vert1);
    step2->tu_vertex->reco('P', reco_vert2);
    double maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('C', reco_vert1);
    step2->tu_vertex->reco('C', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    step1->tu_vertex->reco('D', reco_vert1);
    step2->tu_vertex->reco('D', reco_vert2);
    maxdiff = 0.0;
    for(uint i = 0; i < step2->tu_vertex->full_vert_size; ++i) {
        double d = std::abs(reco_vert2[i]-reco_vert1[i]);
        maxdiff = d>maxdiff ? d : maxdiff;
    }
    CHECK(maxdiff < 1e-8);
    diverge_flow_step_free(step1);
    diverge_flow_step_free(step2);
    free(reco_vert1);
    free(reco_vert2);
    diverge_model_free( mod1 );
    diverge_model_free( mod2 );
}


#endif // DIVERGE_SKIP_TESTS
