#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
#include "../catch.hpp"
#include "../model_examples/model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.hpp"
#include "../../src/misc/read_W90.h"
#include "../../src/misc/read_fplo.h"

#include "graphene_dat.h"
#include <cstdio>

#include <string>

#ifdef USE_FPLO_READER

TEST_CASE("FPLO read in","[FPLOreader]") {
    std::string fname_W90 = gen_graphene_hr_dat(0);
    std::string fname_FPLO = gen_graphene_hr_dat(1);
    FILE* fptr = fopen(fname_FPLO.c_str(), "r");
    char fcontent[1024];
    int nread = fread(fcontent, 1, 1024, fptr);
    fclose(fptr);
    memset( fcontent+nread, 0, 1024-nread );
    mpi_fil_printf( "generated file '%s' (+hamdata). Content:\n%s",
            fname_FPLO.c_str(), fcontent );

    diverge_model_t* fplo_model = diverge_read_fplo( fname_FPLO.c_str() );

    index_t len = -1;
    rs_hopping_t* w90_hop = diverge_read_W90_C( fname_W90.c_str(), 1, &len, NULL );
    remove( fname_FPLO.c_str() );
    remove( fname_W90.c_str() );

    vector<bool> masked_hoppings( len, false );
    for (rs_hopping_t* h=fplo_model->hop; h!=fplo_model->hop+fplo_model->n_hop; ++h) {
        for (index_t i=0; i<len; ++i) {
            rs_hopping_t* t = w90_hop+i;
            if (h->R[0] == t->R[0] && h->R[1] == t->R[1] && h->R[2] == t->R[2] &&
                h->o1 == t->o1 && h->o2 == t->o2 && h->s1 == t->s1 && h->s2 == t->s2 &&
                std::abs(t->t - h->t) < 1.e-5) {
                masked_hoppings[i] = true;
                break;
            }
        }
    }
    for (index_t i=0; i<len; ++i) {
        if (std::abs(w90_hop[i].t) < 1.e-5)
            masked_hoppings[i] = true;
        CHECK( masked_hoppings[i] );
    }

    diverge_model_free( fplo_model );
    free( w90_hop );
}

#endif

TEST_CASE("W90 read in","[W90reader]") {
    diverge_model_t* model = gen_honeycomb_lat_hub(2,2,1,1,1,0,0,0);
    index_t len = 0;
    index_t n_spin = 0;

    std::string fname = gen_graphene_hr_dat(0);

    char fcontent[1024];
    FILE* fptr = fopen(fname.c_str(), "r");
    int nread = fread(fcontent, 1, 1024, fptr);
    fclose(fptr);
    memset( fcontent+nread, 0, 1024-nread );
    mpi_fil_printf( "generated file '%s' as W90 file for testing. Content:\n%s",
            fname.c_str(), fcontent );
    rs_hopping_t* cmp = model->hop;
    rs_hopping_t* hop = diverge_read_W90_C(fname.c_str(),n_spin,&len,NULL);
    remove( fname.c_str() );
    // check all hoppings exist and are equal
    CHECK(len > 0);
    for(index_t elem = 0; elem < len; ++elem) {
        bool found = false;
        if(std::abs(hop[elem].t) > 1e-10) {
            for(index_t trial = 0; trial < model->n_hop; ++trial) {
                if(hop[elem].o1 == cmp[trial].o1 &&
                    hop[elem].o2 == cmp[trial].o2 &&
                    hop[elem].s1 == cmp[trial].s1 &&
                    hop[elem].s2 == cmp[trial].s2 &&
                    hop[elem].R[0] == cmp[trial].R[0] &&
                    hop[elem].R[1] == cmp[trial].R[1] &&
                    hop[elem].R[2] == cmp[trial].R[2] &&
                    std::abs(hop[elem].t - cmp[trial].t) < 1e-5)
                        found = true;
            }
        } else {
            found = true;
        }
        CHECK( found );
    }
    free( hop );
    diverge_model_free( model );
}

TEST_CASE("W90 read in, reversed orbitals", "[W90reader][reversed]") {
    diverge_model_t* model = gen_honeycomb_lat_hub(2,2,1,1,1,0,0,0);
    index_t len = 0;
    index_t n_spin = 0;

    std::string fname = gen_graphene_hr_dat(0);

    char fcontent[1024];
    FILE* fptr = fopen(fname.c_str(), "r");
    int nread = fread(fcontent, 1, 1024, fptr);
    fclose(fptr);
    memset( fcontent+nread, 0, 1024-nread );
    mpi_fil_printf( "generated file '%s' as W90 file for testing. Content:\n%s",
            fname.c_str(), fcontent );
    rs_hopping_t* cmp = model->hop;
    index_t n_orb = 0x0; n_orb = ~n_orb;
    rs_hopping_t* hop = diverge_read_W90_C(fname.c_str(),n_spin,&len,&n_orb);
    remove( fname.c_str() );
    // check all hoppings exist and are equal
    CHECK(len > 0);
    for(index_t elem = 0; elem < len; ++elem) {
        bool found = false;
        if(std::abs(hop[elem].t) > 1e-10) {
            for(index_t trial = 0; trial < model->n_hop; ++trial) {
                // here we have the swapped hopping elements!!
                if(hop[elem].o2 == cmp[trial].o1 &&
                    hop[elem].o1 == cmp[trial].o2 &&
                    hop[elem].s2 == cmp[trial].s1 &&
                    hop[elem].s1 == cmp[trial].s2 &&
                    hop[elem].R[0] == cmp[trial].R[0] &&
                    hop[elem].R[1] == cmp[trial].R[1] &&
                    hop[elem].R[2] == cmp[trial].R[2] &&
                    std::abs(hop[elem].t - cmp[trial].t) < 1e-5)
                        found = true;
            }
        } else {
            found = true;
        }
        CHECK( found );
    }
    free( hop );
    diverge_model_free( model );
}

#endif // DIVERGE_SKIP_TESTS
