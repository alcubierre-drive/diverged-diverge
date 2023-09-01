#include "../../src/diverge_model.h"
#include "../../src/diverge_momentum_gen.h"
#include "../catch.hpp"
#include "../model_examples/model_constructor.hpp"
#include "../../src/diverge_model_internals.h"
#include "../helper_functions.hpp"
#include "../../src/misc/read_W90.h"

#include "graphene_dat.h"
#include <cstdio>

#include <string>

TEST_CASE("W90 read in","[W90reader]") {
    diverge_model_t* model = gen_honeycomb_lat_hub(2,2,1,1,1,0,0,0);
    index_t len = 0;
    index_t n_spin = 0;

    std::string fname = gen_graphene_hr_dat();

    char fcontent[1024];
    FILE* fptr = fopen(fname.c_str(), "r");
    int nread = fread(fcontent, 1, 1024, fptr);
    fclose(fptr);
    memset( fcontent+nread, 0, 1024-nread );
    mpi_fil_printf( "generated file '%s' as W90 file for testing. Content:\n%s",
            fname.c_str(), fcontent );
    rs_hopping_t* cmp = model->hop;
    rs_hopping_t* hop = diverge_read_W90_C(fname.c_str(),n_spin,&len);
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
