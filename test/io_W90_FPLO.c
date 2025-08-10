#ifndef DIVERGE_SKIP_TESTS

#define CT_SUITE_NAME read_hoppings
#include "ctest.h"

#include "../src/diverge_model.h"
#include "../src/diverge_momentum_gen.h"
#include "models/all.h"
#include "../src/diverge_model_internals.h"
#include "helper_functions.h"
#include "../src/misc/read_W90.h"
#include "../src/misc/read_fplo.h"
#include "../src/misc/bitary.h"
#include "../src/diverge_internals_struct.h" // strdup_div

#include "io_W90_FPLO_data.h"

CT_TEST(FPLO_reader) {
    char* fname_W90 = strdup_div(gen_graphene_hr_dat(0));
    char* fname_FPLO = strdup_div(gen_graphene_hr_dat(1));
    FILE* fptr = fopen(fname_FPLO, "r");
    char fcontent[1024];
    int nread = fread(fcontent, 1, 1024, fptr);
    fclose(fptr);
    memset( fcontent+nread, 0, 1024-nread );
    mpi_fil_printf( "generated file '%s' (+hamdata). Content:\n%s", fname_FPLO, fcontent );

    diverge_model_t* fplo_model = diverge_read_fplo( fname_FPLO );
    CT_CHECK( fplo_model != NULL );
    if (!fplo_model) return;

    index_t len = -1;
    index_t n_orb = DIVERGE_W90_READER_SWAP_ORBITALS;
    rs_hopping_t* w90_hop = diverge_read_W90_C( fname_W90, 1, &len, &n_orb );
    remove( fname_FPLO ); free( fname_FPLO );
    remove( fname_W90 ); free( fname_W90 );

    cbyte_t* masked_hoppings = bitary_alloc( len );
    for (rs_hopping_t* h=fplo_model->hop; h!=fplo_model->hop+fplo_model->n_hop; ++h) {
        for (index_t i=0; i<len; ++i) {
            rs_hopping_t* t = w90_hop+i;
            if (h->R[0] == t->R[0] && h->R[1] == t->R[1] && h->R[2] == t->R[2] &&
                h->o1 == t->o1 && h->o2 == t->o2 && h->s1 == t->s1 && h->s2 == t->s2 &&
                cabs(t->t - h->t) < 1.e-5) {
                bitary_set( masked_hoppings, i );
                break;
            }
        }
    }
    for (index_t i=0; i<len; ++i) {
        if (cabs(w90_hop[i].t) < 1.e-5)
            bitary_set( masked_hoppings, i );
        CT_CHECK( bitary_get( masked_hoppings, i ) );
    }
    free( masked_hoppings );

    diverge_model_free( fplo_model );
    free( w90_hop );
}

CT_TEST(W90_reader) {
    diverge_model_t* model = gen_honeycomb_lat_hub(2,2,1,1,1,0,0,0);
    index_t len = 0;
    index_t n_spin = 0;

    const char* fname = gen_graphene_hr_dat(0);

    char fcontent[1024];
    FILE* fptr = fopen(fname, "r");
    int nread = fread(fcontent, 1, 1024, fptr);
    fclose(fptr);
    memset( fcontent+nread, 0, 1024-nread );
    mpi_fil_printf( "generated file '%s' as W90 file for testing. Content:\n%s", fname, fcontent );
    rs_hopping_t* cmp = model->hop;
    rs_hopping_t* hop = diverge_read_W90_C(fname,n_spin,&len,NULL);
    remove( fname );
    // check all hoppings exist and are equal
    CT_CHECK(len > 0);
    for(index_t elem = 0; elem < len; ++elem) {
        bool found = false;
        if(cabs(hop[elem].t) > 1e-10) {
            for(index_t trial = 0; trial < model->n_hop; ++trial) {
                if(hop[elem].o1 == cmp[trial].o2 &&
                    hop[elem].o2 == cmp[trial].o1 &&
                    hop[elem].s1 == cmp[trial].s2 &&
                    hop[elem].s2 == cmp[trial].s1 &&
                    hop[elem].R[0] == cmp[trial].R[0] &&
                    hop[elem].R[1] == cmp[trial].R[1] &&
                    hop[elem].R[2] == cmp[trial].R[2] &&
                    cabs(hop[elem].t - cmp[trial].t) < 1e-5)
                        found = true;
            }
        } else {
            found = true;
        }
        CT_CHECK( found );
    }
    free( hop );
    diverge_model_free( model );
}

CT_TEST_NAME(W90_reader_rev, "W90_reader,reversed") {
    diverge_model_t* model = gen_honeycomb_lat_hub(2,2,1,1,1,0,0,0);
    index_t len = 0;
    index_t n_spin = 0;

    const char* fname = gen_graphene_hr_dat(0);

    char fcontent[1024];
    FILE* fptr = fopen(fname, "r");
    int nread = fread(fcontent, 1, 1024, fptr);
    fclose(fptr);
    memset( fcontent+nread, 0, 1024-nread );
    mpi_fil_printf( "generated file '%s' as W90 file for testing. Content:\n%s", fname, fcontent );
    rs_hopping_t* cmp = model->hop;
    index_t n_orb = DIVERGE_W90_READER_SWAP_ORBITALS;
    rs_hopping_t* hop = diverge_read_W90_C(fname,n_spin,&len,&n_orb);
    remove( fname );
    // check all hoppings exist and are equal
    CT_CHECK(len > 0);
    for(index_t elem = 0; elem < len; ++elem) {
        bool found = false;
        if(cabs(hop[elem].t) > 1e-10) {
            for(index_t trial = 0; trial < model->n_hop; ++trial) {
                // here we have the swapped hopping elements!!
                if(hop[elem].o2 == cmp[trial].o2 &&
                    hop[elem].o1 == cmp[trial].o1 &&
                    hop[elem].s2 == cmp[trial].s2 &&
                    hop[elem].s1 == cmp[trial].s1 &&
                    hop[elem].R[0] == cmp[trial].R[0] &&
                    hop[elem].R[1] == cmp[trial].R[1] &&
                    hop[elem].R[2] == cmp[trial].R[2] &&
                    cabs(hop[elem].t - cmp[trial].t) < 1e-5)
                        found = true;
            }
        } else {
            found = true;
        }
        CT_CHECK( found );
    }
    free( hop );
    diverge_model_free( model );
}

#endif // DIVERGE_SKIP_TESTS
