#ifndef DIVERGE_SKIP_TESTS

#include "../catch.hpp"
#include "../../src/diverge_model.h"
#include "../../src/diverge_flow_step.h"
#include "../../src/diverge_flow_step_internal.hpp"
#include "../../src/diverge_patching.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/diverge_model_output.h"

#include "../../src/grid/vertex_memory.hpp"

#include "../../src/tu/vertex.hpp"
#include "../../src/tu/projection_wrapper.hpp"
#include "../../src/tu/flow_stepper.hpp"

#include "../model_examples/model_constructor.hpp"
#include <sys/stat.h>

#ifndef BHK_NK
#define BHK_NK 4
#endif

#ifndef BHK_L0
#define BHK_L0 1.0
#endif

#ifndef BHK_NUM
#define BHK_NUM 5
#endif

#ifdef USE_GF_FLOATS
#ifndef BHK_EQUAL
#define BHK_EQUAL 1e-6
#endif
#else
#define BHK_EQUAL 1e-11
#endif


static double dot(double* A, double* B) {
    return A[0]*B[0]+A[1]*B[1]+A[2]*B[2];
}

double test_symm_4_pnt( diverge_model_t* model, complex128_t* buf ) {
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
    memcpy( aux, buf, size * sizeof(complex128_t) );
    memset( (void*) buf, 0, size * sizeof(complex128_t) );
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
                    conj(symm_pref[xo3] * (cos(kdotR33) - I128 * sin(kdotR33))) *
                    conj(symm_pref[xo4] * (cos(kdotR44) - I128 * sin(kdotR44)));
                buf[IDX7(k1,k2,k3, o1,o2,o3,o4, nk,nk, n_os,n_os,n_os,n_os)] +=
                     val * norm;
            }
        }
    }
    double max_diff = 0.;
    #pragma omp parallel for reduction(max:max_diff) num_threads(diverge_omp_num_threads())
    for (index_t idx=0; idx<size; ++idx)
        max_diff = MAX(max_diff, std::abs(buf[idx]-aux[idx]));
    mpi_log_printf("maximal error %2.5f \n", max_diff);
    free(aux);
    return max_diff;
}

typedef enum {
    bhk_mode_none,
    bhk_mode_tu,
    bhk_mode_grid,
    bhk_mode_patch,
} bhk_mode_t;

static inline complex128_t* BHK_test_run( diverge_model_t* mod ) {

    #ifdef BHK_OUTPUT_FILES
    char fname[2048];
    mkdir("bhk.dir", S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH);

    sprintf(fname, "bhk.dir/model-%s.dvg", mod->name); \
    diverge_model_to_file( mod, fname );
    #endif

    char* mode = NULL;
    char mode_tu[] = "tu";
    char mode_grid[] = "grid";
    char mode_patch[] = "patch";
    bhk_mode_t bhk_mode = bhk_mode_none;

    if (strstr(mod->name, "_tu")) mode = mode_tu, bhk_mode = bhk_mode_tu;
    else if (strstr(mod->name, "_grid")) mode = mode_grid, bhk_mode = bhk_mode_grid;
    else if (strstr(mod->name, "_patch")) mode = mode_patch, bhk_mode = bhk_mode_patch;
    else bhk_mode = bhk_mode_none;
    if (bhk_mode == bhk_mode_none) {
        mpi_err_printf("no mode\n");
        return NULL;
    }
    diverge_flow_step_t* step = diverge_flow_step_init(mod, mode, "PCD");

    double Lambda = BHK_L0;
    double dLambda = -0.1;
    for (int i=0; i<BHK_NUM; ++i) {
        if (Lambda < 0) break;
        mpi_log_printf("%s: %.5e…", mod->name, Lambda);
        mpi_eprintf( "\n%c[1A", 27 );
        fflush(stderr);
        diverge_flow_step_euler( step, Lambda, dLambda );

        Lambda += dLambda;
    }
    mpi_eprintf("\n");
    fflush(stderr);

    index_t fsize = POW4(mod->n_orb*mod->n_spin)*POW3(kdim(mod->nk));
    complex128_t* vertex = (complex128_t*)calloc(fsize,sizeof(complex128_t));

    switch (bhk_mode) {
        case bhk_mode_tu:
            step->tu_vertex->project_fullV(mod,vertex); break;
        case bhk_mode_grid:
            ((grid::VertexMemory*)step->intls->grid_vertex)->get_vertex( vertex, 'V' ); break;
        case bhk_mode_patch:
            memcpy( vertex, diverge_flow_step_vertex( step, 'V' ).ary, sizeof(complex128_t)*fsize ); break;
        default: break;
    }

    #ifdef BHK_OUTPUT_FILES
    sprintf(fname, "bhk.dir/%s.bin", mod->name);
    if (diverge_mpi_comm_rank() == 0) {
        FILE* f = fopen(fname, "wb"); fwrite(vertex, sizeof(complex128_t), fsize, f); fclose(f);
    }
    #endif

    diverge_flow_step_free(step);

    return vertex;
}

static inline void compare_vertices( complex128_t* v1, complex128_t* v2, index_t size ) {
    double csum_nosort = 0.0, cmax_nosort = 0.0;

    #pragma omp parallel for reduction(+:csum_nosort) reduction(max:cmax_nosort) num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<size; ++i) {
        double cval = std::abs(v1[i]-v2[i]);
        csum_nosort += cval;
        cmax_nosort = MAX(cmax_nosort, cval);
    }

    CHECK( csum_nosort / size < BHK_EQUAL );
    CHECK( cmax_nosort < BHK_EQUAL );
}

#define BHK_diverge_model_internals_patch( mod_patch ) \
    index_t nkp = mod_patch->nk[0]*mod_patch->nk[1]; \
    vector<index_t> patches(nkp); for (int i=0; i<nkp; ++i) patches[i] = i; \
    mod_patch->patching = diverge_patching_from_indices( mod_patch, patches.data(), patches.size() ); \
    diverge_model_internals_patch(mod_patch, -1);

#define BHK_diverge_model_internals_grid( mod_grid ) \
    diverge_model_internals_grid( mod_grid );

#define BHK_diverge_model_internals_tu( mod_tu ) \
    diverge_model_internals_tu( mod_tu, 12.0 );

#define create_bhk_test_case( mode1, mode2, model ) \
TEST_CASE( "diverge_flow_bhk_" #model "_" #mode1 "_" #mode2, "[BHK][" #model "][" #mode1 "-" #mode2 "]" ) { \
    diverge_model_t* mod1 = gen_ ## model (BHK_NK, BHK_NK); \
    diverge_model_t* mod2 = gen_ ## model (BHK_NK, BHK_NK); \
    strcpy( mod1->name, #model "_" #mode1 ); \
    strcpy( mod2->name, #model "_" #mode2 ); \
    BHK_diverge_model_internals_ ## mode1 ( mod1 ); \
    diverge_model_validate( mod1 ); \
    BHK_diverge_model_internals_ ## mode2 ( mod2 ); \
    diverge_model_validate( mod2 ); \
    complex128_t* v1 = BHK_test_run( mod1 ); \
    complex128_t* v2 = BHK_test_run( mod2 ); \
    compare_vertices( v1, v2, POW3(kdim(mod1->nk)) * POW4(mod1->n_orb * mod1->n_spin) ); \
    diverge_model_free( mod1 ); \
    diverge_model_free( mod2 ); \
    free( v1 ); \
    free( v2 ); \
}

#define create_bhk_test_case_sym( mode, model ) \
TEST_CASE( "diverge_flow_bhk_" #model "_" #mode "_sym", "[BHK][" #model "][" #mode ":sym]" ) { \
    diverge_model_t* mod1 = gen_ ## model##_sym (BHK_NK, BHK_NK); \
    mod1->n_sym = 0;\
    diverge_model_t* mod2 = gen_ ## model##_sym (BHK_NK, BHK_NK); \
    strcpy( mod1->name, #model "_" #mode ); \
    strcpy( mod2->name, #model "_" #mode "_sym" ); \
    { BHK_diverge_model_internals_ ## mode ( mod1 ); } \
    diverge_model_validate( mod1 ); \
    { BHK_diverge_model_internals_ ## mode ( mod2 ); } \
    diverge_model_validate( mod2 ); \
    complex128_t* v1 = BHK_test_run( mod1 ); \
    complex128_t* v2 = BHK_test_run( mod2 ); \
    compare_vertices( v1, v2, POW3(kdim(mod1->nk)) * POW4(mod1->n_orb * mod1->n_spin) ); \
    CHECK(test_symm_4_pnt( mod2, v1 ) < 1e-12); \
    CHECK(test_symm_4_pnt( mod2, v2 ) < 1e-12); \
    diverge_model_free( mod1 ); \
    diverge_model_free( mod2 ); \
    free( v1 ); \
    free( v2 ); \
}


create_bhk_test_case( grid, patch, square_lat_hub )
create_bhk_test_case( grid, patch, honeycomb_lat_hub )
create_bhk_test_case( grid, patch, square_lat_rashba )
create_bhk_test_case( grid, patch, random )
create_bhk_test_case( tu, grid, square_lat_hub )
create_bhk_test_case( tu, grid, honeycomb_lat_hub )
create_bhk_test_case( tu, grid, square_lat_rashba )
create_bhk_test_case( tu, grid, random )

create_bhk_test_case( tu, grid, square_lat_hub_sym )
create_bhk_test_case( tu, grid, honeycomb_lat_hub_sym )
create_bhk_test_case( tu, grid, square_lat_rashba_sym )

create_bhk_test_case( grid, patch, square_lat_hub_proj );
create_bhk_test_case( tu, grid, square_lat_hub_proj );

create_bhk_test_case_sym( grid, square_lat_hub )
create_bhk_test_case_sym( grid, square_lat_rashba )
create_bhk_test_case_sym( grid, honeycomb_lat_hub )
create_bhk_test_case_sym( grid, triangular_pxpy )

create_bhk_test_case_sym( patch, square_lat_hub )
create_bhk_test_case_sym( patch, square_lat_rashba )
create_bhk_test_case_sym( patch, honeycomb_lat_hub )
create_bhk_test_case_sym( patch, triangular_pxpy )

create_bhk_test_case_sym( tu, square_lat_hub )
create_bhk_test_case_sym( tu, square_lat_rashba )
create_bhk_test_case_sym( tu, honeycomb_lat_hub )
create_bhk_test_case_sym( tu, triangular_pxpy )


#ifdef BEHAK_ADDITIONAL_TEST_CASES
create_bhk_test_case( tu, patch, square_lat_hub )
create_bhk_test_case( tu, patch, honeycomb_lat_hub )
create_bhk_test_case( tu, patch, square_lat_rashba )
create_bhk_test_case( tu, patch, random )
create_bhk_test_case( grid, grid, square_lat_hub )
create_bhk_test_case( grid, grid, honeycomb_lat_hub )
create_bhk_test_case( grid, grid, square_lat_rashba )
create_bhk_test_case( grid, grid, random )
#endif

#endif // DIVERGE_SKIP_TESTS
