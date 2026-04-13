#define CT_SUITE_NAME grid_sym
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_flow_step.h"
#include "../../src/diverge_flow_step_internal.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/grid/loops.h"

#include "../helper_functions.h"
#include "../models/all.h"

CT_TEST(loop) {
    diverge_model_t* mod_grid = KWARGS(gen_triangular_pxpy_sym, .nkx=6, .nky=6);
    diverge_model_internals_grid(mod_grid);
    index_t nb = mod_grid->n_orb*mod_grid->n_spin;
    index_t nk = kdim(mod_grid->nk);

    complex128_t* Lgrid;
    complex128_t* Lreco;

    grid_loop_t* grid_loop = grid_loop_init(mod_grid);
    index_t fsize = POW4(nb)*grid_loop->qsize*nk;

    grid_loop_toggle_symmetrizer(grid_loop);
    Lgrid = grid_loop_calc( grid_loop, 'C', 0.1 );
    grid_loop_toggle_symmetrizer(grid_loop);
    Lreco = grid_loop_calc( grid_loop, 'C', 0.1 );
    for(index_t i = 0; i < fsize; ++i)
        CT_CHECK(cabs(Lreco[i]-Lgrid[i]) < 1e-12);

    grid_loop_toggle_symmetrizer(grid_loop);
    Lgrid = grid_loop_calc( grid_loop, 'P', 0.1 );
    grid_loop_toggle_symmetrizer(grid_loop);
    Lreco = grid_loop_calc( grid_loop, 'P', 0.1 );
    for(index_t i = 0; i < fsize; ++i)
        CT_CHECK(cabs(Lreco[i]-Lgrid[i]) < 1e-12);

    grid_loop_free(grid_loop);
    diverge_model_free(mod_grid);
}

#endif // DIVERGE_SKIP_TESTS
