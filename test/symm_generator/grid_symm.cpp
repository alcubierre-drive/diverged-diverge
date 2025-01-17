#ifndef DIVERGE_SKIP_TESTS

#include "../catch.hpp"
#include "../../src/diverge_model.h"
#include "../../src/diverge_flow_step.h"
#include "../../src/diverge_flow_step_internal.hpp"
#include "../model_examples/old.h"
#include "../../src/diverge_model_internals.h"
#include "../model_examples/model_constructor.hpp"
#include "../../src/grid/loops.hpp"
#include "../helper_functions.hpp"



TEST_CASE("loopgrid symmcheck", "[grid_symm]") {
    diverge_model_t* mod_grid = gen_triangular_pxpy_sym(6, 6);
    diverge_model_internals_grid(mod_grid);
    index_t nb = mod_grid->n_orb*mod_grid->n_spin;
    index_t nk = kdim(mod_grid->nk);

    complex128_t* Lgrid;
    complex128_t* Lreco;

    grid::Loops grid_loop(mod_grid);
    index_t fsize = POW4(nb)*grid_loop.get_qsize()*nk;

    grid_loop.toggle_symmetrizer();
    Lgrid = grid_loop.X_C(0.1);
    grid_loop.toggle_symmetrizer();
    Lreco = grid_loop.X_C(0.1);
    for(index_t i = 0; i < fsize; ++i)
        CHECK(std::abs(Lreco[i]-Lgrid[i]) < 1e-12);

    grid_loop.toggle_symmetrizer();
    Lgrid = grid_loop.X_P(0.1);
    grid_loop.toggle_symmetrizer();
    Lreco = grid_loop.X_P(0.1);
    for(index_t i = 0; i < fsize; ++i)
        CHECK(std::abs(Lreco[i]-Lgrid[i]) < 1e-12);

    diverge_model_free(mod_grid);
}

#endif // DIVERGE_SKIP_TESTS
