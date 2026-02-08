#include <diverge.h>
#include <diverge_misc.h>

static const double tp = -0.2;

int main( int argc, char** argv ) {
    diverge_init( &argc, &argv );
    diverge_compilation_status();
    diverge_model_t* m = diverge_model_init();

    strcpy(m->name, __FILE__ );
    m->nk[0] = m->nk[1] = 32;
    m->nkf[0] = m->nkf[1] = 1;
    m->lattice[0][0] = m->lattice[1][1] = m->lattice[2][2] = 1.0;
    m->n_orb = m->SU2 = m->n_spin = 1;
    m->ibz_path[1][0] = m->ibz_path[2][0] = m->ibz_path[2][1] = 0.5;
    m->n_ibz_path = 4;

    m->hop = diverge_mem_alloc_rs_hopping_t(128);
    m->hop[m->n_hop++] = (rs_hopping_t){.R={ 1, 0,0}, .t=1};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={-1, 0,0}, .t=1};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={ 0, 1,0}, .t=1};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={ 0,-1,0}, .t=1};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={ 1, 1,0}, .t=tp};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={-1, 1,0}, .t=tp};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={ 1,-1,0}, .t=tp};
    m->hop[m->n_hop++] = (rs_hopping_t){.R={-1,-1,0}, .t=tp};

    m->vert = diverge_mem_alloc_rs_vertex_t(128);
    m->vert[m->n_vert++] = (rs_vertex_t){.chan='D', .R={0,0,0}, .V=3};

    if (diverge_model_validate(m)) {
        mpi_usr_printf("invalid model...\n");
        return EXIT_FAILURE;
    }

    diverge_model_internals_common(m);
    diverge_model_set_chempot(m, NULL, -1, -4.*tp);

    // IOBI
    diverge_model_internals_tu(m, 0.01);
    diverge_flow_step_t* f = diverge_flow_step_init_any(m, "PCD");
    diverge_euler_t eu = diverge_euler_defaults;
    double vmax = 0.0;

    // just use the full kmesh, we're kinda lazy. Actually we should try to find
    // the FS using the patching infrastructure...
    index_t n_kidx = kdim(m->nk);
    index_t* kidx = calloc(n_kidx, sizeof*kidx);
    for (index_t ik=0; ik<n_kidx; ++ik) kidx[ik] = ik;
    complex128_t* chi_pp = calloc(n_kidx, sizeof*chi_pp);

    do {
        diverge_flow_step_euler( f, eu.Lambda, eu.dLambda );
        diverge_flow_step_vertmax( f, &vmax );
        // integrate chi_pp
        diverge_flow_step_sc_greens( f, kidx, n_kidx, eu.Lambda, eu.dLambda, chi_pp );
        mpi_printf( "%.5e %.5e\n", eu.Lambda, vmax );
    } while (diverge_euler_next(&eu, vmax));

    // get the vertex
    complex128_t* V_p = calloc(POW2(n_kidx), sizeof*V_p);
    diverge_flow_step_sc_vertex( f, 'c', kidx, n_kidx, V_p );
    for (index_t ik1=0; ik1<n_kidx; ++ik1)
    for (index_t ik2=0; ik2<n_kidx; ++ik2) {
        // now negative eigenvalues should be attractive!
        V_p[IDX2(ik1,ik2,n_kidx)] *= -0.5*chi_pp[ik2];
    }
    free( kidx );
    free( chi_pp );

    // and eigensolve+sort it
    complex128_t* U = calloc(POW2(n_kidx), sizeof*U);
    complex128_t* V = calloc(POW2(n_kidx), sizeof*V);
    double* S = calloc(n_kidx, sizeof*S);
    single_svd( V_p, U, V, S, -n_kidx ); // dispatch to non-hermitian eigen
    double* Sx = calloc(n_kidx, sizeof*Sx);
    memcpy( Sx, S, sizeof*Sx*n_kidx );
    transpose_inplace(U, n_kidx);
    single_eigen_sort(U, Sx, n_kidx, 'N');
    transpose_inplace(V, n_kidx);
    single_eigen_sort(V, S, n_kidx, 'N');
    int nval = 10;
    FILE* out = fopen(__FILE__ ".bin", "wb");
    fwrite(S, sizeof*S, nval, out);
    fwrite(U, sizeof*U, nval*n_kidx, out);
    fwrite(V, sizeof*V, nval*n_kidx, out);
    fclose(out);
    free(Sx);
    free(S);
    free(U);
    free(V);
    free(V_p);

    diverge_flow_step_free(f);
    diverge_model_free(m);
    diverge_finalize();

    return 0;
}

