#define CT_SUITE_NAME phonons
#include "ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../src/diverge.h"
#include "../src/diverge_phonons.h"
#include "../src/misc/linalg.h"
LINALG_IMPLEMENT

static diverge_model_t* hubbard_ssh_model( int SU2, int nk ) {
    double omega=1.0;
    double g=2.0;
    double t=1.0;
    double max_dist=1.01;
    index_t nkf=5;

    diverge_model_t* m = diverge_model_init();
    strcpy(m->name, "hubbard ssh");
    m->nk[0] = m->nk[1] = nk;
    m->nkf[0] = m->nkf[1] = nkf;
    m->lattice[0][0] = m->lattice[1][1] = m->lattice[2][2] = 1.0;
    m->n_spin = (m->SU2=SU2) ? 1 : 2;
    m->n_orb = 1;

    m->hop = calloc(64, sizeof*m->hop);
    for (int i=0; i<4; ++i) {
        m->hop[m->n_hop].R[i/2] = i%2 ? 1:-1;
        m->hop[m->n_hop].s1 = -1; // auto SU2
        m->hop[m->n_hop++].t = t;
    }
    m->vert = calloc(1, sizeof*m->vert);

    diverge_model_internals_common(m);

    rs_hopping_t phop[] = {
        { .o1=0, .o2=0, .t=omega },
        { .o1=1, .o2=1, .t=omega },
    };
    ph_vertex_t pvert[] = {
        {.mode='x', .R12={ 1, 0, 0}, .s1=-1, .OA=0, .RA_pos={+0.5, 0, 0 }, .g=g },
        {.mode='x', .R12={-1, 0, 0}, .s1=-1, .OA=0, .RA_pos={-0.5, 0, 0 }, .g=g },
        {.mode='y', .R12={ 0, 1, 0}, .s1=-1, .OA=1, .RA_pos={ 0,+0.5, 0 }, .g=g },
        {.mode='y', .R12={ 0,-1, 0}, .s1=-1, .OA=1, .RA_pos={ 0,-0.5, 0 }, .g=g },
    };
    diverge_model_internals_tu_phonons(m, max_dist, phop, sizeof(phop)/sizeof(phop[0]),
            pvert, sizeof(pvert)/sizeof(pvert[0]));
    return m;
}

CT_TEST_NAME(hubbard_SSH_model, "hubbard-ssh,SU2") {
    diverge_model_t* m = hubbard_ssh_model(1, 12);
    diverge_flow_step_t* st = diverge_flow_step_init_any(m, "PCD");
    diverge_euler_t eu = diverge_euler_defaults_CPP();
    double cmax[3], vmax;
    VECTOR_INIT(double, PCD_max);
    do {
        diverge_flow_step_euler( st, eu.Lambda, eu.dLambda );
        diverge_flow_step_chanmax( st, cmax );
        diverge_flow_step_vertmax( st, &vmax );
        double Pmax = diverge_flow_step_eigchan( st, 'p', 0 ),
               Cmax = diverge_flow_step_eigchan( st, 'm', 0 ),
               Dmax = diverge_flow_step_eigchan( st, 'h', 0 );
        VECTOR_PUSH_BACK(PCD_max, Pmax);
        VECTOR_PUSH_BACK(PCD_max, Cmax);
        VECTOR_PUSH_BACK(PCD_max, Dmax);
        mpi_log_printf("%s: %5.2f  %5.2f %5.2f %5.2f…", m->name, eu.Lambda, Pmax, Cmax, Dmax);
        if (mpi_loglevel_get() >= mpi_loglevel_log) mpi_eprintf( "\n%c[1A", 27 ), fflush(stderr);
    } while (diverge_euler_next(&eu, vmax));
    if (mpi_loglevel_get() >= mpi_loglevel_log) mpi_eprintf( "\n" ), fflush(stderr);

    diverge_flow_step_free( st );
    diverge_model_free( m );

    for (index_t i=0; i<PCD_max_sz/3; ++i) {
        double PC = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,1,3)]),
               PD = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,2,3)]);
        CT_CHECK( PC < 1e-4 );
        CT_CHECK( PD < 1e-4 );
    }
    VECTOR_FREE(PCD_max);
}

CT_TEST_NAME(hubbard_SSH_model_nonSU2, "hubbard-ssh,non-SU2") {
    diverge_model_t* m = hubbard_ssh_model(0, 8);
    diverge_flow_step_t* st = diverge_flow_step_init_any(m, "PCD");
    diverge_euler_t eu = diverge_euler_defaults_CPP();
    double cmax[3], vmax;
    VECTOR_INIT(double, PCD_max);
    do {
        diverge_flow_step_euler( st, eu.Lambda, eu.dLambda );
        diverge_flow_step_chanmax( st, cmax );
        diverge_flow_step_vertmax( st, &vmax );
        double Pmax = diverge_flow_step_eigchan( st, 'p', 0 ),
               Cmax = diverge_flow_step_eigchan( st, 'c', 0 ),
               Dmax = diverge_flow_step_eigchan( st, 'd', 0 );
        VECTOR_PUSH_BACK(PCD_max, 0.5*Pmax);
        VECTOR_PUSH_BACK(PCD_max, Cmax);
        VECTOR_PUSH_BACK(PCD_max, Dmax);
        mpi_log_printf("%s: %5.2f  %5.2f %5.2f %5.2f…", m->name, eu.Lambda, Pmax, Cmax, Dmax);
        if (mpi_loglevel_get() >= mpi_loglevel_log) mpi_eprintf( "\n%c[1A", 27 ), fflush(stderr);
    } while (diverge_euler_next(&eu, vmax));
    if (mpi_loglevel_get() >= mpi_loglevel_log) mpi_eprintf( "\n" ), fflush(stderr);

    diverge_flow_step_free( st );
    diverge_model_free( m );

    for (index_t i=0; i<PCD_max_sz/3; ++i) {
        double PC = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,1,3)]),
               PD = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,2,3)]);
        CT_CHECK( PC < 1e-4 );
        CT_CHECK( PD < 1e-4 );
    }
    VECTOR_FREE(PCD_max);
}

CT_TEST_NAME(hubbard_SSH_model_self, "hubbard-ssh,SU2,self") {
    diverge_model_t* m = hubbard_ssh_model(1, 12);
    double nu = diverge_model_get_filling(m, NULL, -1);
    diverge_flow_step_t* st = diverge_flow_step_init_any(m, "PCDS");
    diverge_euler_t eu = diverge_euler_defaults_CPP();
    double cmax[3], vmax;
    VECTOR_INIT(double, PCD_max);
    int ll = mpi_loglevel_get(); if (ll > mpi_loglevel_log) mpi_loglevel_set(mpi_loglevel_log);
    do {
        diverge_flow_step_euler( st, eu.Lambda, eu.dLambda );
        diverge_flow_step_refill( st, nu, NULL );
        diverge_flow_step_chanmax( st, cmax );
        diverge_flow_step_vertmax( st, &vmax );
        double Pmax = diverge_flow_step_eigchan( st, 'p', 0 ),
               Cmax = diverge_flow_step_eigchan( st, 'm', 0 ),
               Dmax = diverge_flow_step_eigchan( st, 'h', 0 );
        VECTOR_PUSH_BACK(PCD_max, Pmax);
        VECTOR_PUSH_BACK(PCD_max, Cmax);
        VECTOR_PUSH_BACK(PCD_max, Dmax);
        mpi_log_printf("%s: %5.2f  %5.2f %5.2f %5.2f…", m->name, eu.Lambda, Pmax, Cmax, Dmax);
        if (mpi_loglevel_get() >= mpi_loglevel_log) mpi_eprintf( "\n%c[1A", 27 ), fflush(stderr);
    } while (diverge_euler_next(&eu, vmax));
    mpi_loglevel_set(ll);
    if (mpi_loglevel_get() >= mpi_loglevel_log) mpi_eprintf( "\n" ), fflush(stderr);

    diverge_flow_step_free( st );
    diverge_model_free( m );

    for (index_t i=0; i<PCD_max_sz/3; ++i) {
        double PC = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,1,3)]),
               PD = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,2,3)]);
        CT_CHECK( PC < 1e-4 );
        CT_CHECK( PD < 1e-4 );
    }
    VECTOR_FREE(PCD_max);
}

CT_TEST_NAME(hubbard_SSH_model_nonSU2_self, "hubbard-ssh,non-SU2,self") {
    diverge_model_t* m = hubbard_ssh_model(0, 8);
    double nu = diverge_model_get_filling(m, NULL, -1);
    diverge_flow_step_t* st = diverge_flow_step_init_any(m, "PCDS");
    diverge_euler_t eu = diverge_euler_defaults_CPP();
    double cmax[3], vmax;
    VECTOR_INIT(double, PCD_max);
    int ll = mpi_loglevel_get(); if (ll > mpi_loglevel_log) mpi_loglevel_set(mpi_loglevel_log);
    do {
        diverge_flow_step_euler( st, eu.Lambda, eu.dLambda );
        diverge_flow_step_refill( st, nu, NULL );
        diverge_flow_step_chanmax( st, cmax );
        diverge_flow_step_vertmax( st, &vmax );
        double Pmax = diverge_flow_step_eigchan( st, 'p', 0 ),
               Cmax = diverge_flow_step_eigchan( st, 'c', 0 ),
               Dmax = diverge_flow_step_eigchan( st, 'd', 0 );
        VECTOR_PUSH_BACK(PCD_max, 0.5*Pmax);
        VECTOR_PUSH_BACK(PCD_max, Cmax);
        VECTOR_PUSH_BACK(PCD_max, Dmax);
        mpi_log_printf("%s: %5.2f  %5.2f %5.2f %5.2f…", m->name, eu.Lambda, Pmax, Cmax, Dmax);
        if (mpi_loglevel_get() >= mpi_loglevel_log) mpi_eprintf( "\n%c[1A", 27 ), fflush(stderr);
    } while (diverge_euler_next(&eu, vmax));
    mpi_loglevel_set(ll);
    if (mpi_loglevel_get() >= mpi_loglevel_log) mpi_eprintf( "\n" ), fflush(stderr);

    diverge_flow_step_free( st );
    diverge_model_free( m );

    for (index_t i=0; i<PCD_max_sz/3; ++i) {
        double PC = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,1,3)]),
               PD = fabs(PCD_max[IDX2(i,0,3)] - PCD_max[IDX2(i,2,3)]);
        CT_CHECK( PC < 1e-4 );
        CT_CHECK( PD < 1e-4 );
    }
    VECTOR_FREE(PCD_max);
}

#include "../src/misc/mergesort.h"
int phsort_cmp(ph_vertex_t a, ph_vertex_t b) {
    int c = 0;
    int a_abs = cabs(a.g) > 1.e-10,
        b_abs = cabs(b.g) > 1.e-10;
    if (!a_abs) return 1;
    if (!b_abs) return -1;
    c = a.R12[0] - b.R12[0]; if (c) return c;
    c = a.R12[1] - b.R12[1]; if (c) return c;
    c = a.R12[2] - b.R12[2]; if (c) return c;
    c = a.RA[0] - b.RA[0]; if (c) return c;
    c = a.RA[1] - b.RA[1]; if (c) return c;
    c = a.RA[2] - b.RA[2]; if (c) return c;
    return c;
}
#undef MERGESORT_COMP
#define MERGESORT_COMP(A,B) phsort_cmp(A,B)
MERGESORT_IMPL(phsort, ph_vertex_t)

CT_TEST_NAME(acoustic_ssh_model, "epc,asr,ssh") {
    diverge_model_t *e = diverge_model_init(),
                    *p = diverge_model_init();

    e->nk[0] = e->nk[1] = p->nk[0] = p->nk[1] = 8;
    e->nkf[0] = e->nkf[1] = p->nkf[0] = p->nkf[1] = 1;
    strcpy(e->name, "a-ssh:el");
    strcpy(p->name, "a-ssh:ph");
    e->lattice[0][0] = e->lattice[1][1] = e->lattice[2][2] = 1.0;
    p->lattice[0][0] = p->lattice[1][1] = p->lattice[2][2] = 1.0;

    e->ibz_path[1][0] = e->ibz_path[2][0] = e->ibz_path[2][1] = 0.5;
    e->n_ibz_path = 4;
    p->ibz_path[1][0] = p->ibz_path[2][0] = p->ibz_path[2][1] = 0.5;
    p->n_ibz_path = 4;

    e->n_orb = e->n_spin = e->SU2 = 1;
    p->n_orb = 3;
    p->n_spin = p->SU2 = 1;

    double tp = 0.1;
    claM3d_t C_diag = {{{-0.1, 0.00, 0.00},
                        { 0.0,-0.08, 0.00},
                        { 0.0, 0.00,-0.02}}};

    VECTOR_INIT(rs_hopping_t, e_hop);
    VECTOR_INIT(rs_hopping_t, p_hop);
    for (int i=-2; i<=2; ++i)
    for (int j=-2; j<=2; ++j) {
        claV3d_t bond = claM3d_vecmat((claV3d_t){{i,j,0}}, claM3d_map(e->lattice[0]));
        double bond_dist = claV3d_norm(bond);
        double theta = atan2(bond.v[1], bond.v[0]);
        claM3d_t rot = claM3d_id();
        rot.m[0][0] =   rot.m[1][1] = cos(theta);
        rot.m[0][1] = -(rot.m[1][0] = sin(theta));
        claM3d_t C_bond = claM3d_matmul(rot, claM3d_matmul(C_diag, claM3d_transpose(rot)));
        if (fabs(bond_dist-1.0) < 1.e-5) {
            VECTOR_PUSH_BACK(e_hop, (rs_hopping_t){.R={i,j,0}, .t=1.0});
            for (int x=0; x<3; ++x)
            for (int y=0; y<3; ++y) {
                VECTOR_PUSH_BACK(p_hop, (rs_hopping_t){.R={i,j,0}, .o1=x, .o2=y, .t=C_bond.m[x][y]});
            }
        } else if (fabs(bond_dist-sqrt(2.0)) < 1.e-5) {
            VECTOR_PUSH_BACK(e_hop, (rs_hopping_t){.R={i,j,0}, .t=tp});
            for (int x=0; x<3; ++x)
            for (int y=0; y<3; ++y) {
                VECTOR_PUSH_BACK(p_hop, (rs_hopping_t){.R={i,j,0}, .o1=x, .o2=y, .t=tp*C_bond.m[x][y]});
            }
        }
    }
    e->hop = e_hop;
    e->n_hop = e_hop_sz;
    p->hop = p_hop;
    p->n_hop = p_hop_sz;

    // Loop over electronic hoppings, project hopping vector on phonon
    // vibrations and calculate EPC. Relative strengh of EPC is alpha
    double alpha = 0.1;
    VECTOR_INIT(ph_vertex_t, p_vert);
    for (index_t i=0; i<e->n_hop; ++i) {
        rs_hopping_t t = e->hop[i];
        claV3d_t d_vec = claV3d_add(claM3d_vecmat((claV3d_t){{t.R[0],t.R[1],t.R[2]}}, claM3d_map(e->lattice[0])),
            claV3d_sub(claV3d_map(e->positions[t.o2]), claV3d_map(e->positions[t.o1])));
        // the orbital position thing is a no-op here, but it's nevertheless good to have it.
        double dist = claV3d_norm(d_vec);
        claV3d_t d_nrm = d_vec;
        if (fabs(dist) > 1.e-5) {
            d_nrm = claV3d_scale(d_vec, 1./dist);
        } else {
            d_nrm = claV3d_scale(d_vec, 0); // turn off site coupling
        }
        ph_vertex_t p = { .o1=t.o1, .o2=t.o2, .R12={t.R[0],t.R[1],t.R[2]} };
        for (int d=0; d<3; ++d) {
            double g = alpha * d_nrm.v[d] * t.t;
            if (fabs(g) > 1.e-5) {
                 for (int x=0; x<3; ++x) p.RA[x] = 0;
                 p.g = g;
                 p.OA = IDX2(t.o1,d,3);
                 VECTOR_PUSH_BACK(p_vert, p);
                 for (int x=0; x<3; ++x) p.RA[x] = p.R12[x];
                 p.g = -g;
                 p.OA = IDX2(t.o2,d,3);
                 VECTOR_PUSH_BACK(p_vert, p);
            }
        }
    }
    p->vert = (void*)p_vert;
    p->n_vert = p_vert_sz;

    diverge_model_validate(e);
    diverge_model_validate(p);

    diverge_phonons_apply_asr(p);
    p->vert = (void*)diverge_phonons_asr_vertex_n_orb((void*)p->vert, &p->n_vert, 3);

    e->orb_symmetries = calloc(8, sizeof*e->orb_symmetries);
    for (int i=0; i<8; ++i) e->orb_symmetries[i] = 1.0;

    site_descr_t sites[1] = {0};
    sites[0].n_functions = 1;
    sites[0].amplitude[0] = 1.;
    sites[0].function[0] = orb_s;

    sym_op_t curr_symm[1] = {0};
    curr_symm[0].type = 'R';
    curr_symm[0].normal_vector[0] = curr_symm[0].normal_vector[1] = 0.;
    curr_symm[0].normal_vector[2] = 1.;

    int cnt = 0;
    // C4
    for (cnt=0; cnt<4; ++cnt) {
        curr_symm[0].angle = 90*cnt;
        diverge_generate_symm_trafo(1, sites, 1, curr_symm, 1, &(e->rs_symmetries[cnt][0][0]),
                e->orb_symmetries+cnt);
    }
    // mirrors
    curr_symm[0].type = 'M';
    curr_symm[0].normal_vector[2] = 0.;
    curr_symm[0].normal_vector[0] = 0.;
    curr_symm[0].normal_vector[1] = 1.;
    diverge_generate_symm_trafo(1, sites, 1, curr_symm, 1, &(e->rs_symmetries[cnt][0][0]),
            e->orb_symmetries+cnt);
    cnt++;
    curr_symm[0].normal_vector[0] = 1.;
    curr_symm[0].normal_vector[1] = 0.;
    diverge_generate_symm_trafo(1, sites, 1, curr_symm, 1, &(e->rs_symmetries[cnt][0][0]),
            e->orb_symmetries+cnt);
    cnt++;
    curr_symm[0].normal_vector[0] = 1.;
    curr_symm[0].normal_vector[1] = 1.;
    diverge_generate_symm_trafo(1, sites, 1, curr_symm, 1, &(e->rs_symmetries[cnt][0][0]),
            e->orb_symmetries+cnt);
    cnt++;
    curr_symm[0].normal_vector[0] = 1.;
    curr_symm[0].normal_vector[1] = -1.;
    diverge_generate_symm_trafo(1, sites, 1, curr_symm, 1, &(e->rs_symmetries[cnt][0][0]),
            e->orb_symmetries+cnt);
    cnt++;
    e->n_sym = -cnt;
    diverge_phonons_fetch_sym(e, p);
    p->n_sym *= -1;

    diverge_model_internals_common(e);
    diverge_model_internals_common(p);

    // save the vertices
    ph_vertex_t* pv = malloc(sizeof*pv * p->n_vert);
    memcpy(pv, p->vert, sizeof*pv * p->n_vert);
    index_t n_pv = p->n_vert;
    // and sort them
    phsort(pv, n_pv, diverge_omp_num_threads(), 0);

    // This is the call we actaully test
    diverge_phonons_symmetrize_epc(e, p);

    // and now we also sort the symmetrized vertices
    p_vert = (ph_vertex_t*)p->vert;
    p_vert_sz = p_vert_cap = p->n_vert;
    phsort(p_vert, p_vert_sz, diverge_omp_num_threads(), 0);

    // check that all elements are the same
    int n_comp = MIN(n_pv, p_vert_sz);
    for (int i=0; i<n_comp; ++i) {
        int R12_RA_equal = !phsort_cmp(pv[i], p_vert[i]);
        CT_CHECK(R12_RA_equal);
        CT_CHECK(cabs(pv[i].g-p_vert[i].g) < 1.e-10);
    }
    // and that any overhang is zero
    if (n_pv > p_vert_sz) {
        for (int i=n_comp; i<n_pv; ++i) {
            CT_CHECK(cabs(pv[i].g) < 1.e-10);
        }
    }
    else if (p_vert_sz > n_pv) {
        for (int i=n_comp; i<p_vert_sz; ++i) {
            CT_CHECK(cabs(p_vert[i].g) < 1.e-10);
        }
    }

    free(pv);
    diverge_model_free(e);
    diverge_model_free(p);
}

#endif
