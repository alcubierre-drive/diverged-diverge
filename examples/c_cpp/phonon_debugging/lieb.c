#include <diverge.h>
#include <diverge_phonons.h>
#include <diverge_misc.h>
LINALG_IMPLEMENT

#include <tu/phonons.h>

#define EQUAL( a, b ) (fabs((a)-(b)) < DIVERGE_EPS_MESH)
static inline claM3d_t rotation_z( double theta ) {
    return (claM3d_t){{{ cos(theta), sin(theta), 0},
                       {-sin(theta), cos(theta), 0},
                       {          0,          0, 1}}};
}

FILE* vertex_file = NULL;
const char vertex_file_name[] = "vertex_eig.bin";
static void append_vertex_eigs_to_file( diverge_flow_step_vertex_t v );

// TODO
static int use_correct_elph_g = 1;

int main( int argc, char** argv ) {
    diverge_init( &argc, &argv );
    // mpi_loglevel_set(6);

    int nk = 18, nkf = 5;
    double t1 = 1.0, t2 = 0.1;
    double U = 0.00, V1 = 0.00, V2 = 0.00;

    diverge_model_t* m = diverge_model_init();
    m->nk[0] = m->nk[1] = nk;
    m->nkf[0] = m->nkf[1] = nkf;
    m->lattice[0][0] = m->lattice[1][1] = m->lattice[2][2] = 1.0;
    claV3d_t a1 = claV3d_map(m->lattice[0]),
             a2 = claV3d_map(m->lattice[1]);
    claV3d_t p1 = {{0,0,0}},
             p2 = {{0.5,0,0}},
             p3 = {{0,0.5,0}};
    claV3d_rmap(p1,m->positions[0]);
    claV3d_rmap(p2,m->positions[1]);
    claV3d_rmap(p3,m->positions[2]);
    m->n_orb = 3;
    m->n_spin = m->SU2 = 1;
    m->n_ibz_path = 4;
    m->ibz_path[1][0] = m->ibz_path[2][0] = m->ibz_path[2][1] = 0.5;

    double* dist = diverge_model_unique_distances(m, 8, 2);
    VECTOR_INIT( rs_hopping_t, hop );
    VECTOR_INIT( rs_vertex_t, vert );
    for (int i=-3; i<=3; ++i)
    for (int j=-3; j<=3; ++j)
    for (int o1=0; o1<m->n_orb; ++o1)
    for (int o2=0; o2<m->n_orb; ++o2) {
        claV3d_t P = claV3d_sub(claV3d_map(m->positions[o2]), claV3d_map(m->positions[o1]));
        claV3d_t R = claV3d_add(P, claV3d_add(claV3d_scale(a1,i), claV3d_scale(a2,j)));
        double d = claV3d_norm2(R);
        if (fabs(d-POW2(dist[0])) < DIVERGE_EPS_MESH) {
            VECTOR_PUSH_BACK( vert, (rs_vertex_t){.chan='C', .R={i,j,0}, .o1=o1, .o2=o2, .V=U} );
        }
        if (fabs(d-POW2(dist[1])) < DIVERGE_EPS_MESH) {
            VECTOR_PUSH_BACK( hop, (rs_hopping_t){.R={i,j,0}, .o1=o1, .o2=o2, .t=t1} );
            VECTOR_PUSH_BACK( vert, (rs_vertex_t){.chan='D', .R={i,j,0}, .o1=o1, .o2=o2, .V=V1} );
        }
        if (fabs(d-POW2(dist[2])) < DIVERGE_EPS_MESH) {
            VECTOR_PUSH_BACK( hop, (rs_hopping_t){.R={i,j,0}, .o1=o1, .o2=o2, .t=t2} );
            VECTOR_PUSH_BACK( vert, (rs_vertex_t){.chan='D', .R={i,j,0}, .o1=o1, .o2=o2, .V=V2} );
        }
    }
    m->hop = hop;
    m->n_hop = hop_sz;
    m->vert = vert;
    m->n_vert = vert_sz;

    diverge_model_internals_common(m);
    diverge_model_output_conf_t oc = diverge_model_output_conf_defaults;
    oc.npath = -1;
    oc.kf_ibz_path = 1;
    oc.kc = 1;
    oc.E = oc.U = 1;
    diverge_model_to_file_finegrained(m, "el.dvg", &oc);

    diverge_model_t* ph = diverge_model_init();
    memcpy(ph->nk, m->nk, sizeof(index_t)*2);
    memcpy(ph->nkf, m->nkf, sizeof(index_t)*2);
    memcpy(ph->lattice, m->lattice, sizeof ph->lattice);
    ph->n_spin = ph->SU2 = 1;
    memcpy(ph->ibz_path, m->ibz_path, sizeof ph->ibz_path);
    ph->n_ibz_path = m->n_ibz_path;

    for (int o=0; o<m->n_orb; ++o)
    for (int r=0; r<3; ++r)
        claV3d_rmap(claV3d_map(m->positions[o]), ph->positions[IDX2(o,r,3)]);

    const double C1x = -100, C1y = -60, C1z = -40;
    const double C2x = -50, C2y = -30, C2z = -10;

    const claV3d_t reference_bonds[] = {
        claV3d_sub(p2,p1),
        claV3d_sub(p3,p2),
    };
    const claM3d_t reference_forces[] = {
        {{{   C1x,       0,       0},
          {     0,     C1y,       0},
          {     0,       0,     C1z}}},
        {{{C2x/2.+C2y/2., C2x/2.-C2y/2.,    0},
          {C2x/2.-C2y/2., C2x/2.+C2y/2.,    0},
          {            0,             0, C2z}}},
    };
    const int n_reference = 2;

    VECTOR_INIT( rs_hopping_t, ph_hop );

    for (int i=-3; i<=3; ++i)
    for (int j=-3; j<=3; ++j)
    for (int o1=0; o1<m->n_orb; ++o1)
    for (int o2=0; o2<m->n_orb; ++o2) {
        claV3d_t D = claV3d_add(claV3d_sub(claV3d_map(m->positions[o2]),claV3d_map(m->positions[o1])),
                claV3d_add(claV3d_scale(a1,i), claV3d_scale(a2,j)));
        int rb = -1;
        for (rb=0; rb<n_reference; ++rb)
            if (EQUAL(claV3d_norm2(D),claV3d_norm2(reference_bonds[rb]))) break;
        if (rb != -1 && rb != n_reference) {
            // found a bond length
            claV3d_t D_x_rb = claV3d_cross(D, reference_bonds[rb]);
            double theta = atan2(D_x_rb.v[2], claV3d_dot(D, reference_bonds[rb]));
            claM3d_t D_forces = claM3d_matmul(claM3d_matmul(rotation_z(-theta),
                        reference_forces[rb]), rotation_z(theta));
            for (int r1=0; r1<3; ++r1)
            for (int r2=0; r2<3; ++r2) {
                double D_force = D_forces.m[r1][r2];
                if (!EQUAL(D_force, 0.0))
                    VECTOR_PUSH_BACK( ph_hop, (rs_hopping_t){ .R={i,j,0},
                            .o1=IDX2(o1,r1,3), .o2=IDX2(o2,r2,3), .t=D_force } );
            }
        }
    }

    ph_hop = diverge_phonons_asr( ph_hop, &ph_hop_sz );
    ph_hop_cap = ph_hop_sz;

    ph->n_orb = m->n_orb * 3;
    ph->hop = ph_hop;
    ph->n_hop = ph_hop_sz;

    diverge_model_internals_common(ph);
    diverge_phonons_apply_sgnsqrt(ph);
    diverge_model_to_file_finegrained(ph, "ph.dvg", &oc);

    double alpha = 1.0,
           beta = 1.0;
    VECTOR_INIT( ph_vertex_t, pv );
    for (index_t i=0; i<m->n_hop; ++i) {
        rs_hopping_t t = m->hop[i];
        claV3d_t P = claV3d_sub(claV3d_map(m->positions[t.o2]), claV3d_map(m->positions[t.o1]));
        claV3d_t R = claV3d_add(P,claV3d_add(claV3d_scale(a1,t.R[0]),claV3d_scale(a2,t.R[1])));
        claV3d_t eR = claV3d_scale(R, 1./claV3d_norm(R));
        ph_vertex_t p = { .o1=t.o1, .o2=t.o2, .R12={t.R[0],t.R[1],t.R[2]} };
        if (EQUAL(claV3d_norm(R),dist[1]) || EQUAL(claV3d_norm(R),dist[2])) {
            double aa = EQUAL(claV3d_norm(R),dist[1]) ? alpha : beta;
            for (int d=0; d<3; ++d) {
                 double g = aa * eR.v[d];
                 if (!EQUAL(g, 0.0)) {
                     for (int x=0; x<3; ++x) p.RA[x] = p.R12[x];
                     p.g = g;
                     p.OA = IDX2(t.o2,d,3);
                     VECTOR_PUSH_BACK( pv, p );
                     if (use_correct_elph_g) {
                        for (int x=0; x<3; ++x) p.RA[x] = 0;
                        p.g = -g;
                        p.OA = IDX2(t.o1,d,3);
                        VECTOR_PUSH_BACK( pv, p );
                    }
                }
            }
        }
    }

    diverge_model_t* pph = diverge_model_internals_tu_phonons(m, 1.01, ph->hop,
            ph->n_hop, pv, pv_sz);
    diverge_model_to_file_finegrained(m, "el_int.dvg", &oc);
    free(pv);
    memcpy(pph->positions, ph->positions, sizeof(ph->positions));
    diverge_model_free(ph);
    ph = pph;
    diverge_model_to_file_finegrained(ph, "elph.dvg", &oc);

    diverge_flow_step_t* s = diverge_flow_step_init_any(m, "PCD");
    {
        FILE* gfile = fopen("g.bin", "wb");
        diverge_flow_step_vertex_t v = diverge_flow_step_vertex( s, 'D' );
        fwrite( diverge_phonons_get_g(m), sizeof(complex128_t), kdim(m->nk)*POW2(v.n_spin)*v.n_orbff * ph->n_orb, gfile );
        fclose( gfile );
    }
    diverge_euler_t eu = diverge_euler_defaults;
    int i=0;
    do {
        diverge_flow_step_euler(s, eu.Lambda, eu.dLambda );
        if (++i == 1) break;
    } while (diverge_euler_next(&eu, 0));
    // append_vertex_eigs_to_file( diverge_flow_step_vertex(s, 'p') );
    // append_vertex_eigs_to_file( diverge_flow_step_vertex(s, 'm') );
    // append_vertex_eigs_to_file( diverge_flow_step_vertex(s, 'h') );
    append_vertex_eigs_to_file( diverge_flow_step_vertex(s, 'P') );
    append_vertex_eigs_to_file( diverge_flow_step_vertex(s, 'C') );
    append_vertex_eigs_to_file( diverge_flow_step_vertex(s, 'D') );
    if (vertex_file) fclose(vertex_file);

    free(dist);

    diverge_postprocess_conf_t pc = diverge_postprocess_conf_defaults;
    pc.tu_storing_relative = 1;
    pc.tu_storing_threshold = 0.01;
    pc.tu_which_solver_mode = 'B';
    diverge_postprocess_and_write_fg( s, "elph_post.dvg", &pc );

    diverge_flow_step_free(s);
    diverge_model_free(m);
    diverge_finalize();
}

static void append_vertex_eigs_to_file( diverge_flow_step_vertex_t v ) {
    if (!vertex_file) {
        vertex_file = fopen(vertex_file_name, "wb");
        mpi_usr_printf("opening %s for vertices\n", vertex_file_name);
    }
    index_t sz = v.n_orbff, num = v.q_1 - v.q_0;
    double* vals = calloc(num*sz, sizeof*vals);
    complex128_t* vecs = calloc(num*POW2(sz), sizeof*vecs);
    complex128_t* tmp = calloc(num*POW2(sz), sizeof*tmp);
    for (index_t q=0; q<num; ++q) {
        single_svd( v.ary + POW2(sz)*q, tmp + POW2(sz)*q, vecs + POW2(sz)*q, vals + sz*q, sz );
        single_eigen_sort( tmp + POW2(sz)*q, vals + sz*q, sz, 'M' );
    }
    free( vecs );
    free( tmp );
    fwrite( vals, sizeof*vals, sz*num, vertex_file );
    // fwrite( v.ary, sizeof*v.ary, sz*sz*num, vertex_file );
    free( vals );
}
