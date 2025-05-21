#include <diverge.h>
#include <diverge_misc.h>
LINALG_IMPLEMENT

static const double t = 1.0, phi = M_PI/4.0;
#define ZERO( a ) (fabs(a) < 1e-4)
static diverge_model_t* square_dirac_model( int nk, int nkf, double tp ) {
    diverge_model_t* m = diverge_model_init();
    m->lattice[0][0] = m->lattice[1][1] = m->lattice[2][2] = 1.0;
    m->positions[1][0] = m->positions[1][1] = 0.5;
    m->n_orb = 2;
    m->n_spin = m->SU2 = 1;

    m->hop = calloc(128, sizeof*m->hop);

    for (int i=-2; i<=2; ++i)
    for (int j=-2; j<=2; ++j)
    for (int o=0; o<2; ++o)
    for (int p=0; p<2; ++p) {
        claV3d_t distvec = claV3d_add((claV3d_t){{i,j,0}},claV3d_sub(
                    claV3d_map(m->positions[p]),claV3d_map(m->positions[o])));
        double dist = claV3d_norm(distvec);

        if ZERO(dist-1.) {
            m->hop[m->n_hop++] = (rs_hopping_t){ .R={i,j}, .o1=o, .o2=p, .t=o==0?t:-t };
        }

        if ZERO(dist-1./sqrt(2.)) {
            complex128_t tc = distvec.v[0] + I*distvec.v[1];
            index_t lphi = lroundl(0.5*(fmod( carg(tc) + 2.*M_PI, 2.*M_PI ) / (0.25*M_PI) - 1.0));
            double phase = 0.0;
            if (((lphi == 0) || (lphi == 2)) && o)
                phase = 0.5 * M_PI + phi;
            else if (((lphi == 0) || (lphi == 2)) && !o)
                phase = -0.5 * M_PI - phi;
            if (((lphi == 1) || (lphi == 3)) && o)
                phase = 0.0 * M_PI + phi;
            else if (((lphi == 1) || (lphi == 3)) && !o)
                phase = -0.0 * M_PI - phi;
            m->hop[m->n_hop++] = (rs_hopping_t){ .R={i,j}, .o1=o, .o2=p, .t=tp*cexp(I*phase) };
        }
    }

    m->ibz_path[1][0] = m->ibz_path[2][0] = m->ibz_path[2][1] = 0.5;
    m->n_ibz_path = 4;

    m->nk[0] = m->nk[1] = nk;
    m->nkf[0] = m->nkf[1] = nkf;
    return m;
}

int main( int argc, char** argv ) {
    diverge_init( &argc, &argv );

    diverge_model_t* m = square_dirac_model( 40, 15, 0.5 );

    m->n_spin = 2;
    m->SU2 = 0;
    m->hop = realloc(m->hop, sizeof*m->hop * (m->n_hop * 2 + 8));
    memcpy( m->hop + m->n_hop, m->hop, sizeof*m->hop * m->n_hop );
    for (int h=0; h<m->n_hop; ++h) {
        rs_hopping_t* H = m->hop+h+m->n_hop;
        H->s1 = H->s2 = 1;
    }
    m->n_hop *= 2;

    double M = 0.2;
    for (int o=0; o<2; ++o)
    for (int s=0; s<2; ++s)
        m->hop[m->n_hop++] = (rs_hopping_t){ .o1=o, .o2=o, .s1=s, .s2=s,
            .t=(o?1:-1) * (s?1:-1) * M };

    // double mass = 0.05;
    // m->hop[m->n_hop++] = (rs_hopping_t){ .o1=0, .o2=0, .t= mass };
    // m->hop[m->n_hop++] = (rs_hopping_t){ .o1=1, .o2=1, .t=-mass };

    diverge_model_internals_common( m );

    diverge_model_output_conf_t oc = diverge_model_output_conf_defaults;
    oc.npath = -1;
    oc.kf_ibz_path = 1;
    oc.E = 1;
    oc.U = 1;
    oc.kf = 1;
    index_t nk[] = { -m->nk[0]*m->nkf[0], m->nk[1]*m->nkf[1], 1 };
    index_t which_bands[] = {0,1};
    m->data = diverge_fukui_non_abelian( m, NULL, -1, nk, which_bands, 2 );
    m->nbytes_data = sizeof(double) * kdimtot(m->nk, m->nkf) * 3;
    diverge_model_to_file_finegrained( m, __FILE__ "_mod.dvg", &oc );

    diverge_model_free(m);

    diverge_finalize();
}
