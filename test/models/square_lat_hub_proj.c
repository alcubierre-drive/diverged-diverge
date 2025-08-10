#ifndef DIVERGE_SKIP_TESTS

#include "all.h"
#include "../../src/diverge.h"

static greensfunc_op_t hubbard_proj_gf(const diverge_model_t* model, complex128_t Lambda, gf_complex_t* buf);

diverge_model_t* gen_square_lat_hub_proj( index_t nkx, index_t nky, double scale_tb ) {
    diverge_model_t* mt = KWARGS(gen_square_lat_hub, .nkx=nkx, .nky=nky);

    diverge_model_t* m = diverge_model_init();
    m->nk[0] = nkx, m->nk[1] = nky;
    m->nkf[0] = m->nkf[1] = 1;
    m->n_orb = 2;
    m->SU2 = 1;
    memcpy( m->lattice, mt->lattice, sizeof(double)*9 );

    m->hop = calloc(2*mt->n_hop, sizeof(rs_hopping_t));
    for (index_t i=0; i<mt->n_hop; ++i) {
        memcpy( m->hop+i, mt->hop+i, sizeof(rs_hopping_t) );
    }
    for (index_t i=0; i<mt->n_hop; ++i) {
        memcpy( m->hop+mt->n_hop+i, mt->hop+i, sizeof(rs_hopping_t) );
        m->hop[mt->n_hop+i].o1 = m->hop[mt->n_hop+i].o2 = 1;
        m->hop[mt->n_hop+i].t *= scale_tb;
    }
    m->n_hop = mt->n_hop * 2;

    memcpy( m->ibz_path, mt->ibz_path, sizeof(double)*12 );
    m->n_ibz_path = 4;

    double U = 3.0, J = 1.0;
    m->vert = (rs_vertex_t*)calloc(128, sizeof(rs_vertex_t));
    for (index_t o1=0; o1<2; ++o1)
    for (index_t o2=0; o2<2; ++o2) {
        if (o1 == o2) {
            m->vert[m->n_vert++] = (rs_vertex_t){'D', {0,0,0}, o1, o2, -1,0,0,0, U};
        } else {
            m->vert[m->n_vert++] = (rs_vertex_t){'D', {0,0,0}, o1, o2, -1,0,0,0, U-2*J};
            m->vert[m->n_vert++] = (rs_vertex_t){'C', {0,0,0}, o1, o2, -1,0,0,0, J};
            m->vert[m->n_vert++] = (rs_vertex_t){'P', {0,0,0}, o1, o2, -1,0,0,0, J};
        }
    }

    m->gproj = &hubbard_proj_gf;
    diverge_model_free(mt);
    diverge_model_internals_common(m);
    return m;
}

static greensfunc_op_t hubbard_proj_gf(const diverge_model_t* model, complex128_t Lambda, gf_complex_t* buf) {
    const index_t nkf = model->nk[0]*model->nk[1]*model->nk[2]
                      * model->nkf[0]*model->nkf[1]*model->nkf[2];
    const complex128_t* U = diverge_model_internals_get_U(model);
    const double* E = diverge_model_internals_get_E(model);

    for (int sign=0; sign<2; ++sign) {
        complex128_t L = sign ? Lambda : conj(Lambda);
        #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
        for (index_t k=0; k<nkf; ++k)
        for (index_t o=0; o<2; ++o)
        for (index_t p=0; p<2; ++p) {
            complex128_t tmp = 0.0;
            if (o==0 && p==0)
                for (index_t b=0; b<2; ++b)
                    tmp += U[IDX3(k,b,o,2,2)] * conj(U[IDX3(k,b,p,2,2)]) / (L - E[IDX2(k,b,2)]);
            buf[IDX4(sign, k, o, p, nkf, 2, 2)] = tmp;
        }
    }
    return greensfunc_op_cpu;
}

#endif // DIVERGE_SKIP_TESTS
