/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "vertex_helper.h"
#include "loop.h"

static inline index_t _k1m2( index_t k1, index_t k2, index_t nk_flat ) {
    return ((k1 / nk_flat - k2 / nk_flat + nk_flat) % nk_flat)*nk_flat +
        (k1 % nk_flat - k2 % nk_flat + nk_flat) % nk_flat;
}

static inline index_t _k1p2( index_t k1, index_t k2, index_t nk_flat ) {
    return ((k1 / nk_flat + k2 / nk_flat) % nk_flat)*nk_flat +
        (k1 % nk_flat + k2 % nk_flat) % nk_flat;
}

static inline void qkkp_init( qkkp_t *q, index_t p1, index_t p2, index_t p3,
        char chan, const index_t* patch, index_t nk_flat ) {
    q->p1 = p1;
    q->p2 = p2;
    q->p3 = p3;
    index_t k1 = patch[p1],
            k2 = patch[p2],
            k3 = patch[p3];
    q->chan = chan;
    switch (chan) {
        case 'P': default_case:
            q->q = _k1p2(k1, k2, nk_flat);
            q->k = k1;
            q->kp = k3;
            break;
        case 'C':
            q->q = _k1m2(k3, k2, nk_flat);
            q->k = k1;
            q->kp = k3;
            break;
        case 'D':
            q->q = _k1m2(k1, k3, nk_flat);
            q->k = k1;
            q->kp = _k1p2(_k1m2(k2, k3, nk_flat), k1, nk_flat);
            break;
        default: goto default_case;
            break;
    }
}

void vhelper_vertex_in_q_channel( const npatch_vertex_t* vert, const npatch_loop_t* loop,
        char chan, bool dV, complex128_t** pvert, qkkp_t** pvindex,
        index_t** pqcounts, index_t* pnq ) {

    index_t np = loop->np;
    index_t nb = loop->nb;
    index_t nk_flat = loop->nk_flat[1];

    const index_t* patches = loop->p_idx;
    const complex128_t* VV = dV ? vert->dV : vert->V;

    complex128_t* vertex = (complex128_t*)malloc(sizeof(complex128_t)*vert->size);
    qkkp_t* vindex = (qkkp_t*)malloc(sizeof(qkkp_t)*np*np*np);

    #pragma omp parallel for schedule(static) collapse(3)
    for (index_t p1=0; p1<np; ++p1)
    for (index_t p2=0; p2<np; ++p2)
    for (index_t p3=0; p3<np; ++p3) {
        index_t k_shuf = p1*np*np + p2*np + p3;
        qkkp_init( vindex + k_shuf, p1, p2, p3, chan, patches, nk_flat );
        for (index_t o1=0; o1<nb; ++o1)
        for (index_t o2=0; o2<nb; ++o2)
        for (index_t o3=0; o3<nb; ++o3)
        for (index_t o4=0; o4<nb; ++o4) {
            index_t o_shuf = o1*nb*nb*nb + o2*nb*nb + o3*nb + o4;
            index_t b1, b2, b3, b4;
            switch (chan) {
                case 'P': default_case:
                    b1 = o1, b2 = o2, b3 = o3, b4 = o4;
                    break;
                case 'C':
                    b1 = o4, b2 = o1, b3 = o2, b4 = o3;
                    break;
                case 'D':
                    b1 = o3, b2 = o1, b3 = o2, b4 = o4;
                    break;
                default: goto default_case;
                    break;
            }
            index_t b_shuf = b1 * nb*nb*nb + b2*nb*nb + b3*nb + b4;
            vertex[k_shuf*nb*nb*nb*nb + b_shuf] = VV[k_shuf*nb*nb*nb*nb + o_shuf];
        }
    }

    *pvert = vertex;
    *pvindex = vindex;
    vhelper_vertex_sort_in_q( vertex, vindex, np, nb, pqcounts, pnq );
}


index_t* vhelper_qkkp_to_ary( const qkkp_t* q, index_t nq ) {
    index_t* r = (index_t*)malloc(sizeof(index_t)*nq*3);
    index_t i=0;
    for (index_t qq=0; qq<nq; ++qq) {
        r[i++] = q[qq].q;
        r[i++] = q[qq].k;
        r[i++] = q[qq].kp;
    }
    return r;
}
