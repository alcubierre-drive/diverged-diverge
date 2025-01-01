/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "vertex_helper.h"
#include "loop.h"
#include "../misc/mpi_functions.h"
#include "../misc/mergesort.h"

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

typedef struct { complex128_t first; qkkp_t second; } v_element_t;
static inline int cmp_v_element_second( const v_element_t* pA, const v_element_t* pB ) {
    if (pA->second.q < pB->second.q)        return -1;
    else if (pA->second.q > pB->second.q)   return 1;
    else {
        if (pA->second.k < pB->second.k)        return -1;
        else if (pA->second.k > pB->second.k)   return 1;
        else {
            if (pA->second.kp < pB->second.kp)      return -1;
            else if (pA->second.kp > pB->second.kp) return 1;
            else                                    return 0;
        }
    }
}
#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) cmp_v_element_second( &(x), &(y) )
MERGESORT_IMPL( mergesort_velem, v_element_t )

static void vhelper_vertex_sort_in_q( complex128_t* vertex, qkkp_t* vindex, index_t np,
        index_t nb, index_t** pqcounts, index_t* pnq ) {

    index_t np3 = np*np*np;
    index_t no4 = nb*nb*nb*nb;
    index_t size = np3 * no4;

    v_element_t* vsort = calloc(size, sizeof*vsort);
    #pragma omp parallel for schedule(static) collapse(2) num_threads(diverge_omp_num_threads())
    for (index_t p3=0; p3<np3; ++p3) for (index_t o4=0; o4<no4; ++o4)
        vsort[p3*no4+o4] = (v_element_t){ vertex[p3*no4+o4], vindex[p3] };

    mergesort_velem( vsort, size, diverge_omp_num_threads(), 0 );

    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t p3=0; p3<np3; ++p3) {
        for (index_t o4=0; o4<no4; ++o4) {
            vertex[p3*no4+o4] = vsort[p3*no4+o4].first;
        }
        vindex[p3] = vsort[p3*no4].second;
    }
    free( vsort );

    index_t* qcounts = (index_t*)calloc(np*np, sizeof(index_t));
    qcounts[0] = 1;
    index_t nq = 0;
    for (index_t p3=1; p3<np3; ++p3) {
        if (vindex[p3-1].q != vindex[p3].q)
            nq++;
        qcounts[nq]++;
    }
    nq++;

    *pqcounts = qcounts;
    *pnq = nq;
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

    #pragma omp parallel for schedule(static) collapse(3) num_threads(diverge_omp_num_threads())
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

#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) (x.k < y.k ? -1 : x.k > y.k ? 1 : 0)
MERGESORT_IMPL( mergesort_qkkp_k, qkkp_t )

#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) (x.kp < y.kp ? -1 : x.kp > y.kp ? 1 : 0)
MERGESORT_IMPL( mergesort_qkkp_kp, qkkp_t )

#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) (x < y ? -1 : x > y ? 1 : 0)
MERGESORT_IMPL( mergesort_index, index_t )

static inline index_t* vhelper_qkkp_to_ary( const qkkp_t* q, index_t nq ) {
    index_t* r = (index_t*)malloc(sizeof(index_t)*nq*3);
    index_t i=0;
    for (index_t qq=0; qq<nq; ++qq) {
        r[i++] = q[qq].q;
        r[i++] = q[qq].k;
        r[i++] = q[qq].kp;
    }
    return r;
}

static inline index_t unique_k( qkkp_t* arr, index_t size ) {
    if (size == 0) return 0;
    index_t j = 0;
    for (index_t i=1; i<size; ++i)
        if (arr[i].k != arr[j].k)
            arr[++j] = arr[i];
    return j+1;
}

static inline index_t unique_kp( qkkp_t* arr, index_t size ) {
    if (size == 0) return 0;
    index_t j = 0;
    for (index_t i=1; i<size; ++i)
        if (arr[i].kp != arr[j].kp)
            arr[++j] = arr[i];
    return j+1;
}

static inline index_t unique( index_t* arr, index_t size ) {
    if (size == 0) return 0;
    index_t j = 0;
    for (index_t i=1; i<size; ++i)
        if (arr[i] != arr[j])
            arr[++j] = arr[i];
    return j+1;
}

static inline index_t find_bin_irreg( const index_t* vals, index_t n_vals, index_t test ) {
    if (test >= vals[n_vals-1]) return n_vals-1;
    if (test <= vals[0]) return 0;
    index_t low = 0,
            high = n_vals-1;
    while (low <= high) {
        index_t mid = low + (high - low)/2;
        if (vals[mid] <= test)
            low = mid + 1;
        else
            high = mid -1;
    }
    return low - 1;
}

npatch_qvert_mat_t* vhelper_q_vertex_matrix_from_vertex( const complex128_t* q_vertex,
        const qkkp_t* q_vindex, const index_t* q_counts, index_t nq,
        index_t nb ) {

    index_t* q_displs = calloc(nq, sizeof(index_t));
    for (index_t q=0; q<nq; ++q) if (q>0) q_displs[q] = q_displs[q-1] + q_counts[q-1];

    npatch_qvert_mat_t* result = malloc(sizeof(npatch_qvert_mat_t)*nq);

    for (index_t q=0; q<nq; ++q) {
        VECTOR_INIT_CAP( index_t, k_kp_vector, q_counts[q] );

        qkkp_t* qvi = calloc(q_counts[q], sizeof*qvi);
        memcpy( qvi, q_vindex + q_displs[q], sizeof(qkkp_t)*q_counts[q] );
        mergesort_qkkp_k( qvi, q_counts[q], diverge_omp_num_threads(), 0 );
        index_t qc = unique_k( qvi, q_counts[q] );
        for (index_t i=0; i<qc; ++i)
            VECTOR_PUSH_BACK( k_kp_vector, qvi[i].k );

        memcpy( qvi, q_vindex + q_displs[q], sizeof(qkkp_t)*q_counts[q] );
        mergesort_qkkp_kp( qvi, q_counts[q], diverge_omp_num_threads(), 0 );
        qc = unique_kp( qvi, q_counts[q] );
        for (index_t i=0; i<qc; ++i)
            VECTOR_PUSH_BACK( k_kp_vector, qvi[i].kp );
        free( qvi );

        mergesort_index( k_kp_vector, k_kp_vector_sz, diverge_omp_num_threads(), 0 );
        k_kp_vector_sz = unique( k_kp_vector, k_kp_vector_sz );
        VECTOR_SHRINK_TO_FIT( k_kp_vector );

        index_t nkkp = k_kp_vector_sz;
        index_t matrix_size = POW2(nkkp) * POW4(nb);

        complex128_t* matrix = calloc(matrix_size, sizeof*matrix);
        for (index_t qq=0; qq<q_counts[q]; ++qq) {
            const qkkp_t* qvv = q_vindex + q_displs[q] + qq;
            index_t ki = find_bin_irreg( k_kp_vector, k_kp_vector_sz, qvv->k );
            index_t kpi = find_bin_irreg( k_kp_vector, k_kp_vector_sz, qvv->kp );
            for (index_t o12=0; o12<nb*nb; ++o12)
            for (index_t o34=0; o34<nb*nb; ++o34)
                matrix[(ki*nb*nb+o12) * (nkkp*nb*nb) + kpi*nb*nb+o34] =
                    q_vertex[(q_displs[q]+qq)*nb*nb*nb*nb+o12*nb*nb+o34];
        }


        result[q].data = matrix; // no call to free(matrix);
        result[q].k_kp_descr = k_kp_vector; // no call to VECTOR_FREE( k_kp_vector );
        result[q].n_k_kp = nkkp;
        result[q].nb = nb;
        result[q].q = q_vindex[q_displs[q]].q;
        result[q].eigen = false;
        result[q].values = NULL;
        result[q].nv = -1;
    }

    free(q_displs);

    return result;
}

complex128_t* vhelper_q_vertex_matrices_to_buffer( npatch_qvert_mat_t* qvm, index_t nq, index_t Nmax ) {
    complex128_t* result = calloc(Nmax*Nmax*nq, sizeof*result);
    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t q=0; q<nq; ++q) {
        index_t N_orig = qvm[q].nb * qvm[q].nb * qvm[q].n_k_kp;
        for (index_t i=0; i<MIN(N_orig,Nmax); ++i)
        for (index_t j=0; j<MIN(N_orig,Nmax); ++j)
            result[IDX3(q,i,j,Nmax,Nmax)] = qvm[q].data[IDX2(i,j,N_orig)];
    }
    return result;
}

static inline double cabs2( const complex128_t* in ) {
    const double* din = (const double*)in;
    return POW2(din[0]) + POW2(din[1]);
}

static inline double q_vertex_norm( const complex128_t* data, index_t matrix_size ) {
    double norm = 0;
    for (index_t i=0; i<POW2(matrix_size); ++i) norm += cabs2(&data[i]);
    return sqrt(norm) / (double)matrix_size;
}

typedef struct {
    double norm;
    index_t idx;
} norm_t;

#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) ( (x.norm > y.norm) ? -1 : \
                                 (x.norm < y.norm) ? 1 : 0 )
MERGESORT_IMPL( mergesort_norm, norm_t )

void vhelper_q_vertex_sort_max( npatch_qvert_mat_t* qvm, index_t* pnq, double max_rel ) {
    norm_t* norms = calloc(*pnq, sizeof*norms);
    #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
    for (index_t q=0; q<*pnq; ++q)
        norms[q] = (norm_t){ q_vertex_norm( qvm[q].data, qvm[q].n_k_kp*qvm[q].nb*qvm[q].nb ), q };

    mergesort_norm( norms, *pnq, diverge_omp_num_threads(), 0 );

    index_t qmax = 0;
    for (index_t q=0; q<*pnq; ++q) if (norms[q].norm >= norms[0].norm*max_rel) qmax++;

    npatch_qvert_mat_t* buf = calloc(*pnq, sizeof*buf);
    for (index_t q=0; q<*pnq; ++q) buf[q] = qvm[norms[q].idx];
    for (index_t q=0; q<*pnq; ++q) qvm[q] = buf[q];

    *pnq = qmax;
    free( buf );
    free( norms );
}
