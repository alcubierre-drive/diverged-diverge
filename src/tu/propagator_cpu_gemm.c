/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

// tu_fft_algorithm_gemm: TU loop on coarse mesh as a GEMM.
//
// For every task (mo1,mo3,s1..4) the greedy/split algorithms compute
//      L(r) = sum_F [ X1_c1(sR + D) Y1_c2(R) + X2_c1(sR + D) Y2_c2(R) ],
// with
//      R = r + nk·F (fine split of the real-space mesh),
//      D = R3 - R1,
//      s = ±1 (pp/ph),
// which is then followed by an FFT of the coarse r. X/Y are the real-space
// Green's functions with c1 = (o1,s1,o3,s3), c2 = (ob1,s2,ob3,s4) (pp) or
// (ob3,s2,ob1,s4) (ph). This is a matrix product over the combined index K =
// (term,F) for fixed r (where term denotes the summand):
//      C_r[(c1,D), c2] = sum_K A_r[(c1,D), K] B_r[K, c2],
// with A_r[(c1,D), (term,F)] = X_term[c1](s(r + nk F) + D) and
// B_r[(term,F), c2] = Y_term[c2](r + nk F). The shift sR + D splits per axis
// into a coarse point r' = (sr + D) mod nk and a carry c = floor((sr+D)/nk),
// so sR + D = r' + nk·((c + sF) mod nkf): with the Green's functions stored as
// [c][r][F] each row of A_r is a permuted copy of a contiguous F-block.
// All rows (c1,D) and columns c2 needed by the tasks are computed (no symmetry
// or hermiticity assumption), then each C(·)[row,col] is Fourier transformed
// over r and the my_nk entries of each task are copied out.
//
#include "propagator_details.h"
#include "../misc/gemm.h"
#include <omp.h>

// row block of A_r handled by a single GEMM call (bounds the per-thread buffer)
#ifndef TU_GEMM_LOOP_ABUF_BYTES
#define TU_GEMM_LOOP_ABUF_BYTES (4l<<20)
#endif

#ifndef TU_GEMM_LOOP_FFTW_TIMELIMIT
#define TU_GEMM_LOOP_FFTW_TIMELIMIT 5.0
#endif

#ifndef TU_GEMM_LOOP_FFTW_FLAGS
#define TU_GEMM_LOOP_FFTW_FLAGS FFTW_ESTIMATE
#endif

static inline index_t floordiv( index_t a, index_t b ) {
    index_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

static inline index_t posmod( index_t a, index_t b ) {
    index_t m = a % b;
    return m < 0 ? m + b : m;
}

typedef struct {
    index_t mo1, mo3;
    index_t o1, o3, ob1, ob3;
    index_t d; // index into the unique D list
} gemm_task_t;

// task decomposition, chunking and FFT plans of one variant (pp or ph)
typedef struct {
    bool valid;
    // key
    bool reduced;
    index_t ntasks, taskoff;
    double mem_GiB;

    gemm_task_t* tasks;
    index_t* dvec; // unique D (3 per entry)
    index_t nd;
    index_t *rowmap, *colmap; // (c1 + nb2*d) -> row, c2 -> col (-1: unused)
    index_t *row_c1, *row_d, *col_c2;
    index_t nrow, ncol;
    index_t chunk, rblk;
    fftw_plan plan_full, plan_last; // S = chunk*ncol and the remainder chunk
} gemm_variant_t;

struct tu_gemm_loop_t {
    gemm_variant_t v[2]; // [pp]
    complex128_t* Gt; // [term][c][r][F], only if !t->gemm_gf_layout
    complex128_t *Cbuf, *Abuf, *Bbuf;
    index_t Cbuf_sz, Abuf_sz, Bbuf_sz;
    index_t* Fmap;
    int nthr;
    double t_plan; // setup time
};

static void gemm_variant_clear( gemm_variant_t* v ) {
    if (v->plan_full) fftw_destroy_plan( v->plan_full );
    if (v->plan_last) fftw_destroy_plan( v->plan_last );
    free( v->tasks );
    free( v->dvec );
    free( v->rowmap );
    free( v->colmap );
    free( v->row_c1 );
    free( v->row_d );
    free( v->col_c2 );
    memset( v, 0, sizeof(*v) );
}

void tu_gemm_loop_free( tu_gemm_loop_t* g ) {
    if (!g) return;
    for (int i=0; i<2; ++i) gemm_variant_clear( g->v+i );
    fftw_free( g->Gt );
    fftw_free( g->Cbuf );
    fftw_free( g->Abuf );
    fftw_free( g->Bbuf );
    free( g->Fmap );
    free( g );
}

static void gemm_buffer_reserve( complex128_t** buf, index_t* cap, index_t sz ) {
    if (*cap >= sz) return;
    fftw_free( *buf );
    *buf = (complex128_t*)fftw_alloc_complex( sz );
    *cap = sz;
}

static void gemm_variant_init( const tu_prop_t* t, tu_gemm_loop_t* g, bool pp,
        index_t ntasks, index_t taskoff, bool reduced, double mem_GiB ) {
    const double tick = diverge_mpi_wtime();
    gemm_variant_t* v = g->v + (pp ? 1 : 0);
    gemm_variant_clear( v );
    v->reduced = reduced, v->ntasks = ntasks, v->taskoff = taskoff, v->mem_GiB = mem_GiB;

    const index_t n_orb = t->n_orb, n_spin = t->n_spin, n_orbff = t->n_orbff;
    const index_t nb2 = POW2(n_orb*n_spin);
    const index_t* nk = t->model->nk;
    const index_t* nkf = t->model->nkf;
    const index_t nkc = nk[0]*nk[1]*nk[2];
    const index_t K = 2*nkf[0]*nkf[1]*nkf[2];

    // tasks and their unique D = R3 - R1
    v->tasks = calloc( ntasks, sizeof*v->tasks );
    v->dvec = calloc( 3*ntasks, sizeof*v->dvec );
    for (index_t m=0; m<ntasks; ++m) {
        gemm_task_t* tk = v->tasks + m;
        if (reduced) {
            tk->mo1 = t->tu_data->unique_mi_l[m+taskoff];
            tk->mo3 = t->tu_data->unique_mi_r[m+taskoff];
        } else {
            tk->mo3 = (taskoff+m)/n_orbff;
            tk->mo1 = (taskoff+m) - tk->mo3*n_orbff;
        }
        tk->o1 = t->mi_to_ofrom[tk->mo1];
        tk->ob1 = t->mi_to_oto[tk->mo1];
        tk->o3 = t->mi_to_ofrom[tk->mo3];
        tk->ob3 = t->mi_to_oto[tk->mo3];
        index_t d[3];
        for (int i=0; i<3; ++i)
            d[i] = t->mi_to_R[3*tk->mo3+i] - t->mi_to_R[3*tk->mo1+i];
        index_t j = 0;
        for (; j<v->nd; ++j)
            if (v->dvec[3*j] == d[0] && v->dvec[3*j+1] == d[1] && v->dvec[3*j+2] == d[2]) break;
        if (j == v->nd) {
            for (int i=0; i<3; ++i) v->dvec[3*v->nd+i] = d[i];
            v->nd++;
        }
        tk->d = j;
    }

    // rows (c1, D) and columns c2 that are needed
    const index_t nd = v->nd;
    v->rowmap = malloc( sizeof(index_t)*nb2*nd );
    v->colmap = malloc( sizeof(index_t)*nb2 );
    v->row_c1 = malloc( sizeof(index_t)*nb2*nd );
    v->row_d = malloc( sizeof(index_t)*nb2*nd );
    v->col_c2 = malloc( sizeof(index_t)*nb2 );
    for (index_t i=0; i<nb2*nd; ++i) v->rowmap[i] = -1;
    for (index_t i=0; i<nb2; ++i) v->colmap[i] = -1;
    for (index_t m=0; m<ntasks; ++m) {
        const gemm_task_t* tk = v->tasks + m;
        for (index_t s3=0; s3<n_spin; ++s3)
        for (index_t s1=0; s1<n_spin; ++s1) {
            index_t c1 = tk->o1 + n_orb*(s1 + n_spin*(tk->o3 + n_orb*s3));
            index_t key = c1 + nb2*tk->d;
            if (v->rowmap[key] < 0) {
                v->row_c1[v->nrow] = c1;
                v->row_d[v->nrow] = tk->d;
                v->rowmap[key] = v->nrow++;
            }
        }
        for (index_t s4=0; s4<n_spin; ++s4)
        for (index_t s2=0; s2<n_spin; ++s2) {
            index_t c2 = pp ? tk->ob1 + n_orb*(s2 + n_spin*(tk->ob3 + n_orb*s4)) :
                              tk->ob3 + n_orb*(s2 + n_spin*(tk->ob1 + n_orb*s4));
            if (v->colmap[c2] < 0) {
                v->col_c2[v->ncol] = c2;
                v->colmap[c2] = v->ncol++;
            }
        }
    }

    // chunking of rows such that C stays below the memory limit
    index_t chunk = (index_t)(mem_GiB*(double)(1l<<30) / (double)(sizeof(complex128_t)*nkc*v->ncol));
    v->chunk = MAX(1, MIN(chunk, v->nrow));
    index_t rblk = TU_GEMM_LOOP_ABUF_BYTES / (index_t)(sizeof(complex128_t)*K);
    v->rblk = MAX(1, MIN(rblk, v->chunk));

    // buffers (shared by variants, grown if needed)
    gemm_buffer_reserve( &g->Cbuf, &g->Cbuf_sz, nkc*v->chunk*v->ncol );
    gemm_buffer_reserve( &g->Abuf, &g->Abuf_sz, g->nthr*v->rblk*K );
    gemm_buffer_reserve( &g->Bbuf, &g->Bbuf_sz, g->nthr*K*v->ncol );

    // FFT over the coarse r of every (row, col) of a chunk: stride S.
    int dims_coarse[3] = { (int)nk[0], (int)nk[1], (int)nk[2] };
    const index_t S_full = v->chunk*v->ncol;
    const index_t S_last = (v->nrow % v->chunk)*v->ncol;
    fftw_set_timelimit( TU_GEMM_LOOP_FFTW_TIMELIMIT );
    v->plan_full = fftw_plan_many_dft( 3, dims_coarse, (int)S_full,
            (fftw_complex*)g->Cbuf, NULL, (int)S_full, 1,
            (fftw_complex*)g->Cbuf, NULL, (int)S_full, 1,
            pp ? FFTW_FORWARD : FFTW_BACKWARD, TU_GEMM_LOOP_FFTW_FLAGS );
    if (S_last > 0)
        v->plan_last = fftw_plan_many_dft( 3, dims_coarse, (int)S_last,
                (fftw_complex*)g->Cbuf, NULL, (int)S_last, 1,
                (fftw_complex*)g->Cbuf, NULL, (int)S_last, 1,
                pp ? FFTW_FORWARD : FFTW_BACKWARD, TU_GEMM_LOOP_FFTW_FLAGS );
    fftw_set_timelimit( FFTW_NO_TIMELIMIT );
    v->valid = true;
    g->t_plan += diverge_mpi_wtime() - tick;
}

double tu_gemm_loop_plan_time( const tu_gemm_loop_t* g ) {
    return g ? g->t_plan : 0.0;
}

static tu_gemm_loop_t* gemm_loop_get( tu_prop_t* t, bool pp, index_t ntasks, index_t taskoff,
        bool reduced ) {
    if (!t->gemm_loop) {
        const double tick = diverge_mpi_wtime();
        tu_gemm_loop_t* g = calloc( 1, sizeof(*g) );
        g->nthr = diverge_omp_num_threads();
        const index_t* nkf = t->model->nkf;
        g->Fmap = calloc( g->nthr*(nkf[0]+nkf[1]+nkf[2]), sizeof*g->Fmap );
        if (!t->gemm_gf_layout)
            g->Gt = (complex128_t*)fftw_alloc_complex( 2*POW2(t->n_orb*t->n_spin)*t->nktot );
        g->t_plan += diverge_mpi_wtime() - tick;
        t->gemm_loop = g;
    }
    tu_gemm_loop_t* g = t->gemm_loop;
    const double mem_GiB = t->tu_data->gemm_loop_mem_GiB > 0 ? t->tu_data->gemm_loop_mem_GiB : 4.0;
    const gemm_variant_t* v = g->v + (pp ? 1 : 0);
    if (!v->valid || v->reduced != reduced || v->ntasks != ntasks || v->taskoff != taskoff ||
            v->mem_GiB != mem_GiB)
        gemm_variant_init( t, g, pp, ntasks, taskoff, reduced, mem_GiB );
    return g;
}

void S_loop_static_gemm( tu_prop_t* t_, bool pp, bool subtract,
        complex128_t* loop, index_t ntasks, index_t taskoff, bool reduced ) {

    const tu_prop_t* t = t_;
    const index_t n_orb = t->n_orb, n_spin = t->n_spin, n_orbff = t->n_orbff;
    const index_t nb2 = POW2(n_orb*n_spin);
    const index_t* nk = t->model->nk;
    const index_t* nkf = t->model->nkf;
    const index_t nkc = nk[0]*nk[1]*nk[2];
    const index_t nF = nkf[0]*nkf[1]*nkf[2];
    const index_t nktot = t->nktot;
    const index_t K = 2*nF;
    const index_t sigma = pp ? 1 : -1;

    if (ntasks == -1) {
        ntasks = reduced ? t->tu_data->unique_mi_l_sz : POW2(n_orbff);
        taskoff = 0;
    }
    if (ntasks <= 0) return;

    tu_gemm_loop_t* g = gemm_loop_get( t_, pp, ntasks, taskoff, reduced );
    const gemm_variant_t* v = g->v + (pp ? 1 : 0);
    const gemm_task_t* tasks = v->tasks;
    const index_t* dvec = v->dvec;
    const index_t *rowmap = v->rowmap, *colmap = v->colmap;
    const index_t *row_c1 = v->row_c1, *row_d = v->row_d, *col_c2 = v->col_c2;
    const index_t nrow = v->nrow, ncol = v->ncol, chunk = v->chunk, rblk = v->rblk;
    complex128_t *Cbuf = g->Cbuf, *Abuf = g->Abuf, *Bbuf = g->Bbuf;
    index_t* Fmap = g->Fmap;
    const int nthr = g->nthr;

    // Green's functions as [c][r][F] per term: 0 GF_fft_plus, 1 GF_fft_conj_plus
    const complex128_t* Gterm[2] = { t->GF_fft_plus, t->GF_fft_conj_plus };
    if (!t->gemm_gf_layout) {
        complex128_t* Gt = g->Gt;
        const complex128_t* src[2] = { t->GF_fft_plus, t->GF_fft_conj_plus };
        const index_t D1 = nk[1]*nkf[1], D2 = nk[2]*nkf[2];
        #pragma omp parallel for collapse(3) schedule(static) num_threads(nthr)
        for (index_t a=0; a<2; ++a)
        for (index_t c=0; c<nb2; ++c)
        for (index_t r=0; r<nkc; ++r) {
            const index_t rx = r/(nk[1]*nk[2]), ry = (r/nk[2])%nk[1], rz = r%nk[2];
            const complex128_t* S = src[a] + c*nktot;
            complex128_t* T = Gt + ((a*nb2 + c)*nkc + r)*nF;
            for (index_t Fx=0; Fx<nkf[0]; ++Fx)
            for (index_t Fy=0; Fy<nkf[1]; ++Fy)
            for (index_t Fz=0; Fz<nkf[2]; ++Fz) {
                index_t kx = rx + nk[0]*Fx, ky = ry + nk[1]*Fy, kz = rz + nk[2]*Fz;
                T[Fz + nkf[2]*(Fy + nkf[1]*Fx)] = S[kz + D2*(ky + D1*kx)];
            }
        }
        Gterm[0] = Gt;
        Gterm[1] = Gt + nb2*nktot;
    }
    // term t of the A (X) and B (Y) operands
    //   pp: X = (conj_plus, plus), Y = (plus, conj_plus)
    //   ph: X = (plus, conj_plus), Y = (plus, conj_plus)
    const index_t Xterm[2] = { pp ? 1 : 0, pp ? 0 : 1 };
    const index_t Yterm[2] = { 0, 1 };

    const int gemm_handle = single_gemm_parallel_begin();
    for (index_t row0=0; row0<nrow; row0+=chunk) {
        const index_t nr = MIN(chunk, nrow-row0);
        const index_t S = nr*ncol;
        const index_t nblk = (nr + rblk-1)/rblk;
        // contiguous ranges of r per thread: neighbouring r share most of the
        // Green's function blocks r' = sr + D (cache reuse), B_r is built once
        // per r and all row blocks follow
        #pragma omp parallel for schedule(static) num_threads(nthr)
        for (index_t r=0; r<nkc; ++r) {
            const int th = omp_get_thread_num();
            complex128_t* A = Abuf + th*rblk*K;
            complex128_t* B = Bbuf + th*K*ncol;
            index_t* mapx = Fmap + th*(nkf[0]+nkf[1]+nkf[2]);
            index_t* mapy = mapx + nkf[0];
            index_t* mapz = mapy + nkf[1];
            const index_t rv[3] = { r/(nk[1]*nk[2]), (r/nk[2])%nk[1], r%nk[2] };

            // B_r[(term,F), col]
            for (index_t a=0; a<2; ++a)
            for (index_t col=0; col<ncol; ++col) {
                const complex128_t* Y = Gterm[Yterm[a]] + (col_c2[col]*nkc + r)*nF;
                for (index_t F=0; F<nF; ++F)
                    B[(a*nF + F)*ncol + col] = Y[F];
            }

            for (index_t ib=0; ib<nblk; ++ib) {

            // A_r[(c1,D), (term,F)] for the rows of this block
            const index_t rb0 = row0 + ib*rblk;
            const index_t nrb = MIN(rblk, row0+nr-rb0);
            index_t dprev = -1, rp = 0;
            for (index_t i=0; i<nrb; ++i) {
                const index_t row = rb0 + i;
                const index_t d = row_d[row];
                if (d != dprev) {
                    index_t rpv[3];
                    index_t* maps[3] = { mapx, mapy, mapz };
                    for (int x=0; x<3; ++x) {
                        index_t u = sigma*rv[x] + dvec[3*d+x];
                        index_t c = floordiv(u, nk[x]);
                        rpv[x] = u - c*nk[x];
                        for (index_t F=0; F<nkf[x]; ++F)
                            maps[x][F] = posmod(c + sigma*F, nkf[x]);
                    }
                    rp = rpv[2] + nk[2]*(rpv[1] + nk[1]*rpv[0]);
                    dprev = d;
                }
                for (index_t a=0; a<2; ++a) {
                    const complex128_t* X = Gterm[Xterm[a]] + (row_c1[row]*nkc + rp)*nF;
                    complex128_t* Ai = A + i*K + a*nF;
                    for (index_t Fx=0; Fx<nkf[0]; ++Fx)
                    for (index_t Fy=0; Fy<nkf[1]; ++Fy) {
                        const complex128_t* Xxy = X + nkf[2]*(mapy[Fy] + nkf[1]*mapx[Fx]);
                        complex128_t* Axy = Ai + nkf[2]*(Fy + nkf[1]*Fx);
                        for (index_t Fz=0; Fz<nkf[2]; ++Fz)
                            Axy[Fz] = Xxy[mapz[Fz]];
                    }
                }
            }

            // C_r[row, col] (row major, rows rb0..rb0+nrb of this chunk)
            single_gemm_parallel( A, B, Cbuf + r*S + (rb0-row0)*ncol, ncol, nrb, K, 1.0, 0.0 );
            }
        }

        // FFT over the coarse r of every (row, col)
        fftw_execute_dft( nr == chunk ? v->plan_full : v->plan_last,
                (fftw_complex*)Cbuf, (fftw_complex*)Cbuf );

        // put into loop
        #pragma omp parallel for collapse(5) schedule(static) num_threads(nthr)
        for (index_t s3=0; s3<n_spin; ++s3)
        for (index_t s2=0; s2<n_spin; ++s2)
        for (index_t s4=0; s4<n_spin; ++s4)
        for (index_t s1=0; s1<n_spin; ++s1)
        for (index_t m=0; m<ntasks; ++m) {
            const gemm_task_t* tk = tasks + m;
            index_t c1 = tk->o1 + n_orb*(s1 + n_spin*(tk->o3 + n_orb*s3));
            index_t row = rowmap[c1 + nb2*tk->d];
            if (row < row0 || row >= row0+nr) continue;
            index_t c2 = pp ? tk->ob1 + n_orb*(s2 + n_spin*(tk->ob3 + n_orb*s4)) :
                              tk->ob3 + n_orb*(s2 + n_spin*(tk->ob1 + n_orb*s4));
            index_t col = colmap[c2];
            const complex128_t* src = Cbuf + (row-row0)*ncol + col;
            const index_t mo1 = tk->mo1, mo3 = tk->mo3;
            for (index_t q=0; q<t->my_nk; ++q) {
                index_t idx_to = pp ?
                    mo1+n_orbff*(s1+s2*n_spin+POW2(n_spin)*(mo3+n_orbff*(s3 + s4*n_spin+
                                    POW2(n_spin)*q))) :
                    mo1+n_orbff*(s1+s4*n_spin+POW2(n_spin)*(mo3+n_orbff*(s3 + s2*n_spin+
                                    POW2(n_spin)*q)));
                if (subtract)
                    loop[idx_to] -= src[t->c2cmap[q]*S];
                else
                    loop[idx_to] = src[t->c2cmap[q]*S];
            }
        }
    }

    single_gemm_parallel_end( gemm_handle );
}

// in-place permutation of the real-space Green's function gf (nb^2 components
// of nktot fine points, fine index k = kz + D2*(ky + D1*kx)) into the layout
// [c][r][F] (k = r + nk·F per axis), with one component-sized scratch buffer
void tu_gemm_gf_to_layout( const tu_prop_t* t, complex128_t* gf ) {
    const index_t nb2 = POW2(t->n_orb*t->n_spin);
    const index_t* nk = t->model->nk;
    const index_t* nkf = t->model->nkf;
    const index_t nkc = nk[0]*nk[1]*nk[2];
    const index_t nF = nkf[0]*nkf[1]*nkf[2];
    const index_t nktot = t->nktot;
    const index_t D1 = nk[1]*nkf[1], D2 = nk[2]*nkf[2];
    complex128_t* scratch = (complex128_t*)fftw_alloc_complex( nktot );
    for (index_t c=0; c<nb2; ++c) {
        complex128_t* G = gf + c*nktot;
        #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t k=0; k<nktot; ++k) scratch[k] = G[k];
        #pragma omp parallel for schedule(static) num_threads(diverge_omp_num_threads())
        for (index_t r=0; r<nkc; ++r) {
            const index_t rx = r/(nk[1]*nk[2]), ry = (r/nk[2])%nk[1], rz = r%nk[2];
            complex128_t* T = G + r*nF;
            for (index_t Fx=0; Fx<nkf[0]; ++Fx)
            for (index_t Fy=0; Fy<nkf[1]; ++Fy)
            for (index_t Fz=0; Fz<nkf[2]; ++Fz) {
                index_t kx = rx + nk[0]*Fx, ky = ry + nk[1]*Fy, kz = rz + nk[2]*Fz;
                T[Fz + nkf[2]*(Fy + nkf[1]*Fx)] = scratch[kz + D2*(ky + D1*kx)];
            }
        }
    }
    fftw_free( scratch );
}
