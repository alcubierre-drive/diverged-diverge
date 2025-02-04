#include "post_processing.h"
#include "vertex_memory.h"
#include "loops.h"
#include "reorder.h" // dist_info
#include "../misc/version.h"
#include "../misc/batched_gemms.h"
#include "../misc/eigen.h"
#include "../misc/gemm.h"

typedef struct {
    grid_vertex_t* V;
    grid_loop_t* L;

    uindex_t num_singular_values;
    complex128_t* singular_matrix_U;
    complex128_t* singular_matrix_V;
    double* singular_values;

    complex128_t* eigen_vectors;
    double* eigen_values;

    index_t nk, nb, qsize;
    bool is_q_zero;
} grid_lingap_t;
static grid_lingap_t* grid_lingap_init( grid_vertex_t* V, grid_loop_t* L,
        double T, char chan, uindex_t n, bool use_loop, const char* vertex_file );
static void grid_lingap_free( grid_lingap_t* l );
static void grid_lingap_calc( grid_lingap_t* l, double T, char chan, bool use_loop,
        const char* vertex_file );
static double* grid_lingap_pack_data( grid_lingap_t* l, index_t* count_doubles );

struct grid_postproc_t {
    index_t header[64];
    grid_vertex_t* V;
    grid_loop_t* L;

    index_t qsize, nb, nk, nff, Xsize;
    bool SU2;

    bool use_loop/*=true*/;
    uindex_t num_singular_values/*=10*/;
    bool has_communicated;

    complex128_t *susc_P_dist, *susc_C_dist, *susc_D_dist;
    grid_lingap_t *lgap_P, *lgap_C, *lgap_D;
};
static void grid_postproc_calc_sus( grid_postproc_t* p, complex128_t* buf, double T, char chan );
static void grid_postproc_communicate_susceptibility_to_master( grid_postproc_t* p, char chan );

grid_postproc_t* grid_postproc_init( grid_vertex_t* V, grid_loop_t* L ) {
    grid_postproc_t* p = calloc(1, sizeof*p);
    p->V = V;
    p->L = L;
    p->SU2 = V->SU2;
    p->qsize = L->qsize;
    p->nk = L->nk;
    p->nb = L->nb;
    p->nff = L->nff;
    p->Xsize = p->qsize * POW4(p->nb) * p->nff;

    p->use_loop = true;
    p->num_singular_values = 10;

    return p;
}

void grid_postproc_free( grid_postproc_t* p ) {
    free( p->susc_P_dist );
    free( p->susc_C_dist );
    free( p->susc_D_dist );
    if (p->lgap_P) grid_lingap_free(p->lgap_P);
    if (p->lgap_C) grid_lingap_free(p->lgap_C);
    if (p->lgap_D) grid_lingap_free(p->lgap_D);
    free( p );
}

void grid_postproc_set_num_singular_values( grid_postproc_t* p, uindex_t n ) {
    p->num_singular_values = n;
}

void grid_postproc_set_loop_usage( grid_postproc_t* p, bool loop ) {
    p->use_loop = loop;
}

void grid_postproc_lingap( grid_postproc_t* p, double T, char chan, const char* vertex_file ) {
    grid_lingap_t** l = chan == 'P' ? &p->lgap_P :
                        chan == 'C' ? &p->lgap_C :
                        /*chan=='D'*/ &p->lgap_D;
    *l = grid_lingap_init( p->V, p->L, T, chan, p->num_singular_values, p->use_loop, vertex_file );
}

void grid_postproc_susc( grid_postproc_t* p, double T, char chan ) {
    complex128_t** sus_buf = chan == 'P' ? &p->susc_P_dist :
                             chan == 'C' ? &p->susc_C_dist :
                             /*chan=='D'*/ &p->susc_D_dist;
    *sus_buf = calloc(p->Xsize, sizeof**sus_buf);
    if (*sus_buf) grid_postproc_calc_sus( p, *sus_buf, T, chan );
}

void grid_postproc_communicate( grid_postproc_t* p ) {
    grid_postproc_communicate_susceptibility_to_master( p, 'P' );
    grid_postproc_communicate_susceptibility_to_master( p, 'C' );
    grid_postproc_communicate_susceptibility_to_master( p, 'D' );
    p->has_communicated = true;
}

// header consists of 64 index_t specifying the counts and
// displacements (in units of doubles) of the channels, i.e.:
// nk, nb, nff, SU2, formfactor_displ, formfactor_count,
// P_susc_displ, P_susc_count, P_lin_displ, P_lin_count,
// C_susc_displ, C_susc_count, C_lin_displ, C_lin_count,
// D_susc_displ, D_susc_count, D_lin_displ, D_lin_count,
// lingap_num_ev, lingap_matrix_size (linear),
// ...
// file_size
//
// header is followed by data (size*sizeof(double))
const index_t* grid_postproc_to_file( grid_postproc_t* p, const char* fname ) {
    if (!p->has_communicated) {
        mpi_err_printf( "must have communicated before to_file can be called\n" );
        return NULL;
    }
    index_t size_lP = 0, size_lC = 0, size_lD = 0;
    double *lP = NULL, *lC = NULL, *lD = NULL;
    if (p->lgap_P) lP = grid_lingap_pack_data( p->lgap_P, &size_lP );
    if (p->lgap_C) lC = grid_lingap_pack_data( p->lgap_C, &size_lC );
    if (p->lgap_D) lD = grid_lingap_pack_data( p->lgap_D, &size_lD );

    index_t size_susc_P = p->susc_P_dist == NULL ? 0 : p->nk*p->nff*2*POW4(p->nb);
    index_t size_susc_C = p->susc_C_dist == NULL ? 0 : p->nk*p->nff*2*POW4(p->nb);
    index_t size_susc_D = p->susc_D_dist == NULL ? 0 : p->nk*p->nff*2*POW4(p->nb);

    index_t* header = p->header; ZERO_OUT( header, 64 );
    int i=0;

    header[i] = MAGIC_NUMBER_POST_PROCESSING; i++;
    header[i] = p->nk; i++;
    header[i] = p->nb; i++;
    header[i] = p->nff; i++;
    header[i] = p->SU2; i++;

    // formfactors
    header[i] = 64; i++;
    header[i] = p->nff * p->nk * 2; i++;

    // susc P
    header[i] = header[i-2] + header[i-1]; i++;
    header[i] = size_susc_P; i++;
    // lin gap P
    header[i] = header[i-2] + header[i-1]; i++;
    header[i] = size_lP; i++;

    // susc C
    header[i] = header[i-2] + header[i-1]; i++;
    header[i] = size_susc_C; i++;
    // lin gap C
    header[i] = header[i-2] + header[i-1]; i++;
    header[i] = size_lC; i++;

    // susc D
    header[i] = header[i-2] + header[i-1]; i++;
    header[i] = size_susc_D; i++;
    // lin gap D
    header[i] = header[i-2] + header[i-1]; i++;
    header[i] = size_lD; i++;

    index_t total_size = header[i-2] + header[i-1];

    // lingap info
    header[i] = p->num_singular_values; i++;
    header[i] = p->nk*POW2(p->nb); i++;

    // file format version string
    header[60] = total_size;
    const char* version = tag_version();
    index_t version_sz = ceil( (double)strlen(version)/(double)(sizeof(double)) );
    void* version_mem = calloc( version_sz, sizeof(double) );
    memcpy( version_mem, version, strlen(version) );
    header[61] = version_sz;
    total_size += header[61];

    header[62] = MAGIC_NUMBER_POST_PROCESSING;
    header[63] = total_size;

    if (diverge_mpi_comm_rank() != 0) return header;

    FILE* outfile = fopen( fname, "w" );
    if (!outfile) {
        mpi_err_printf( "could not open file '%s'\n", fname );
        return header;
    }
    fwrite( header, sizeof(index_t), 64, outfile );
    fwrite( p->L->formfac, sizeof(double), header[6], outfile );

    fwrite( p->V->vertex_buffers[1], sizeof(double), header[8], outfile );
    if (lP) {
        fwrite( lP, sizeof(double), header[10], outfile );
        free(lP);
    }

    fwrite( p->V->vertex_buffers[2], sizeof(double), header[12], outfile );
    if (lC) {
        fwrite( lC, sizeof(double), header[14], outfile );
        free(lC);
    }

    fwrite( p->V->vertex_buffers[3], sizeof(double), header[16], outfile );
    if (lD) {
        fwrite( lD, sizeof(double), header[18], outfile );
        free(lD);
    }

    fwrite( version_mem, sizeof(double), version_sz, outfile );
    free( version_mem );

    fclose( outfile );
    return header;
}

void grid_postproc_print_header( const index_t* header ) {
    if (!header) {
        mpi_err_printf( "header is empty\n" );
        return;
    }
    index_t nk = header[1];
    index_t nb = header[2];
    index_t nff = header[3];
    index_t SU2 = header[4];

    index_t ff[2] = { header[5], header[6] };
    index_t P_susc[2] = { header[7], header[8] };
    index_t P_lin_gap[2] = { header[9], header[10] };
    index_t C_susc[2] = { header[11], header[12] };
    index_t C_lin_gap[2] = { header[13], header[14] };
    index_t D_susc[2] = { header[15], header[16] };
    index_t D_lin_gap[2] = { header[17], header[18] };

    index_t lingap_num_ev = header[19];
    index_t lingap_matrix_sz = header[20];

    index_t magic_number = header[62];
    index_t file_size = header[63];

    mpi_fil_printf( "file size %li (*64bit), magic number %li\n", file_size, magic_number );
    mpi_fil_printf( "nk=%li nb=%li nff=%li SU2=%i nev=%li mat_sz=%li\n", nk, nb, nff,
            SU2, lingap_num_ev, lingap_matrix_sz );
    mpi_fil_printf( "ff      %10li (+%10li; *64bit)\n", ff[0]       , ff[1] );
    mpi_fil_printf( "P susc  %10li (+%10li; *64bit)\n", P_susc[0]   , P_susc[1] );
    mpi_fil_printf( "P lingap%10li (+%10li; *64bit)\n", P_lin_gap[0], P_lin_gap[1] );
    mpi_fil_printf( "C susc  %10li (+%10li; *64bit)\n", C_susc[0]   , C_susc[1] );
    mpi_fil_printf( "C lingap%10li (+%10li; *64bit)\n", C_lin_gap[0], C_lin_gap[1] );
    mpi_fil_printf( "D susc  %10li (+%10li; *64bit)\n", D_susc[0]   , D_susc[1] );
    mpi_fil_printf( "D lingap%10li (+%10li; *64bit)\n", D_lin_gap[0], D_lin_gap[1] );
}

static grid_lingap_t* grid_lingap_init( grid_vertex_t* V, grid_loop_t* L,
        double T, char chan, uindex_t n, bool use_loop, const char* vertex_file ) {
    grid_lingap_t* l = calloc(1, sizeof*l);
    l->V = V;
    l->L = L;
    l->num_singular_values = n;
    l->nk = L->nk;
    l->nb = L->nb;
    l->qsize = L->qsize;
    l->is_q_zero = (V->dist->myrank == 0);
    if (l->is_q_zero) {
        l->singular_matrix_U = calloc(l->nk*POW2(l->nb)*l->num_singular_values,sizeof(complex128_t));
        l->singular_matrix_V = calloc(l->nk*POW2(l->nb)*l->num_singular_values,sizeof(complex128_t));
        l->singular_values = calloc(l->nk*POW2(l->nb),sizeof(double));

        l->eigen_vectors = calloc(l->nk*POW2(l->nb)*l->num_singular_values,sizeof(complex128_t));
        l->eigen_values = calloc(l->nk*POW2(l->nb),sizeof(double));
    }

    grid_lingap_calc( l, T, chan, use_loop, vertex_file );
    return l;
}

static void grid_lingap_free( grid_lingap_t* l ) {
    free( l->singular_matrix_U );
    free( l->singular_matrix_V );
    free( l->singular_values );
    free( l->eigen_vectors );
    free( l->eigen_values );
    free( l );
}

static void grid_lingap_calc( grid_lingap_t* l, double T, char chan, bool use_loop,
        const char* vertex_file ) {

    complex128_t* loop = use_loop ? chan == 'P' ? grid_loop_calc( l->L, 'P', T ) :
                                                  grid_loop_calc( l->L, 'C', T ) :
                                    NULL;
    switch (chan) {
        case 'P': grid_vertex_project_name( l->V, l->V->vertex_buffers[0],
                          l->V->vertex_buffers[1], "D_to_P", false, 1.0 ); break;
        case 'C': grid_vertex_project_name( l->V, l->V->vertex_buffers[0],
                          l->V->vertex_buffers[1], "D_to_C", false, 1.0 ); break;
        default:
        case 'D': grid_vertex_copy( l->V, l->V->vertex_buffers[0],
                          l->V->vertex_buffers[1], false, 1.0 ); break;
    }

    if (!l->is_q_zero) return;

    index_t N = l->nk * POW2(l->nb);
    if (vertex_file && strcmp(vertex_file,"")) {
        FILE* v = fopen( vertex_file, "w" );
        mpi_log_printf( "writing MF q=0 '%c' vertex to file\n", chan )
        if (!v) {
            mpi_err_printf( "could not open file %s\n", vertex_file );
        } else {
            fwrite( l->V->vertex_buffers[1], sizeof(complex128_t), POW2(N), v );
            fclose( v );
        }
    }

    complex128_t* v = l->V->vertex_buffers[1];
    complex128_t* U_ = l->V->vertex_buffers[2];
    complex128_t* V_ = l->V->vertex_buffers[3];

    // eigensolution of vertex (hermitian!)
    single_eigen( v, U_, l->eigen_values, N );
    transpose_inplace( U_, N );
    single_eigen_sort( U_, l->eigen_values, N, 0 );
    for (index_t i=0; i<N; ++i)
    for (uindex_t v=0; v<l->num_singular_values; ++v)
        l->eigen_vectors[v*N+i] = U_[IDX2(v,i,N)];

    if (loop)
        batched_gemm_vertex_loop( l->V->vertex_buffers[1], loop, l->qsize, l->nk, l->nb );
    single_svd( v, U_, V_, l->singular_values, N );
    for (index_t i=0; i<N; ++i)
    for (uindex_t v=0; v<l->num_singular_values; ++v) {
        l->singular_matrix_U[v*N+i] = U_[IDX2(i,v,N)];
        l->singular_matrix_V[v*N+i] = conj(V_[IDX2(v,i,N)]);
    }
}

static double* grid_lingap_pack_data( grid_lingap_t* l, index_t* count_doubles ) {
    index_t N = l->nk*POW2(l->nb);
    index_t Nmat = N*l->num_singular_values*2;
    *count_doubles = Nmat*3 + l->num_singular_values*2;

    if (!l->is_q_zero) return NULL;

    double* result = malloc(*count_doubles * sizeof*result);
    for (index_t i=0; i<Nmat; ++i)
        result[i] = ((double*)(l->singular_matrix_U))[i];
    for (index_t i=0; i<Nmat; ++i)
        result[Nmat+i] = ((double*)(l->singular_matrix_V))[i];
    for (uindex_t i=0; i<l->num_singular_values; ++i)
        result[Nmat*2+i] = l->singular_values[i];
    for (index_t i=0; i<Nmat; ++i)
        result[Nmat*2+l->num_singular_values+i] = ((double*)l->eigen_vectors)[i];
    for (uindex_t i=0; i<l->num_singular_values; ++i)
        result[Nmat*3+l->num_singular_values+i] = l->eigen_values[i];
    return result;
}

static void grid_postproc_calc_sus( grid_postproc_t* p, complex128_t* buf, double T, char chan ) {
    complex128_t* loop = chan == 'P' ? grid_loop_calc( p->L, 'P', T ) :
                                       grid_loop_calc( p->L, 'C', T );
    switch (chan) {
        case 'P': grid_vertex_project_name( p->V, p->V->vertex_buffers[0],
                          p->V->vertex_buffers[1], "D_to_P", false, 1.0 ); break;
        case 'C': grid_vertex_project_name( p->V, p->V->vertex_buffers[0],
                          p->V->vertex_buffers[1], "D_to_C", false, 1.0 ); break;
        default:
        case 'D': grid_vertex_copy( p->V, p->V->vertex_buffers[0],
                          p->V->vertex_buffers[1], false, 1.0 ); break;
    }

    batched_gemm_vertex_loop( p->V->vertex_buffers[1], loop, p->qsize, p->nk, p->nb );
    batched_gemm_loop_vertex( loop, p->V->vertex_buffers[1], p->qsize, p->nk, p->nb );

    const complex128_t *ff = p->L->formfac,
                       *VV = p->V->vertex_buffers[1];
    #pragma omp parallel for collapse(6) num_threads(diverge_omp_num_threads())
    for (index_t q=0; q<p->qsize; ++q)
    for (index_t f=0; f<p->nff; ++f)
    for (index_t b1=0; b1<p->nb; ++b1)
    for (index_t b2=0; b2<p->nb; ++b2)
    for (index_t b3=0; b3<p->nb; ++b3)
    for (index_t b4=0; b4<p->nb; ++b4) {
        for (index_t k=0; k<p->nk; ++k)
        for (index_t kp=0; kp<p->nk; ++kp) {
            index_t Vertexindex = IDX7( q, k,b1,b2, kp,b3,b4, p->nk,p->nb,p->nb, p->nk,p->nb,p->nb );
            index_t Suscindex = IDX6( q,f, b1,b2,b3,b4, p->nff,p->nb,p->nb,p->nb,p->nb );
            buf[Suscindex] += VV[Vertexindex] * ff[f*p->nk+k] * ff[f*p->nk+kp];
        }
    }
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<p->Xsize; ++i)
        buf[i] /= (double)(p->nk*p->nk);
}

static void grid_postproc_communicate_susceptibility_to_master( grid_postproc_t* p, char chan ) {
    complex128_t* send_buf = chan == 'P' ? p->susc_P_dist :
                             chan == 'C' ? p->susc_C_dist :
                             /*chan=='D'*/ p->susc_D_dist;
    if (!send_buf) return;

    complex128_t* recv_buf = chan == 'P' ? p->V->vertex_buffers[1] :
                             chan == 'C' ? p->V->vertex_buffers[2] :
                             /*chan=='D'*/ p->V->vertex_buffers[3];

    int* susc_counts_displs = calloc(2*p->V->dist->nranks, sizeof*susc_counts_displs);
    for (int i=0; i<2*p->V->dist->nranks; ++i)
        susc_counts_displs[i] = p->V->dist->qcounts[i] * p->nff * POW4(p->nb);

    int *susc_counts = susc_counts_displs,
        *susc_displs = susc_counts_displs + p->V->dist->nranks;

    diverge_mpi_barrier();
    diverge_mpi_gatherv_cdoub( send_buf, susc_counts[p->V->dist->myrank], recv_buf, susc_counts, susc_displs, 0 );
    diverge_mpi_barrier();

    free( susc_counts_displs );
}

