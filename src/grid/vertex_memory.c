#include "vertex_memory.h"
#include "reorder.h"
#include "../diverge_patching_helpers.h" // atomic_add
#include <alloca.h>

static void grid_vertex_gen_index_maps( grid_vertex_t* v );
static int grid_vertex_find_send_trafo( const grid_vertex_t* v, const char* name );
static int grid_vertex_find_recv_trafo( const grid_vertex_t* v, const char* name );
static void grid_vertex_gather_chan_in_v1( grid_vertex_t* v, char chan );

grid_vertex_t* grid_vertex_init( index_t nk, index_t nb, index_t* k1m2, index_t* k1p2, bool SU2 ) {
    grid_vertex_t* v = calloc(1, sizeof*v);
    v->SU2 = SU2;
    v->nk = nk;
    v->nb = nb;
    v->k1m2 = k1m2;
    v->k1p2 = k1p2;
    v->sub_D[0] = v->sub_D[1] = v->sub_D[2] = 1; // default

    v->dist = dist_info_init( nk, nb );
    v->myrank = v->dist->myrank;
    v->nranks = v->dist->nranks;
    v->qcounts = v->dist->qcounts;
    v->qdispls = v->dist->qdispls;
    v->counts = v->dist->counts;
    v->displs = v->dist->displs;
    v->qcount = v->qcounts[v->myrank];
    v->qdispl = v->qdispls[v->myrank];
    v->size = v->counts[v->myrank];
    v->n_send_recv_maps = 8;
    if (!v->SU2) v->n_send_recv_maps--;

    // print memory requirement
    index_t memory = sizeof(complex128_t) * (4/*vertex_buffers*/+2/*send/recv*/) * v->size +
                     sizeof(index_t) * v->n_send_recv_maps * v->size;
    index_t memory_per_rank[v->nranks];
    diverge_mpi_allgather_index( &memory, memory_per_rank, 1 );

    double mean_mem = 0.;
    index_t min_max[2] = {memory_per_rank[0], memory_per_rank[0]};
    for (index_t i=0; i<v->nranks; ++i) {
        mean_mem += memory_per_rank[i];
        min_max[0] = MIN(min_max[0], memory_per_rank[i]);
        min_max[1] = MAX(min_max[1], memory_per_rank[i]);
    }
    mean_mem /= (double)v->nranks;
    const double GB = 1024.*1024.*1024.;
    mpi_vrb_printf( "grid vertex size (per rank) %.1fGB~(%.1fGB)~%.1fGB\n",
            (double)min_max[0]/GB, mean_mem/GB, (double)min_max[1]/GB );

    // allocate map buffers
    for (int i=0; i<v->n_send_recv_maps; ++i) {
        v->send_recv_maps[i] = calloc(v->size, sizeof(index_t));
        v->send_recv_displs[i] = calloc(v->nranks, sizeof(int));
        v->send_recv_counts[i] = calloc(v->nranks, sizeof(int));
    }
    // then generate index maps (which allocates memory internally)
    grid_vertex_gen_index_maps( v );
    // and lastly allocate *other* buffers
    for (int i=0; i<4; ++i) v->vertex_buffers[i] = calloc(v->size, sizeof(complex128_t));
    v->send_pack = calloc(v->size, sizeof(complex128_t));
    v->recv_pack = calloc(v->size, sizeof(complex128_t));

    return v;
}

static void grid_vertex_gen_index_maps( grid_vertex_t* v ) {
    double tick = diverge_mpi_wtime();
    gen_maps( v->send_recv_maps[0], v->send_recv_displs[0], v->send_recv_counts[0],
              v->send_recv_maps[1], v->send_recv_displs[1], v->send_recv_counts[1],
              v->dist, v->nk, v->nb, v->k1m2, v->k1p2, index_gymnastics_D_to_P,
              index_gymnastics_P_to_D );
    strcpy( v->send_recv_descr[0], "D_to_P_send" );
    strcpy( v->send_recv_descr[1], "D_to_P_recv" );
    mpi_tim_printf( "gen_D_to_P_index_map() took %5.2fs\n", diverge_mpi_wtime()-tick );

    tick = diverge_mpi_wtime();
    gen_maps( v->send_recv_maps[2], v->send_recv_displs[2], v->send_recv_counts[2],
              v->send_recv_maps[3], v->send_recv_displs[3], v->send_recv_counts[3],
              v->dist, v->nk, v->nb, v->k1m2, v->k1p2, index_gymnastics_P_to_D,
              index_gymnastics_D_to_P );
    strcpy( v->send_recv_descr[2], "P_to_D_send" );
    strcpy( v->send_recv_descr[3], "P_to_D_recv" );
    mpi_tim_printf( "gen_P_to_D_index_map() took %5.2fs\n", diverge_mpi_wtime()-tick );

    if (v->SU2) {
        tick = diverge_mpi_wtime();
        gen_maps( v->send_recv_maps[4], v->send_recv_displs[4], v->send_recv_counts[4],
                  v->send_recv_maps[5], v->send_recv_displs[5], v->send_recv_counts[5],
                  v->dist, v->nk, v->nb, v->k1m2, v->k1p2, index_gymnastics_D_to_C,
                  index_gymnastics_C_to_D );
        strcpy( v->send_recv_descr[4], "D_to_C_send" );
        strcpy( v->send_recv_descr[5], "D_to_C_recv" );
        mpi_tim_printf( "gen_D_to_C_index_map() took %5.2fs\n", diverge_mpi_wtime()-tick );

        tick = diverge_mpi_wtime();
        gen_maps( v->send_recv_maps[6], v->send_recv_displs[6], v->send_recv_counts[6],
                  v->send_recv_maps[7], v->send_recv_displs[7], v->send_recv_counts[7],
                  v->dist, v->nk, v->nb, v->k1m2, v->k1p2, index_gymnastics_C_to_D,
                  index_gymnastics_D_to_C );
        strcpy( v->send_recv_descr[6], "C_to_D_send" );
        strcpy( v->send_recv_descr[7], "C_to_D_recv" );
        mpi_tim_printf( "gen_C_to_D_index_map() took %5.2fs\n", diverge_mpi_wtime()-tick );
    } else {
        tick = diverge_mpi_wtime();
        gen_maps( v->send_recv_maps[4], v->send_recv_displs[4], v->send_recv_counts[4],
                  v->send_recv_maps[5], v->send_recv_displs[5], v->send_recv_counts[5],
                  v->dist, v->nk, v->nb, v->k1m2, v->k1p2, index_gymnastics_D_swap_3_and_4,
                  index_gymnastics_D_swap_3_and_4 );
        strcpy( v->send_recv_descr[4], "D_swap_3_and_4_send" );
        strcpy( v->send_recv_descr[5], "D_swap_3_and_4_recv" );
        strcpy( v->send_recv_descr[6], "None" );
        strcpy( v->send_recv_descr[7], "None" );
        mpi_tim_printf( "gen_D_swap_3_and_4_index_map() took %5.2fs\n", diverge_mpi_wtime()-tick );
    }
}

void grid_vertex_free( grid_vertex_t* v ) {
    free(v->send_pack);
    free(v->recv_pack);
    for (int i=0; i<4; ++i) free(v->vertex_buffers[i]);
    for (int i=0; i<v->n_send_recv_maps; ++i) {
        free(v->send_recv_maps[i]);
        free(v->send_recv_displs[i]);
        free(v->send_recv_counts[i]);
    }
    dist_info_free( v->dist );
    free( v );
}

void grid_vertex_from_qbuf( grid_vertex_t* v, const complex128_t* qbuf_in, char chan_in, index_t ns ) {
    if (chan_in != 'P' && chan_in != 'C' && chan_in != 'D') {
        mpi_wrn_printf("invalid channel '%c', fallback to 'D'.\n", chan_in);
        chan_in = 'D';
    }
    const index_t nk1 = v->qcount;

    complex128_t* vbuf_in = v->vertex_buffers[2];
    complex128_t* vbuf = v->vertex_buffers[0];

    if (v->SU2) ns = 1;
    const index_t no = v->nb / ns;
    ZERO_OUT( vbuf_in, nk1*POW2(v->nk)*POW4(v->nb) );

    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for (index_t k1i=0; k1i<nk1; ++k1i)
    for (index_t k2=0; k2<v->nk; ++k2)
    for (index_t k3=0; k3<v->nk; ++k3) {
        index_t k1 = v->qdispl + k1i;
        index_t qq = 0;
        switch (chan_in) {
            case 'P': qq = v->k1p2[k1*v->nk+k2]; break;
            case 'C': qq = v->k1m2[k3*v->nk+k2]; break;
            case 'D': qq = v->k1m2[k1*v->nk+k3]; break;
            default: break;
        }
        for (index_t OA=0; OA<no; ++OA)
        for (index_t OB=0; OB<no; ++OB) {
            index_t o1 = 0, o2 = 0, o3 = 0, o4 = 0;
            switch (chan_in) {
                case 'P': o1 = OA; o2 = OA; o3 = OB; o4 = OB; break;
                case 'C': o1 = OA; o2 = OB; o3 = OB; o4 = OA; break;
                case 'D': o1 = OA; o2 = OB; o3 = OA; o4 = OB; break;
                default: break;
            }
            for (index_t SA=0; SA<ns; ++SA)
            for (index_t SB=0; SB<ns; ++SB)
            for (index_t SC=0; SC<ns; ++SC)
            for (index_t SD=0; SD<ns; ++SD) {
                index_t s1 = 0, s2 = 0, s3 = 0, s4 = 0;
                switch (chan_in) {
                    case 'P': s1 = SA; s2 = SB; s3 = SC; s4 = SD; break;
                    case 'C': s1 = SA; s2 = SD; s3 = SC; s4 = SB; break;
                    case 'D': s1 = SA; s2 = SD; s3 = SB; s4 = SC; break;
                    default: break;
                }
                index_t qbuf_in_idx = IDX7( qq, SA, SB, OA, SC, SD, OB, ns, ns, no, ns, ns, no );
                index_t b1 = IDX2(s1, o1, no),
                        b2 = IDX2(s2, o2, no),
                        b3 = IDX2(s3, o3, no),
                        b4 = IDX2(s4, o4, no);
                index_t vbuf_in_idx = IDX7( k1i, k2, k3, b1, b2, b3, b4, v->nk,v->nk,v->nb,v->nb,v->nb,v->nb );
                atomic_add( vbuf_in + vbuf_in_idx, qbuf_in + qbuf_in_idx );
            }
        }
    }

    grid_vertex_overwrite_transform( v, 0, 1, "V_to_D" );
    grid_vertex_project_name( v, vbuf_in, vbuf, "V_to_D", true, 1.0 );
    grid_vertex_overwrite_transform( v, 0, 1, "D_to_P" );
}

void grid_vertex_project_chans( grid_vertex_t* v, const complex128_t* in, complex128_t* out,
        char chan_in, char chan_out, bool add, double scale ) {
    if (chan_in == chan_out) {
        grid_vertex_copy( v, in, out, add, scale );
        return;
    }

    char trafo_name[64] = { chan_in, '_', 't', 'o', '_', chan_out, '\0' };
    if (((chan_in == 'D' && chan_out == 'C') || (chan_out == 'D' && chan_in == 'C')) && !v->SU2)
        strcpy( trafo_name, "D_swap_3_and_4" );

    int send_id = grid_vertex_find_send_trafo( v, trafo_name );
    int recv_id = grid_vertex_find_recv_trafo( v, trafo_name );

    bool update_trafo = (send_id == -1 || recv_id == -1);
    if (update_trafo) grid_vertex_overwrite_transform( v, 0, 1, trafo_name );
    grid_vertex_project_name( v, in, out, trafo_name, add, scale );
    if (update_trafo) grid_vertex_overwrite_transform( v, 0, 1, "D_to_P" );
}

void grid_vertex_project_name( grid_vertex_t* v, const complex128_t* in, complex128_t* out,
        const char* name, bool add, double scale ) {
    int send_id = grid_vertex_find_send_trafo(v, name);
    int recv_id = grid_vertex_find_recv_trafo(v, name);
    if (send_id == -1 || recv_id == -1) {
        mpi_err_printf( "could not find '%s' vertex transform\n", name );
        return;
    }
    full_pack_exchange( in, out, v->send_recv_maps[send_id], v->send_pack, v->send_recv_displs[send_id],
            v->send_recv_counts[send_id], v->send_recv_maps[recv_id], v->recv_pack,
            v->send_recv_displs[recv_id], v->send_recv_counts[recv_id], v->dist, add, scale );
}

void grid_vertex_copy( grid_vertex_t* v, const complex128_t* in, complex128_t* out, bool add_, double scale ) {
    const bool add = add_;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t i=0; i<v->size; ++i) {
        if (add) out[i] += in[i] * scale;
        else     out[i] = in[i] * scale;
    }
}

static int grid_vertex_find_send_trafo( const grid_vertex_t* v, const char* name ) {
    int result = -1;
    char* xnam = alloca(strlen(name)+6); strcpy(xnam, name); strcat(xnam, "_send");
    for (int i=0; i<8; ++i) if (!strcmp(v->send_recv_descr[i], xnam)) result = i;
    return result;
}

static int grid_vertex_find_recv_trafo( const grid_vertex_t* v, const char* name ) {
    int result = -1;
    char* xnam = alloca(strlen(name)+6); strcpy(xnam, name); strcat(xnam, "_recv");
    for (int i=0; i<8; ++i) if (!strcmp(v->send_recv_descr[i], xnam)) result = i;
    return result;
}

void grid_vertex_overwrite_transform( grid_vertex_t* v, int send, int recv, const char* new_name ) {
    if (0) {}
#define MAPGEN( name, name_inv ) else if (!strcmp(#name, new_name)) \
    gen_maps( v->send_recv_maps[send], v->send_recv_displs[send], v->send_recv_counts[send], \
              v->send_recv_maps[recv], v->send_recv_displs[recv], v->send_recv_counts[recv], \
              v->dist, v->nk, v->nb, v->k1m2, v->k1p2, index_gymnastics_##name, index_gymnastics_##name_inv )
    MAPGEN( D_to_P, P_to_D );
    MAPGEN( P_to_D, D_to_P );
    MAPGEN( D_to_C, C_to_D );
    MAPGEN( C_to_D, D_to_C );
    MAPGEN( D_to_V, V_to_D );
    MAPGEN( V_to_D, D_to_V );
    MAPGEN( C_to_V, V_to_C );
    MAPGEN( V_to_C, C_to_V );
    MAPGEN( P_to_V, V_to_P );
    MAPGEN( V_to_P, P_to_V );
    MAPGEN( D_swap_3_and_4, D_swap_3_and_4 );
    MAPGEN( D_swap_1_and_2, D_swap_1_and_2 );
    else {
        mpi_wrn_printf( "cannot find index gymnastics '%s'. skipping.\n", new_name );
        return;
    }
    strncpy( v->send_recv_descr[send], new_name, 63 ); strncat( v->send_recv_descr[send], "_send", 63 );
    strncpy( v->send_recv_descr[recv], new_name, 63 ); strncat( v->send_recv_descr[recv], "_recv", 63 );
}

void grid_vertex_exchange_symmetry( grid_vertex_t* v, double factor/*=0.5*/ ) {
    complex128_t* vertex = v->vertex_buffers[0];
    complex128_t* buf1 = v->vertex_buffers[1];
    complex128_t* buf2 = v->vertex_buffers[2];

    // buf1 <- vertex - vertex(1<->2)
    grid_vertex_overwrite_transform( v, 0, 1, "D_swap_1_and_2" );
    grid_vertex_copy( v, vertex, buf1, false, 1.0 );
    grid_vertex_project_name( v, vertex, buf1, "D_swap_1_and_2", true, -1.0 );
    grid_vertex_overwrite_transform( v, 0, 1, "D_to_P" );

    // buf2 <- buf1
    grid_vertex_copy( v, buf1, buf2, false, 1.0 );

    // buf1 <- buf1 - buf1(3<->4)
    grid_vertex_project_name( v, buf2, buf1, "D_swap_3_and_4", true, -1.0 );

    // vertex <- factor * buf1
    grid_vertex_copy( v, buf1, vertex, false, factor );
}

static void grid_vertex_gather_chan_in_v1( grid_vertex_t* v, char chan ) {
    bool create_transform = false;
    bool do_transform = false;
    char trafo_name[64] = { 'D', '_', 't', 'o', '_', chan };
    if (chan == 'P' || chan == 'V' || chan == 'C') {
        do_transform = true;
        int send_trafo = grid_vertex_find_send_trafo( v, trafo_name );
        int recv_trafo = grid_vertex_find_recv_trafo( v, trafo_name );
        create_transform = (send_trafo == -1 || recv_trafo == -1);
    } else if (chan == 'D') {
    } else {
        mpi_wrn_printf( "invalid trafo chan '%c' using D.\n", chan );
    }

    if (do_transform) {
        if (create_transform) grid_vertex_overwrite_transform( v, 0, 1, trafo_name );
        grid_vertex_project_name( v, v->vertex_buffers[0], v->vertex_buffers[1], trafo_name, false, 1.0 );
        if (create_transform) grid_vertex_overwrite_transform( v, 0, 1, "D_to_P" );
    } else {
        grid_vertex_copy( v, v->vertex_buffers[0], v->vertex_buffers[1], false, 1.0 );
    }
}

void grid_vertex_to_file( grid_vertex_t* v, const char* fname, char chan ) {
    grid_vertex_gather_chan_in_v1( v, chan );
    diverge_mpi_write_cdoub_to_file( fname, v->vertex_buffers[1], v->displs[v->myrank], v->size );
}

void grid_vertex_get_chan_allranks( grid_vertex_t* v, complex128_t* buf, char chan ) {
    grid_vertex_gather_chan_in_v1( v, chan );
    index_t rest = POW4(v->nb)*POW2(v->nk);
    ZERO_OUT( buf, rest*v->nk );

    #pragma omp parallel for num_threads(diverge_omp_num_threads()) collapse(2)
    for (index_t i=0; i<v->qcount; ++i)
    for (index_t r=0; r<rest; ++r)
        buf[IDX2(i+v->qdispl, r, rest)] = v->vertex_buffers[1][IDX2(i, r, rest)];
    diverge_mpi_allreduce_complex_sum_inplace(buf, rest*v->nk);
}
