#include "vertex_store.h"
#include <assert.h>
#include "../tu/vertex_details.h"
#include "../diverge_flow_step_internal.h"
#include "mpi_functions.h"
#include "qnan.h"
#include "version.h"
#include "number_repr_checks.h"

tu_vertex_storage_t* tu_vertex_storage_init_fg( diverge_flow_step_t* s, char chan,
        const tu_vertex_storage_qpt_t* qpts, index_t n_qpts ) {
    if (diverge_flow_step_type(s) != diverge_tu) {
        mpi_err_printf( "cannot store vertices of non-TU backend\n" );
        return NULL;
    }
    assert(n_qpts >= 0);

    tu_vertex_storage_t* t = calloc(1, sizeof*t);
    t->s = s;
    tu_vertex_t* tu_V = s->tu_vertex;
    t->linsz = POW2(tu_V->n_spin) * tu_V->n_orbff;
    t->chan = chan;
    if (toupper(chan) != 'P' && toupper(chan) != 'C' && toupper(chan) != 'D' && chan != 'm' && chan != 'h') {
        mpi_wrn_printf("for TUFRG vertex chan must be one of 'PCDpcdmh'. default: 'D'.\n");
        t->chan = chan = 'D';
    }
    t->n_qpts = n_qpts;
    t->qpts = calloc(t->n_qpts, sizeof*t->qpts);
    for (index_t iq=0; iq<t->n_qpts; ++iq) t->qpts[iq] = qpts[iq];

    index_t q_0 = tu_V->my_nk_off,
            q_1 = tu_V->my_nk_off + tu_V->my_nk;

    // find which rank each qpt is on
    t->qpts_ranks = calloc( t->n_qpts, sizeof*t->qpts_ranks );
    const int rank = diverge_mpi_comm_rank();
    for (index_t iq=0; iq<t->n_qpts; ++iq) {
        index_t qq = t->qpts[iq].qpt;
        t->qpts_ranks[iq] = (q_1 > qq && q_0 <= qq) ? rank : -1;
    }
    diverge_mpi_allreduce_int_max_inplace( t->qpts_ranks, t->n_qpts );

    for (index_t iq=0; iq<t->n_qpts; ++iq) {
        if (t->qpts_ranks[iq] == -1) {
            mpi_err_printf( "qpt #%li (%li) not found, reset to 0\n", iq, t->qpts[iq].qpt );
            t->qpts_ranks[iq] = 0;
            t->qpts[iq].qpt = 0;
            strcpy( t->qpts[iq].name, "ERROR" );
        }
    }

    return t;
}

tu_vertex_storage_t* tu_vertex_storage_init( diverge_flow_step_t* s, char chan,
        const index_t* qpts, index_t n_qpts ) {
    assert(n_qpts >= 0);
    tu_vertex_storage_qpt_t* qpts_fg = calloc(n_qpts, sizeof*qpts_fg);
    for (index_t q=0; q<n_qpts; ++q) {
        qpts_fg[q].qpt = qpts[q];
        snprintf( qpts_fg[q].name, 23, "%li", qpts[q] );
    }
    tu_vertex_storage_t* t = tu_vertex_storage_init_fg( s, chan, qpts_fg, n_qpts );
    free( qpts_fg );
    return t;
}

void tu_vertex_storage_lsnap( tu_vertex_storage_t* t, double Lambda ) {
    assert(t != NULL);
    const int rank = diverge_mpi_comm_rank();
    const index_t l2 = POW2(t->linsz);
    diverge_flow_step_vertex_t v = diverge_flow_step_vertex( t->s, t->chan );

    for (index_t iq=0; iq<t->n_qpts; ++iq) {
        index_t qpt_rel = t->qpts[iq].qpt - v.q_0;
        index_t offset = -1;
        if (rank == 0) { // add some memory when we're on rank 0
            offset = t->memory_sz;
            VECTOR_RESIZE( t->memory, offset + l2 );
        }
        if (t->qpts_ranks[iq] != 0) {
            if (rank == 0) diverge_mpi_recv_double( t->memory+offset, l2*2, t->qpts_ranks[iq], t->qpts[iq].qpt );
            else           diverge_mpi_send_double( v.ary+qpt_rel*l2, l2*2, 0, t->qpts[iq].qpt );
        } else {
            if (rank == 0) memcpy( t->memory+offset, v.ary+qpt_rel*l2, sizeof(complex128_t)*l2 );
        }
    }
    t->n_steps++;
    VECTOR_PUSH_BACK( t->Lambda, Lambda );
    VECTOR_PUSH_BACK( t->chans, t->chan );
}

void tu_vertex_storage_lsnap_chan( tu_vertex_storage_t* t, double Lambda, char chan ) {
    char old_chan = t->chan;
    t->chan = chan;
    tu_vertex_storage_lsnap( t, Lambda );
    t->chan = old_chan;
}
void tu_vertex_storage_snap( tu_vertex_storage_t* t ) {
    tu_vertex_storage_lsnap( t, qnan_gen(0) );
}
void tu_vertex_storage_snap_chan( tu_vertex_storage_t* t, char chan ) {
    tu_vertex_storage_lsnap_chan( t, qnan_gen(0), chan );
}

complex128_t* tu_vertex_storage_get( tu_vertex_storage_t* t, index_t istep, index_t iq ) {
    if (diverge_mpi_comm_rank() != 0)
        return NULL;
    return t->memory + (istep*t->n_qpts + iq) * POW2(t->linsz);
}

void tu_vertex_storage_pop( tu_vertex_storage_t* t ) {
    assert(t != NULL);
    if (t->n_steps <= 0) return;
    t->n_steps--;
    t->memory_sz -= t->n_qpts*POW2(t->linsz);
    t->Lambda_sz--;
    t->chans_sz--;
}

void tu_vertex_storage_to_file( tu_vertex_storage_t* t, const char* fname ) {
    if (!t) {
        mpi_wrn_printf( "vertex store object points to NULL, not saving to file %s\n", fname );
        return;
    }
    if (diverge_mpi_comm_rank() != 0) return;

    index_t header[128] = {0};
    char magic[] = "tu_vertex_storage_t";
    int nmagic = sizeof(magic) / sizeof(header[0]) + 1;
    memcpy( header, magic, sizeof(magic) );
    index_t iheader = nmagic;

    header[iheader++] = t->chan;
    header[iheader++] = t->linsz;
    header[iheader++] = t->n_steps;
    header[iheader++] = t->n_qpts;

    index_t displ = sizeof header;

    header[iheader++] = displ;
    displ += (header[iheader++] = sizeof(tu_vertex_storage_qpt_t) * t->n_qpts);

    header[iheader++] = displ;
    displ += (header[iheader++] = sizeof(complex128_t) * t->n_steps * t->n_qpts * POW2(t->linsz));

    header[iheader++] = displ;
    displ += (header[iheader++] = sizeof(double) * t->n_steps);

    header[iheader++] = displ;
    displ += (header[iheader++] = sizeof(char) * t->n_steps);

    // file format version
    iheader = 125;
    const char* version = tag_version();
    header[iheader++] = displ;
    displ += (header[iheader++] = (strlen(version)+1));

    // health check bit field
    uindex_t checkmask = 0;
    uint8_t one = 1;
    uindex_t firstbit = checkmask | (one << 7);
    if (!check_32bit_float())   checkmask |= firstbit >> 0;
    if (!check_64bit_double())  checkmask |= firstbit >> 1;
    if (!check_little_endian()) checkmask |= firstbit >> 2;
    if (!check_8bit_char())     checkmask |= firstbit >> 3;
    uindex_t* uheader = (uindex_t*)header;
    uheader[iheader++] = checkmask;

    FILE* f = fopen(fname, "wb");
    if (!f) {
        mpi_err_printf( "could not open file %s\n", fname );
        return;
    }

    fwrite( header, sizeof header, 1, f );
    fwrite( t->qpts, sizeof*t->qpts, t->n_qpts, f );
    fwrite( t->memory, sizeof*t->memory, t->n_steps*t->n_qpts*POW2(t->linsz), f );
    fwrite( t->Lambda, sizeof*t->Lambda, t->n_steps, f );
    fwrite( t->chans, sizeof*t->chans, t->n_steps, f );
    fwrite( version, 1, strlen(version)+1, f );
    fclose( f );
}

void tu_vertex_storage_free( tu_vertex_storage_t* t ) {
    if (t) {
        free( t->qpts );
        free( t->memory );
        free( t->Lambda );
        free( t->chans );
        free( t->qpts_ranks );
        free( t );
    }
}

