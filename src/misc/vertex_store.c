#include "vertex_store.h"
#include "mpi_functions.h"

tu_highsym_vertex_t* tu_highsym_vertex_store( diverge_flow_step_t* step,
        index_t* qpts, index_t nqpts, tu_highsym_vertex_t* vert ) {
    diverge_flow_step_vertex_t vP = diverge_flow_step_vertex( step, 'P' );
    diverge_flow_step_vertex_t vC = diverge_flow_step_vertex( step, 'C' );
    diverge_flow_step_vertex_t vD = diverge_flow_step_vertex( step, 'D' );
    diverge_flow_step_vertex_t vs[] = {vP, vC, vD};

    if (!vert) {
        vert = calloc(1, sizeof(tu_highsym_vertex_t));
        vert->stepsz = nqpts * (POW2(vP.n_orbff) + POW2(vC.n_orbff) + POW2(vD.n_orbff));
        vert->memory_cap = vert->stepsz;
        vert->memory = calloc(vert->memory_cap, sizeof(complex128_t));

        vert->nqpts = nqpts;
        vert->n_orbff_P = vP.n_orbff;
        vert->n_orbff_C = vC.n_orbff;
        vert->n_orbff_D = vD.n_orbff;
    }

    if (vert->memory_cap < (++vert->nsteps)*vert->stepsz)
        vert->memory = realloc(vert->memory, sizeof(complex128_t)*(vert->memory_cap *= 2));

    complex128_t* vertices = vert->memory + vert->memory_idx;
    for (int chans=0; chans<3; ++chans)
    for (index_t i=0; i<nqpts; ++i) {
        index_t sz = POW2(vs[chans].n_orbff);
        memcpy( vertices, vs[chans].ary + qpts[i]*sz, sizeof(complex128_t)*sz );
        vertices += sz;
    }
    vert->memory_idx += vert->stepsz;

    return vert;
}

void tu_highsym_vertex_to_file( tu_highsym_vertex_t* vert, const char* fname ) {
    index_t header[128] = {0};
    char magic[] = "tu_highsym_vertex_t";
    int nmagic = sizeof(magic) / sizeof(header[0]) + 1;
    memcpy( header, magic, sizeof(magic) );
    index_t iheader = nmagic;

    header[iheader++] = vert->nsteps;
    header[iheader++] = vert->stepsz;
    header[iheader++] = vert->nqpts;
    header[iheader++] = vert->n_orbff_P;
    header[iheader++] = vert->n_orbff_C;
    header[iheader++] = vert->n_orbff_D;

    index_t displ = sizeof(index_t)*128;

    header[iheader++] = displ;
    displ += (header[iheader++] = sizeof(complex128_t) * vert->nsteps * vert->stepsz);

    if (vert->qpts) {
        header[iheader++] = displ;
        displ += (header[iheader++] = sizeof(index_t) * vert->nqpts);
    }
    if (vert->qpts_descr_lens) {
        header[iheader++] = displ;
        displ += (header[iheader++] = sizeof(index_t) * vert->n_qpts_descr);
    }
    if (vert->qpts_descr_nams) {
        header[iheader++] = displ;
        displ += (header[iheader++] = sizeof(char) * vert->n_qpts_descr);
    }

    FILE* f = diverge_mpi_comm_rank() == 0 ? fopen(fname, "wb") : NULL;
    if (!f) {
        mpi_err_printf( "could not open file %s\n", fname );
        return;
    }

    fwrite( header, 128, sizeof(index_t), f );
    fwrite( vert->memory, vert->nsteps*vert->stepsz, sizeof(complex128_t), f );
    if (vert->qpts) fwrite( vert->qpts, vert->nqpts, sizeof(index_t), f );
    if (vert->qpts_descr_lens) fwrite( vert->qpts_descr_lens, vert->n_qpts_descr, sizeof(index_t), f );
    if (vert->qpts_descr_nams) fwrite( vert->qpts_descr_nams, vert->n_qpts_descr, sizeof(char), f );
    fclose( f );
}

void tu_highsym_vertex_free( tu_highsym_vertex_t* vert ) {
    if (!vert) return;
    free( vert->memory );
    free( vert );
}
