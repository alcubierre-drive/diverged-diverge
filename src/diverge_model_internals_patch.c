/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_model_internals.h"
#include "diverge_momentum_gen.h"
#include "diverge_internals_struct.h"
#include "npatch/vertex.h"
#include "npatch/flow_step.h"
#include "diverge_patching_helpers.h"
#include "diverge_patching.h"
#include "misc/mpi_functions.h"

static void patch_vertex_destructor( void* vvert ) {
    npatch_vertex_t* vert = (npatch_vertex_t*)vvert;
    npatch_vertex_free( vert );
}

static void add_to_npatch_vert_from_qbuf( complex128_t* vbuf,
    const complex128_t* qbuf_in, const index_t* patches, const index_t* k3map,
    index_t np, index_t no, index_t ns, const index_t nk[3], char chan );

void diverge_model_internals_patch( diverge_model_t* model, index_t np_ibz ) {
    int errors = 0;
    if (model->internals->dim > 2) {
        mpi_err_printf("no support for $N$-patch FRG in more than 2D\n");
        errors++;
    }
    if (kdim(model->nkf) > 1) {
        mpi_err_printf("no support for refined mesh in $N$-patch FRG\n");
        errors++;
    }

    if (errors) {
        mpi_err_printf("exiting patch setup due to previous errors\n");
        return;
    }

    model->internals->patch_gpu_workshare = 1.0;
    model->internals->patch_gpu_loop_workshare = 1.0;
    model->internals->patch_gpus = NULL;
    model->internals->patch_nblocks = NULL;
    model->internals->patch_ngpus = -1;

    model->internals->patch_gpu_weights = NULL;
    model->internals->patch_loop_gpu_weights = NULL;

    if (model->patching == NULL) {
        mpi_vrb_printf("continuing with autopatch and np_ibz = %li\n", np_ibz);
        index_t *pts, npts;
        diverge_patching_find_fs_pts_C( model, NULL, model->n_orb * model->n_spin, np_ibz, np_ibz > 100 ? np_ibz : 100, &pts, &npts );
        mom_patching_t* patch = diverge_patching_from_indices( model, pts, npts );
        free(pts);
        diverge_patching_autofine( model, patch, NULL, model->n_orb * model->n_spin, 40, 0.1, 2.0, 0.5 );
        diverge_patching_symmetrize_refinement( model, patch );
        model->patching = patch;
    } else {
        mpi_vrb_printf("discarding np_ibz parameter in patch internals\n");
    }

    double kbasis[3][3];
    diverge_model_generate_mom_basis(model->lattice, kbasis);

    memcpy( &model->internals->patch_g1, &(kbasis[0]), sizeof(double)*2 );
    memcpy( &model->internals->patch_g2, &(kbasis[1]), sizeof(double)*2 );

    model->internals->patch_nkx = model->nk[0];
    model->internals->patch_nky = model->nk[1];
    model->internals->patch_nk = model->nk[0] * model->nk[1];
    model->internals->patch_nb = model->n_orb * model->n_spin;

    index_t np = model->patching->n_patches;
    model->internals->patch_pmesh = (complex128_t*)malloc(sizeof(complex128_t)*np);
    complex128_t* pmesh = model->internals->patch_pmesh;
    index_t* patches = model->patching->patches;
    kpoint_union_t* kun = (kpoint_union_t*) model->internals->kmesh;
    for (index_t p=0; p<np; ++p)
        pmesh[p] = kun[patches[p]].k;

    model->internals->patch_kmesh = (complex128_t*)malloc(sizeof(complex128_t)*model->internals->patch_nk);
    complex128_t* pkmesh = model->internals->patch_kmesh;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t k=0; k<model->internals->patch_nk; ++k)
        pkmesh[k] = kun[k].k;

    model->internals->patch_k3map = (index_t*)malloc(sizeof(index_t)*POW3(np));
    index_t* k3map = model->internals->patch_k3map;
    complex128_t g1 = model->internals->patch_g1,
                 g2 = model->internals->patch_g2;
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for (index_t k1=0; k1<np; ++k1)
    for (index_t k2=0; k2<np; ++k2)
    for (index_t k3=0; k3<np; ++k3)
        k3map[(k1*np+k2)*np+k3] = find_closest_patch( pmesh[k1] + pmesh[k2] - pmesh[k3], pmesh, np, g1, g2 );

    npatch_vertex_t* V = npatch_vertex_init( model );
    complex128_t* vbuf = V->V;
    memset( vbuf, 0, sizeof(complex128_t)*V->size );
    if (model->ffill != NULL) {
        mpi_vrb_printf("using full vertex generator\n");
        const index_t no4 = POW4(model->internals->patch_nb);
        #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
        for (index_t p1=0; p1<np; ++p1)
        for (index_t p2=0; p2<np; ++p2)
        for (index_t p3=0; p3<np; ++p3)
            (*(model->ffill))( model, patches[p1], patches[p2], patches[p3],
                    vbuf + IDX3(p1, p2, p3, np, np)*no4 );
    } else {
        mpi_vrb_printf("using channel vertex generators\n");
        complex128_t* qbuf_in = (complex128_t*)calloc(sizeof(complex128_t),
                model->internals->patch_nk * POW2(POW2(model->n_spin) * model->n_orb));
        char chans[] = "CPD";
        for (char *pchan = chans; *pchan; ++pchan)
            if ((*(model->vfill))( model, *pchan, qbuf_in ))
                add_to_npatch_vert_from_qbuf( vbuf, qbuf_in, patches, k3map,
                        np, model->n_orb, model->n_spin, model->nk, *pchan );
        free(qbuf_in);
    }

    if (model->internals->enforce_exchange) {
        mpi_vrb_printf( "enforcing exchange symmetry of initial vertex\n" );
        npatch_flow_step_enforce_exchange( vbuf, NULL, k3map, np, model->internals->patch_nb, 0.5 );
    }
    model->internals->patch_vertex = V;
    model->internals->patch_vertex_destructor = &patch_vertex_destructor;
}

static void add_to_npatch_vert_from_qbuf( complex128_t* vbuf,
    const complex128_t* qbuf_in, const index_t* patches, const index_t* k3map,
    index_t np, index_t no, index_t ns, const index_t nk[3], char chan ) {

    const index_t nb = no * ns;
    #pragma omp parallel for collapse(3) num_threads(diverge_omp_num_threads())
    for (index_t p1=0; p1<np; ++p1)
    for (index_t p2=0; p2<np; ++p2)
    for (index_t p3=0; p3<np; ++p3) {
        index_t k1 = patches[p1],
                k2 = patches[p2],
                k3 = patches[p3],
                k4 = patches[k3map[IDX3(p1, p2, p3, np, np)]];
        index_t qq_1 = 0, qq_2 = 0;
        switch (chan) {
            case 'P': qq_1 = k1p2(k1, k2, nk); qq_2 = k1p2(k3, k4, nk); break;
            case 'C': qq_1 = k1m2(k3, k2, nk); qq_2 = k1m2(k1, k4, nk); break;
            case 'D': qq_1 = k1m2(k1, k3, nk); qq_2 = k1m2(k4, k2, nk); break;
            default: break;
        }
        for (index_t OA=0; OA<no; ++OA)
        for (index_t OB=0; OB<no; ++OB) {
            index_t o1 = 0, o2 = 0, o3 = 0, o4 = 0;
            switch (chan) {
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
                switch (chan) {
                    case 'P': s1 = SA; s2 = SB; s3 = SC; s4 = SD; break;
                    case 'C': s1 = SB; s2 = SC; s3 = SD; s4 = SA; break;
                    case 'D': s1 = SB; s2 = SC; s3 = SA; s4 = SD; break;
                    default: break;
                }
                complex128_t vadd = 0.5 * qbuf_in[IDX7( qq_1, SD, SC, OB, SB, SA, OA, ns, ns, no, ns, ns, no )] +
                                    0.5 * qbuf_in[IDX7( qq_2, SD, SC, OB, SB, SA, OA, ns, ns, no, ns, ns, no )];
                index_t b1 = IDX2(s1, o1, no),
                        b2 = IDX2(s2, o2, no),
                        b3 = IDX2(s3, o3, no),
                        b4 = IDX2(s4, o4, no);

                atomic_add( vbuf + IDX7( p1, p2, p3, b1, b2, b3, b4, np, np, nb, nb, nb, nb ), &vadd );
            }
        }
    }

}
