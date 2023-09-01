#include "projection_handler.hpp"
#include "cuda_init.hpp"

inline void add_to_all_frq(complex128_t* const to, const complex128_t* const inc,
                           index_t nb, index_t rest) {
    #pragma omp parallel for collapse(2) schedule(static)
    for(index_t i = 0; i < nb; ++i)
    for(index_t r = 0; r < rest; ++r) {
        to[r+rest*i] += inc[r];
    }
}


void P_projection(complex128_t* proj_vertex, Vertex* const vertex,
                const Projection* const proj) {
    if(!vertex->P_flow) return;
    mcopy(proj_vertex, vertex->Pch, vertex->full_vert_size);
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;

    #ifdef USE_CUDA
    if(vertex->my_nk > (index_t)tu_num_of_devices) {
        if(vertex->C_flow) {
            vertex->template reco_ft<'C'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
            mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));

            GPUproj_handler_ft(&(proj->proj_descr),'P',bufMPI, buf, proj->get_CP_PC_map_TO_padded(),
                proj->get_CP_PC_map_FROM(), proj->get_CP_PC_prefac_map(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->my_nk,
                proj->get_nr());

            add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
                vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        }
        if(vertex->D_flow) {
            vertex->template reco_ft<'D'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
            mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));

            GPUproj_handler_ft(&(proj->proj_descr),'P',bufMPI, buf,
                proj->get_D_to_P_map_TO_padded(),
                proj->get_D_to_P_map_FROM(), proj->get_D_to_P_prefac_map(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->my_nk,
                proj->get_nr());

            add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
                    vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        }
        return ;
    }
    #endif // USE_CUDA
    if(vertex->C_flow) {
        vertex->template reco_ft<'C'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
        mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        _to_P(bufMPI, buf, proj->get_CP_PC_map_TO(),
            proj->get_CP_PC_map_FROM(), proj->get_CP_PC_prefac_map(),
            proj->get_CP_PC_offsets(), proj->get_CP_PC_lengths(),
            proj->get_my_prefac_proj(), vertex->n_orbff,
            vertex->n_spin,vertex->nk,vertex->my_nk,
            proj->get_nr());

        add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
               vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
    }
    if(vertex->D_flow) {
        vertex->template reco_ft<'D'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
        mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        _to_P(bufMPI, buf, proj->get_D_to_P_map_TO(),
            proj->get_D_to_P_map_FROM(), proj->get_D_to_P_prefac_map(),
            proj->get_D_to_P_offsets(), proj->get_D_to_P_lengths(),
            proj->get_my_prefac_proj(), vertex->n_orbff,
            vertex->n_spin,vertex->nk, vertex->my_nk,
            proj->get_nr());
        add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
            vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
    }
}


void D_projection(complex128_t* proj_vertex,
                    Vertex* const vertex,
                    const Projection* const proj) {
    if(!vertex->D_flow) return;
    mcopy(proj_vertex, vertex->Dch, vertex->full_vert_size);
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    #ifdef USE_CUDA
    if(vertex->my_nk > (index_t)tu_num_of_devices)
    {
        if(vertex->C_flow) {
            vertex->template reco_ft<'C'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
            mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
            GPUproj_handler_ft(&(proj->proj_descr),'D',bufMPI, buf,
                    proj->get_CD_DC_map_TO_padded(),
                    proj->get_CD_DC_map_FROM(), proj->get_CD_DC_prefac_map(),
                    proj->get_my_prefac_proj(), vertex->n_orbff,
                    vertex->n_spin,vertex->my_nk,
                    proj->get_nr());
            add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
                       vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        }
        if(vertex->P_flow) {
            vertex->template reco_ft<'P'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
            mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
            GPUproj_handler_ft(&(proj->proj_descr),'D',bufMPI, buf,
                    proj->get_P_to_D_map_TO_padded(),
                    proj->get_P_to_D_map_FROM(), proj->get_P_to_D_prefac_map(),
                    proj->get_my_prefac_proj(), vertex->n_orbff,
                    vertex->n_spin,vertex->my_nk,
                    proj->get_nr());
             add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
                        vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        }
        return ;
    }
    #endif // USE_CUDA
    if(vertex->C_flow) {
        vertex->template reco_ft<'C'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
        mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        _to_D(bufMPI,buf,proj->get_CD_DC_map_TO(),
                proj->get_CD_DC_map_FROM(), proj->get_CD_DC_prefac_map(),
                proj->get_CD_DC_offsets(), proj->get_CD_DC_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk,vertex->my_nk,
                proj->get_nr());
        add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
    }
    if(vertex->P_flow) {
        vertex->template reco_ft<'P'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
        mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        _to_D(bufMPI,buf,proj->get_P_to_D_map_TO(),
                proj->get_P_to_D_map_FROM(), proj->get_P_to_D_prefac_map(),
                proj->get_P_to_D_offsets(), proj->get_P_to_D_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk,vertex->my_nk,
                proj->get_nr());
        add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
    }
}


void C_projection(complex128_t* proj_vertex,
                    Vertex* const vertex,
                    const Projection* const proj) {
    if(!vertex->C_flow) return;
    mcopy(proj_vertex, vertex->Cch, vertex->full_vert_size);
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
        
    #ifdef USE_CUDA
    if(vertex->my_nk > (index_t)tu_num_of_devices)
    {
        if(vertex->P_flow) {
            vertex->template reco_ft<'P'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
            mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
            GPUproj_handler_ft(&(proj->proj_descr),'C',bufMPI, buf,
                    proj->get_CP_PC_map_TO_padded(),
                    proj->get_CP_PC_map_FROM(), proj->get_CP_PC_prefac_map(),
                    proj->get_my_prefac_proj(), vertex->n_orbff,
                    vertex->n_spin,vertex->my_nk,
                    proj->get_nr());
            add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
                       vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        }
        if(vertex->D_flow) {
            vertex->template reco_ft<'D'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
            mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
            
            GPUproj_handler_ft(&(proj->proj_descr),'C',bufMPI, buf, proj->get_CD_DC_map_TO_padded(),
                        proj->get_CD_DC_map_FROM(), proj->get_CD_DC_prefac_map(),
                        proj->get_my_prefac_proj(), vertex->n_orbff,
                        vertex->n_spin, vertex->my_nk,
                        proj->get_nr());
            add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
                       vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        }
        return ;
    }
    #endif // USE_CUDA

    if(vertex->D_flow) {
        vertex->template reco_ft<'D'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
        mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        _to_C(bufMPI,buf, proj->get_CD_DC_map_TO(),
                proj->get_CD_DC_map_FROM(), proj->get_CD_DC_prefac_map(),
                proj->get_CD_DC_offsets(), proj->get_CD_DC_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk,vertex->my_nk,
                proj->get_nr());
        add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
    }
    if(vertex->P_flow) {
        vertex->template reco_ft<'P'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
        mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        _to_C(bufMPI,buf, proj->get_CP_PC_map_TO(),
                proj->get_CP_PC_map_FROM(), proj->get_CP_PC_prefac_map(),
                proj->get_CP_PC_offsets(), proj->get_CP_PC_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk, vertex->my_nk,
                proj->get_nr());
        add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
    }
}


void P_projection_no_GPU(complex128_t* proj_vertex,
                    Vertex* const vertex,const Projection* const proj) {
    if(!vertex->P_flow) return;
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    mcopy(proj_vertex, vertex->Pch, vertex->full_vert_size);
    if(vertex->C_flow) {
        vertex->template reco_ft<'C'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
        mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        _to_P(bufMPI, buf, proj->get_CP_PC_map_TO(),
            proj->get_CP_PC_map_FROM(), proj->get_CP_PC_prefac_map(),
            proj->get_CP_PC_offsets(), proj->get_CP_PC_lengths(),
            proj->get_my_prefac_proj(), vertex->n_orbff,
            vertex->n_spin,vertex->nk,vertex->my_nk,
            proj->get_nr());

        add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
               vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
    }
    if(vertex->D_flow) {
        vertex->template reco_ft<'D'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
        mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        _to_P(bufMPI, buf, proj->get_D_to_P_map_TO(),
            proj->get_D_to_P_map_FROM(), proj->get_D_to_P_prefac_map(),
            proj->get_D_to_P_offsets(), proj->get_D_to_P_lengths(),
            proj->get_my_prefac_proj(), vertex->n_orbff,
            vertex->n_spin,vertex->nk, vertex->my_nk,
            proj->get_nr());
        add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
            vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
    }
}


void D_projection_no_GPU(complex128_t* proj_vertex,
                    Vertex* const vertex,
                    const Projection* const proj) {
    if(!vertex->D_flow) return;
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    mcopy(proj_vertex, vertex->Dch, vertex->full_vert_size);
    if(vertex->C_flow) {
        vertex->template reco_ft<'C'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
        mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        _to_D(bufMPI,buf,proj->get_CD_DC_map_TO(),
                proj->get_CD_DC_map_FROM(), proj->get_CD_DC_prefac_map(),
                proj->get_CD_DC_offsets(), proj->get_CD_DC_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk,vertex->my_nk,
                proj->get_nr());
        add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
    }
    if(vertex->P_flow) {
        vertex->template reco_ft<'P'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
        mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        _to_D(bufMPI,buf,proj->get_P_to_D_map_TO(),
                proj->get_P_to_D_map_FROM(), proj->get_P_to_D_prefac_map(),
                proj->get_P_to_D_offsets(), proj->get_P_to_D_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk,vertex->my_nk,
                proj->get_nr());
        add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
    }
}


void C_projection_no_GPU(complex128_t* proj_vertex,
                    Vertex* const vertex,
                    const Projection* const proj) {
    if(!vertex->C_flow) return;
    complex128_t* buf = vertex->product_helper;
    complex128_t* bufMPI = vertex->bubble_helper;
    mcopy(proj_vertex, vertex->Cch, vertex->full_vert_size);
    if(vertex->D_flow) {
        vertex->template reco_ft<'D'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
        mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        _to_C(bufMPI,buf, proj->get_CD_DC_map_TO(),
                proj->get_CD_DC_map_FROM(), proj->get_CD_DC_prefac_map(),
                proj->get_CD_DC_offsets(), proj->get_CD_DC_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk,vertex->my_nk,
                proj->get_nr());
        add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
    }
    if(vertex->P_flow) {
        vertex->template reco_ft<'P'>(buf,proj->my_prefac_reco,proj->get_nr(),bufMPI);
        mfill_0(bufMPI,vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
        _to_C(bufMPI,buf, proj->get_CP_PC_map_TO(),
                proj->get_CP_PC_map_FROM(), proj->get_CP_PC_prefac_map(),
                proj->get_CP_PC_offsets(), proj->get_CP_PC_lengths(),
                proj->get_my_prefac_proj(), vertex->n_orbff,
                vertex->n_spin,vertex->nk, vertex->my_nk,
                proj->get_nr());
        add_to_all_frq(proj_vertex, bufMPI,vertex->n_frq,
                 vertex->my_nk*POW2(vertex->n_orbff*POW2(vertex->n_spin)));
    }
}
