/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once


#include "../diverge_model.h"
#include "../diverge_common.h"
#include "../diverge_internals_struct.h"
#include "../diverge_Eigen3.hpp"
#include "diverge_interface.hpp"
#include "global.hpp"
#include "cuda_init.hpp"

class Projection
{
    public:
        Projection(const diverge_model_t* const);
        ~Projection();

        const vector<array<index_t, 2>>& get_CP_PC_map_TO() const
                {return m_PC_CP_t;}
        const vector<array<index_t, 2>>& get_CD_DC_map_TO() const
                {return m_DC_CD_t;}
        const vector<array<index_t, 2>>& get_D_to_P_map_TO() const
                {return m_D_to_P_t;}
        const vector<array<index_t, 2>>& get_P_to_D_map_TO() const
                {return m_P_to_D_t;}

        const vector<index_t>& get_CP_PC_map_TO_padded() const
                {return m_PC_CP_t_padded;}
        const vector<index_t>& get_CD_DC_map_TO_padded() const
                {return m_DC_CD_t_padded;}
        const vector<index_t>& get_D_to_P_map_TO_padded() const
                {return m_D_to_P_t_padded;}
        const vector<index_t>& get_P_to_D_map_TO_padded() const
                {return m_P_to_D_t_padded;}


        const vector<index_t>& get_CP_PC_map_FROM() const
                {return m_PC_CP_f;}
        const vector<index_t>& get_CD_DC_map_FROM() const
                {return m_DC_CD_f;}
        const vector<index_t>& get_D_to_P_map_FROM() const
                {return m_D_to_P_f;}
        const vector<index_t>& get_P_to_D_map_FROM() const
                {return m_P_to_D_f;}

        const vector<index_t>& get_CP_PC_prefac_map() const
            {return m_PC_CP_mom;}
        const vector<index_t>& get_CD_DC_prefac_map() const
            {return m_DC_CD_mom;}
        const vector<index_t>& get_D_to_P_prefac_map() const
            {return m_D_to_P_mom;}
        const vector<index_t>& get_P_to_D_prefac_map() const
            {return m_P_to_D_mom;}

        const vector<index_t>& get_CP_PC_offsets() const
                {return m_PC_CP_d;}
        const vector<index_t>& get_CD_DC_offsets() const
                {return m_DC_CD_d;}
        const vector<index_t>& get_D_to_P_offsets() const
                {return m_D_to_P_d;}
        const vector<index_t>& get_P_to_D_offsets() const
                {return m_P_to_D_d;}

        const vector<index_t>& get_CP_PC_lengths() const
            {return m_PC_CP_l;}
        const vector<index_t>& get_CD_DC_lengths() const
            {return m_DC_CD_l;}
        const vector<index_t>& get_D_to_P_lengths() const
            {return m_D_to_P_l;}
        const vector<index_t>& get_P_to_D_lengths() const
            {return m_P_to_D_l;}

        index_t get_nr() const
            {return _nr;}

        const complex128_t * get_my_prefac_proj() const {return my_prefac_proj;}
        const complex128_t * get_my_prefac_proj_rk() const {return my_prefac_proj_rk;}

        complex128_t * my_prefac_reco;
        cuda_proj_descr_t proj_descr;

    private:
        void init_CP_PC_proj(const diverge_model_t* const);
        void init_CD_DC_proj(const diverge_model_t* const);
        void init_D_to_P_proj(const diverge_model_t* const);
        void init_P_to_D_proj(const diverge_model_t* const);

        void mpi_communicate();

        void init_to_mom_map_existing_r(const diverge_model_t* model);

        void init_prefac_vector(const diverge_model_t* const);

        complex128_t * my_prefac_proj;
        complex128_t * my_prefac_proj_rk;


        vector<array<index_t, 2>> m_PC_CP_t;
        vector<index_t> m_PC_CP_f;
        vector<index_t> m_PC_CP_t_padded;
        vector<index_t> m_PC_CP_mom;
        vector<index_t> m_PC_CP_d;
        vector<index_t> m_PC_CP_l;

        vector<array<index_t, 2>> m_DC_CD_t;
        vector<index_t> m_DC_CD_f;
        vector<index_t> m_DC_CD_t_padded;
        vector<index_t> m_DC_CD_mom;
        vector<index_t> m_DC_CD_d;
        vector<index_t> m_DC_CD_l;

        vector<array<index_t, 2>> m_D_to_P_t;
        vector<index_t> m_D_to_P_f;
        vector<index_t> m_D_to_P_t_padded;
        vector<index_t> m_D_to_P_mom;
        vector<index_t> m_D_to_P_d;
        vector<index_t> m_D_to_P_l;

        vector<array<index_t, 2>> m_P_to_D_t;
        vector<index_t> m_P_to_D_f;
        vector<index_t> m_P_to_D_t_padded;
        vector<index_t> m_P_to_D_mom;
        vector<index_t> m_P_to_D_d;
        vector<index_t> m_P_to_D_l;

        index_t _nr;
        const index_t n_orb;
        const index_t n_bonds;
        const index_t n_orbff;
        const index_t* bond_sizes;
        const index_t* bond_offsets;
        const index_t* ob_to_orbff;
        const tu_formfactor_t* tu_ff;

        vector<Vec3i> existing_r;
        vector<index_t> PC_r;
        vector<index_t> DC_r;
        vector<index_t> DP_r;
        vector<index_t> PD_r;
};
