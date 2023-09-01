#pragma once

#include "typedefs.hpp"

namespace grid {

class IndexGymnastics {
    public:
        // k3map(k1,k2,k3) = k1 + k2 - k3
        IndexGymnastics( const index_t* k1m2, const index_t* k1p2, const index_t* k3map = nullptr );

        index_t D_to_P( index_t idx, index_t nk, index_t nb ) const;
        index_t P_to_D( index_t idx, index_t nk, index_t nb ) const;
        index_t D_to_C( index_t idx, index_t nk, index_t nb ) const;
        index_t C_to_D( index_t idx, index_t nk, index_t nb ) const;

        index_t D_swap_3_and_4( index_t idx, index_t nk, index_t nb ) const;
        index_t D_swap_1_and_2( index_t idx, index_t nk, index_t nb ) const;

        index_t V_to_D( index_t idx, index_t nk, index_t nb ) const;
        index_t D_to_V( index_t idx, index_t nk, index_t nb ) const;

        index_t V_to_P( index_t idx, index_t nk, index_t nb ) const;
        index_t V_to_C( index_t idx, index_t nk, index_t nb ) const;
        index_t P_to_V( index_t idx, index_t nk, index_t nb ) const;
        index_t C_to_V( index_t idx, index_t nk, index_t nb ) const;

        const index_t* get_k1m2() const { return _k1m2; }
        const index_t* get_k1p2() const { return _k1p2; }
        const index_t* get_k3map() const { return _k3map; }
    private:
        const index_t* _k1m2;
        const index_t* _k1p2;
        const index_t* _k3map;
        void _unravel_V( index_t idx, index_t nk, index_t nb, array<index_t,7>& IDX ) const;
        void _unravel_I( index_t idx, index_t nk, index_t nb, array<index_t,7>& IDX ) const;
        index_t _ravel_V( index_t nk, index_t nb, const array<index_t,7>& IDX ) const;
        index_t _ravel_I( index_t nk, index_t nb, const array<index_t,7>& IDX ) const;

        index_t V_swap_1_and_2( index_t idx, index_t nk, index_t nb ) const;
        index_t V_swap_3_and_4( index_t idx, index_t nk, index_t nb ) const;
};

#define InGyFun( obj, name ) [&obj](index_t i, index_t nk, index_t nb){ return obj . name (i,nk,nb); }
#define InGyFunPtr( obj, name ) [&](index_t i, index_t nk, index_t nb){ return obj -> name (i,nk,nb); }

}
