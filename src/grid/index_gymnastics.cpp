#include "index_gymnastics.hpp"

namespace grid {

index_t IndexGymnastics::V_to_P( index_t idx, index_t nk, index_t nb ) const {
    array<index_t,7> IDX;
    _unravel_V( idx, nk, nb, IDX );
    index_t k1 = IDX[0]; index_t k2 = IDX[1]; index_t k3 = IDX[2];
    index_t b1 = IDX[3]; index_t b2 = IDX[4]; index_t b3 = IDX[5]; index_t b4 = IDX[6];
    IDX = array<index_t,7>{ _k1p2[nk*k2+k1], k1, b1, b2, k3, b3, b4 };
    return _ravel_I( nk, nb, IDX );
}

index_t IndexGymnastics::V_to_C( index_t idx, index_t nk, index_t nb ) const {
    array<index_t,7> IDX;
    _unravel_V( idx, nk, nb, IDX );
    index_t k1 = IDX[0]; index_t k2 = IDX[1]; index_t k3 = IDX[2];
    index_t b1 = IDX[3]; index_t b2 = IDX[4]; index_t b3 = IDX[5]; index_t b4 = IDX[6];
    IDX = array<index_t,7>{ _k1m2[nk*k3+k2], k1, b4, b1, k3, b2, b3 };
    return _ravel_I( nk, nb, IDX );
}

index_t IndexGymnastics::V_to_D( index_t idx, index_t nk, index_t nb ) const {
    array<index_t,7> IDX;
    _unravel_V( idx, nk, nb, IDX );
    index_t k1 = IDX[0]; index_t k2 = IDX[1]; index_t k3 = IDX[2];
    index_t b1 = IDX[3]; index_t b2 = IDX[4]; index_t b3 = IDX[5]; index_t b4 = IDX[6];
    if (_k3map)
        IDX = array<index_t,7>{ _k1m2[nk*k1+k3], k1, b3, b1, _k3map[nk*nk*k1 + nk*k2 + k3], b2, b4 };
    else
        IDX = array<index_t,7>{ _k1m2[nk*k1+k3], k1, b3, b1, _k1p2[nk*_k1m2[nk*k2+k3]+k1], b2, b4 };
    return _ravel_I( nk, nb, IDX );
}

index_t IndexGymnastics::P_to_V( index_t idx, index_t nk, index_t nb ) const {
    array<index_t,7> IDX;
    _unravel_I( idx, nk, nb, IDX );
    index_t q = IDX[0]; index_t k  = IDX[1]; index_t b1 = IDX[2]; index_t b2 = IDX[3];
                      index_t kp = IDX[4]; index_t b3 = IDX[5]; index_t b4 = IDX[6];
    IDX = array<index_t,7>{ k, _k1m2[nk*q+k], kp, b1, b2, b3, b4 };
    return _ravel_V( nk, nb, IDX );
}

index_t IndexGymnastics::C_to_V( index_t idx, index_t nk, index_t nb ) const {
    array<index_t,7> IDX;
    _unravel_I( idx, nk, nb, IDX );
    index_t q = IDX[0]; index_t k  = IDX[1]; index_t b1 = IDX[2]; index_t b2 = IDX[3];
                      index_t kp = IDX[4]; index_t b3 = IDX[5]; index_t b4 = IDX[6];
    IDX = array<index_t,7>{ k, _k1m2[nk*kp+q], kp, b2, b3, b4, b1 };
    return _ravel_V( nk, nb, IDX );
}

index_t IndexGymnastics::D_to_V( index_t idx, index_t nk, index_t nb ) const {
    array<index_t,7> IDX;
    _unravel_I( idx, nk, nb, IDX );
    index_t q = IDX[0]; index_t k  = IDX[1]; index_t b1 = IDX[2]; index_t b2 = IDX[3];
                      index_t kp = IDX[4]; index_t b3 = IDX[5]; index_t b4 = IDX[6];
    IDX = array<index_t,7>{ k, _k1m2[nk*kp+q], _k1m2[nk*k+q], b2, b3, b1, b4 };
    return _ravel_V( nk, nb, IDX );
}

index_t IndexGymnastics::D_to_P( index_t idx, index_t nk, index_t nb ) const {
    return V_to_P( D_to_V( idx, nk, nb ), nk, nb );
}

index_t IndexGymnastics::P_to_D( index_t idx, index_t nk, index_t nb ) const {
    return V_to_D( P_to_V( idx, nk, nb ), nk, nb );
}

index_t IndexGymnastics::D_to_C( index_t idx, index_t nk, index_t nb ) const {
    return V_to_C( D_to_V( idx, nk, nb ), nk, nb );
}

index_t IndexGymnastics::C_to_D( index_t idx, index_t nk, index_t nb ) const {
    return V_to_D( C_to_V( idx, nk, nb ), nk, nb );
}

index_t IndexGymnastics::D_swap_3_and_4( index_t idx, index_t nk, index_t nb ) const {
    return V_to_D( V_swap_3_and_4( D_to_V( idx, nk, nb ), nk, nb ), nk, nb );
}

index_t IndexGymnastics::D_swap_1_and_2( index_t idx, index_t nk, index_t nb ) const {
    return V_to_D( V_swap_1_and_2( D_to_V( idx, nk, nb ), nk, nb ), nk, nb );
}

index_t IndexGymnastics::V_swap_3_and_4( index_t idx, index_t nk, index_t nb ) const {
    array<index_t,7> IDX;
    _unravel_V( idx, nk, nb, IDX );
    index_t k1 = IDX[0]; index_t k2 = IDX[1]; index_t k3 = IDX[2];
    index_t b1 = IDX[3]; index_t b2 = IDX[4]; index_t b3 = IDX[5]; index_t b4 = IDX[6];
    if (_k3map)
        IDX = array<index_t,7>{ k1, k2, _k3map[nk*nk*k1 + nk*k2 + k3], b1, b2, b4, b3 };
    else
        IDX = array<index_t,7>{ k1, k2, _k1m2[nk*_k1p2[nk*k1+k2]+k3], b1, b2, b4, b3 };
    return _ravel_V( nk, nb, IDX );
}

index_t IndexGymnastics::V_swap_1_and_2( index_t idx, index_t nk, index_t nb ) const {
    array<index_t,7> IDX;
    _unravel_V( idx, nk, nb, IDX );
    index_t k1 = IDX[0]; index_t k2 = IDX[1]; index_t k3 = IDX[2];
    index_t b1 = IDX[3]; index_t b2 = IDX[4]; index_t b3 = IDX[5]; index_t b4 = IDX[6];
    IDX = array<index_t,7>{ k2, k1, k3, b2, b1, b3, b4 };
    return _ravel_V( nk, nb, IDX );
}

IndexGymnastics::IndexGymnastics( const index_t* k1m2, const index_t* k1p2, const index_t* k3map ) {
    _k1m2 = k1m2;
    _k1p2 = k1p2;
    _k3map = k3map;
}

void IndexGymnastics::_unravel_V( index_t idx, index_t nk, index_t nb, array<index_t,7>& IDX ) const {
    index_t bandsz = nb*nb*nb*nb;
    index_t K = idx/bandsz;
    IDX[0] = K/(nk*nk);
    K %= (nk*nk);
    IDX[1] = K/nk;
    IDX[2] = K%nk;
    index_t B = idx%bandsz;
    IDX[3] = B/(nb*nb*nb);
    B %= (nb*nb*nb);
    IDX[4] = B/(nb*nb);
    B %= (nb*nb);
    IDX[5] = B/nb;
    IDX[6] = B%nb;
}

void IndexGymnastics::_unravel_I( index_t idx, index_t nk, index_t nb, array<index_t,7>& IDX ) const {
    index_t kband = nk*nk*nb*nb*nb*nb;
    IDX[0] = idx/kband;
    index_t II = idx%kband;
    IDX[1] = II/(nb*nb*nk*nb*nb);
    II %= (nb*nb*nk*nb*nb);
    IDX[2] = II/(nb*nk*nb*nb);
    II %= (nb*nk*nb*nb);
    IDX[3] = II/(nk*nb*nb);
    II %= (nk*nb*nb);
    IDX[4] = II/(nb*nb);
    II %= (nb*nb);
    IDX[5] = II/(nb);
    IDX[6] = II%(nb);
}

index_t IndexGymnastics::_ravel_V( index_t nk, index_t nb, const array<index_t,7>& IDX ) const {
    return IDX[0]*nk*nk*nb*nb*nb*nb + IDX[1]*nk*nb*nb*nb*nb + IDX[2]*nb*nb*nb*nb +
        IDX[3]*nb*nb*nb + IDX[4]*nb*nb + IDX[5]*nb + IDX[6];
}

index_t IndexGymnastics::_ravel_I( index_t nk, index_t nb, const array<index_t,7>& IDX ) const {
    return IDX[0]*nk*nk*nb*nb*nb*nb + IDX[1]*nb*nb*nk*nb*nb + IDX[2]*nb*nk*nb*nb +
        IDX[3]*nk*nb*nb + IDX[4]*nb*nb + IDX[5]*nb + IDX[6];
}

} // grid
