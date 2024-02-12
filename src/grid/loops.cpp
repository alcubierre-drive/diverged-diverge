/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "loops.hpp"
#include "symmetry.hpp"
#include "vertex_memory.hpp"
#include "index_generator.hpp"
#include "../diverge_model.h"
#include "../diverge_internals_struct.h"

#include "../misc/cmath_helpers.h"

namespace grid {

template<bool isProjectedLoop = false>
static void frg_loop_calc_P_greens( complex128_t* mem, const complex128_t* greens_p,
        const complex128_t* greens_m, index_t qstart, index_t qstop, index_t nk, index_t nkf,
        index_t nb, const index_t* kfrev, const double* kfweights, const index_t* k1m2 );
template<bool isProjectedLoop = false>
static void frg_loop_calc_C_greens( complex128_t* mem, const complex128_t* greens_p,
        const complex128_t* greens_m, index_t qstart, index_t qstop, index_t nk, index_t nkf,
        index_t nb, const index_t*, const double* kfweights, const index_t* k1m2 );

static void transform_to_gbuf( const gf_complex_t* greens, complex128_t* greens_p,
        complex128_t* greens_m, const index_t* grid_kfmesh, index_t grid_nk_nkf,
        index_t div_nk_nkf, index_t nb );

static complex128_t* generate_trivial_formfactors( diverge_model_t* dmod, index_t& nff );

Loops::Loops( diverge_model_t* dmod ) {

    VertexMemory* V = (VertexMemory*)dmod->internals->grid_vertex;
    SymmetryTransform* S = (SymmetryTransform*)dmod->internals->grid_sym;

    _dmod = dmod;
    _nk = _dmod->internals->grid_nk;
    _nb = _dmod->internals->grid_nb;
    _greens_cache_p = (complex128_t*)_dmod->internals->grid_greens_p_buf;
    _greens_cache_m = (complex128_t*)_dmod->internals->grid_greens_m_buf;

    _nkf = _dmod->internals->grid_nkf;
    _k1m2 = _dmod->internals->grid_k1m2;
    _kfrev = _dmod->internals->grid_kfrev;
    _kfweights = _dmod->internals->grid_weights;

    if (_dmod->gproj != NULL) {
        _greens_proj_cache_p = (complex128_t*)calloc(_nk*_nkf*_nb*_nb, sizeof(complex128_t));
        _greens_proj_cache_m = (complex128_t*)calloc(_nk*_nkf*_nb*_nb, sizeof(complex128_t));
    }

    _qstart = V->get_index_gen()->qstart;
    _qstop = V->get_index_gen()->qstop;
    _qsize = _qstop-_qstart;
    _Xsize = _qsize*_nk*_nb*_nb*_nb*_nb;
    _X_P = (complex128_t*)malloc( _Xsize*sizeof(complex128_t) );
    _X_C = (complex128_t*)malloc( _Xsize*sizeof(complex128_t) );

    if (S != NULL) {
        SymmetryCommunicatorFactory lsym_fac( S, V->get_mpi_info(),
            _k1m2, _dmod->internals->grid_k1p2, _qstart, _qstop, _nk, _nb );
        set_symmetrizer( new LoopSymmetrizer( &lsym_fac, V->get_mpi_info() ) );
        _SYM_free = true;
    }

    _formfac = generate_trivial_formfactors( dmod, _nff );
    _formfac_free = true;
}

Loops::~Loops() {
    free(_X_P);
    free(_X_C);

    if (_greens_proj_cache_p) free(_greens_proj_cache_p);
    if (_greens_proj_cache_m) free(_greens_proj_cache_m);

    if (_SYM_free)
        delete _SYM;

    if (_formfac_free)
        free(_formfac);
}

index_t Loops::get_kzero(void) {
    return 0;
}

index_t Loops::get_nspin(void) {
    return std::abs(_dmod->n_spin);
}

void Loops::_greens_cache_fill(double Lambda) {
    if (_greens_cache_lambda == Lambda)
        return;
    _greens_cache_lambda = Lambda;

    complex128_t Lam(0,Lambda);
    complex128_t& cLam = *(complex128_t*)&Lam;
    if ((*_dmod->gfill)(_dmod, cLam, _dmod->internals->greens) != greensfunc_op_cpu)
        mpi_err_printf("grid FRG does not support GPU Green functions\n");

    transform_to_gbuf( _dmod->internals->greens, _greens_cache_p, _greens_cache_m,
            _dmod->internals->grid_kfmesh, _dmod->internals->grid_nk * _dmod->internals->grid_nkf,
            kdimtot( _dmod->nk, _dmod->nkf ), _nb );

    if (_dmod->gproj) {
        if ((*_dmod->gproj)(_dmod, cLam, _dmod->internals->greens) != greensfunc_op_cpu)
            mpi_err_printf("grid FRG does not support GPU Green functions\n");
        transform_to_gbuf( _dmod->internals->greens, _greens_proj_cache_p, _greens_proj_cache_m,
            _dmod->internals->grid_kfmesh, _dmod->internals->grid_nk * _dmod->internals->grid_nkf,
            kdimtot( _dmod->nk, _dmod->nkf ), _nb );
    }
}

complex128_t* Loops::X_P(double Lambda) {
    _greens_cache_fill(Lambda);
    frg_loop_calc_P_greens( _X_P, _greens_cache_p, _greens_cache_m,
            _qstart, _qstop, _nk, _nkf, _nb, _kfrev, _kfweights, _k1m2 );
    if (_dmod->gproj != NULL)
        frg_loop_calc_P_greens<true>( _X_P, _greens_proj_cache_p, _greens_proj_cache_m,
            _qstart, _qstop, _nk, _nkf, _nb, _kfrev, _kfweights, _k1m2 );
    _max_P = absmax( _X_P, _Xsize );
    return _X_P;
}

complex128_t* Loops::X_C(double Lambda) {
    _greens_cache_fill(Lambda);
    frg_loop_calc_C_greens( _X_C, _greens_cache_p, _greens_cache_m,
            _qstart, _qstop, _nk, _nkf, _nb, _kfrev, _kfweights, _k1m2 );
    if (_dmod->gproj != NULL)
        frg_loop_calc_C_greens<true>( _X_C, _greens_proj_cache_p, _greens_proj_cache_m,
            _qstart, _qstop, _nk, _nkf, _nb, _kfrev, _kfweights, _k1m2 );
    _max_C = absmax( _X_C, _Xsize );
    return _X_C;
}

complex128_t* Loops::symmetrize( complex128_t* loop, char chan ) {
    if (_SYM != nullptr) {
        return (*_SYM)( loop, chan );
    } else {
        mpi_err_printf("no symmetrizer available\n");
        return loop;
    }
}

template<bool isProjectedLoop>
static void frg_loop_calc_P_greens( complex128_t* mem, const complex128_t* greens_p,
        const complex128_t* greens_m, index_t qstart, index_t qstop, index_t nk, index_t nkf,
        index_t nb, const index_t* kfrev, const double* kfweights, const index_t* k1m2 ) {

    index_t LoopSize = (qstop-qstart)*nk*nb*nb*nb*nb;

    const double norm = 1./2./M_PI * (isProjectedLoop?-1.0:1.0);

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t Loopindex=0; Loopindex<LoopSize; ++Loopindex) {
        index_t q = Loopindex/(nk*nb*nb*nb*nb) + qstart,
              k = (Loopindex%(nk*nb*nb*nb*nb))/(nb*nb*nb*nb),
              o1 = (Loopindex%(nb*nb*nb*nb))/(nb*nb*nb),
              o2 = (Loopindex%(nb*nb*nb))/(nb*nb),
              o3 = (Loopindex%(nb*nb))/nb,
              o4 = Loopindex%nb;

        if constexpr(!isProjectedLoop)
            mem[Loopindex] = 0;

        index_t k1 = k;
        index_t k2 = k1m2[q*nk+k];

        for (index_t kf1=0; kf1<nkf; ++kf1) {
            index_t kf2 = kfrev[kf1];
            mem[Loopindex] += norm * kfweights[kf1] * ( greens_m[k1*nkf*nb*nb+kf1*nb*nb+o1*nb+o3] *
                                                        greens_p[k2*nkf*nb*nb+kf2*nb*nb+o2*nb+o4] +
                                                        greens_p[k1*nkf*nb*nb+kf1*nb*nb+o1*nb+o3] *
                                                        greens_m[k2*nkf*nb*nb+kf2*nb*nb+o2*nb+o4] );
        }
    }
}

template<bool isProjectedLoop>
static void frg_loop_calc_C_greens( complex128_t* mem, const complex128_t* greens_p,
        const complex128_t* greens_m, index_t qstart, index_t qstop, index_t nk, index_t nkf,
        index_t nb, const index_t*, const double* kfweights, const index_t* k1m2 ) {

    index_t LoopSize = (qstop-qstart)*nk*nb*nb*nb*nb;

    const double norm = 1./2./M_PI * (isProjectedLoop?-1.0:1.0);

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t Loopindex=0; Loopindex<LoopSize; ++Loopindex) {
        index_t q = Loopindex/(nk*nb*nb*nb*nb) + qstart,
              k = (Loopindex%(nk*nb*nb*nb*nb))/(nb*nb*nb*nb),
              O1 = (Loopindex%(nb*nb*nb*nb))/(nb*nb*nb),
              O2 = (Loopindex%(nb*nb*nb))/(nb*nb),
              O3 = (Loopindex%(nb*nb))/nb,
              O4 = Loopindex%nb;
        index_t o1 = O2, o2 = O3, o3 = O4, o4 = O1;

        if constexpr(!isProjectedLoop)
            mem[Loopindex] = 0;

        index_t k1 = k;
        index_t k2 = k1m2[k*nk+q];

        for (index_t kf1=0; kf1<nkf; ++kf1) {
            index_t kf2 = kf1;
            mem[Loopindex] += norm * kfweights[kf1] * ( greens_m[k1*nkf*nb*nb+kf1*nb*nb+o1*nb+o3] *
                                                        greens_m[k2*nkf*nb*nb+kf2*nb*nb+o2*nb+o4] +
                                                        greens_p[k1*nkf*nb*nb+kf1*nb*nb+o1*nb+o3] *
                                                        greens_p[k2*nkf*nb*nb+kf2*nb*nb+o2*nb+o4] );
        }
    }
}

static void transform_to_gbuf( const gf_complex_t* greens, complex128_t* greens_p,
        complex128_t* greens_m, const index_t* grid_kfmesh, index_t grid_nk_nkf,
        index_t div_nk_nkf, index_t nb ) {

    const gf_complex_t* div_gp = greens;
    const gf_complex_t* div_gm = greens + div_nk_nkf*nb*nb;
    const index_t nb2 = nb*nb;

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t kg=0; kg<grid_nk_nkf; ++kg) {
        index_t kd = grid_kfmesh[kg];
        for (index_t bb=0; bb<nb2; ++bb)
            greens_p[kg*nb2+bb] = -div_gp[kd*nb2+bb];
    }

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t kg=0; kg<grid_nk_nkf; ++kg) {
        index_t kd = grid_kfmesh[kg];
        for (index_t bb=0; bb<nb2; ++bb)
            greens_m[kg*nb2+bb] = -div_gm[kd*nb2+bb];
    }

}

static complex128_t* generate_trivial_formfactors( diverge_model_t* dmod, index_t& nff ) {
    internals_t* intls = dmod->internals;
    Mat3d lattice = Map<Mat3d>( (double*)(dmod->lattice) );
    for (int d=dmod->internals->dim-1; d<3; ++d)
        lattice.col(d) *= GRID_FF_SHELL_DISTANCE * 2 / lattice.col(d).norm();

    vector<array<index_t,3>> rs_ff_candidates;
    rs_ff_candidates.reserve( POW2(GRID_FF_SHELL_DISTANCE * 2 + 1) );
    double norm_fac = lattice.block(0,0,dmod->internals->dim,dmod->internals->dim).colwise().norm().mean();

    for (int r1=-3*GRID_FF_SHELL_DISTANCE; r1<=3*GRID_FF_SHELL_DISTANCE; ++r1)
    for (int r2=-3*GRID_FF_SHELL_DISTANCE; r2<=3*GRID_FF_SHELL_DISTANCE; ++r2)
    for (int r3=-3*GRID_FF_SHELL_DISTANCE; r3<=3*GRID_FF_SHELL_DISTANCE; ++r3) {
        Vec3d R = lattice.col(0) * r1 + lattice.col(1) * r2 + lattice.col(2) * r3;
        if (R.norm() / norm_fac < GRID_FF_SHELL_DISTANCE && (r1 != 0 && r2 != 0 && r3 != 0))
            rs_ff_candidates.push_back( {r1,r2,r3} );
    }

    vector<array<index_t,3>> rs_ff;
    rs_ff.reserve( rs_ff_candidates.size() );
    auto it = rs_ff_candidates.begin();
    while (it != rs_ff_candidates.end()) {
        array<index_t,3> mff = *it;
        mff[0] *= -1;
        mff[1] *= -1;
        mff[2] *= -1;
        auto it_neg = std::find( rs_ff_candidates.begin(), rs_ff_candidates.end(), mff );
        if (it_neg != rs_ff_candidates.end())
            rs_ff_candidates.erase( it_neg );
        rs_ff.push_back( *it );
        ++it;
    }

    std::sort( rs_ff.begin(), rs_ff.end(), [&](const array<index_t,3>& A, const array<index_t,3>& B)->bool{
            Vec3d AA = A[0] * lattice.col(0) + A[1] * lattice.col(1) + A[2] * lattice.col(2);
            Vec3d BB = B[0] * lattice.col(0) + B[1] * lattice.col(1) + B[2] * lattice.col(2);
            return AA.norm() < BB.norm();
    });

    nff = rs_ff.size()*2+1;

    if (nff > intls->grid_nk) {
        mpi_wrn_printf( "small grid (nk=%li), truncating #formfactors (nff=%li)\n",
                intls->grid_nk, nff );
        rs_ff.resize(intls->grid_nk-1);
        nff = intls->grid_nk;
    }
    complex128_t* formfac = (complex128_t*)malloc(sizeof(complex128_t)*nff*intls->grid_nk);
    complex128_t* buf = formfac;
    for (index_t k=0; k<intls->grid_nk; ++k)
        buf[k] = 1.0;
    buf += intls->grid_nk;
    for (uindex_t ff=0; ff<rs_ff.size(); ++ff) {
        Vec3d Rvec = rs_ff[ff][0] * lattice.col(0) + rs_ff[ff][1] * lattice.col(1) + rs_ff[ff][2] * lattice.col(2);
        for (index_t sign=0; sign<2; ++sign) {
            for (index_t k=0; k<intls->grid_nk; ++k) {
                Map<Vec3d> Kvec(intls->kmesh+3*k);
                buf[k] = sign ? cmath_cos(Rvec.dot(Kvec)) : cmath_sin(Rvec.dot(Kvec));
            }
            buf += intls->grid_nk;
        }
    }

    for (index_t ff=0; ff<nff; ++ff) {
        Map<VecXcd> F(formfac+ff*intls->grid_nk, intls->grid_nk);
        F /= F.norm();
    }
    return formfac;
}

}
