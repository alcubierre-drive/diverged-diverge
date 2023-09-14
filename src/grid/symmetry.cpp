/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "symmetry.hpp"
#include <algorithm>
#ifdef USE_BOOST_SORT

// ignore boost warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <boost/sort/sort.hpp>
#pragma GCC diagnostic pop
#include <omp.h>
#define _CUSTOM_STABLE_SORT_(begin, end, comp) boost::sort::parallel_stable_sort(begin, end, comp, omp_get_max_threads())

#else // USE_BOOST_SORT

#define _CUSTOM_STABLE_SORT_(begin, end, comp) std::stable_sort(begin, end, comp)

#endif // USE_BOOST_SORT

namespace grid {

typedef Eigen::Matrix<index_t,-1,-1> MatXi;
typedef Eigen::Matrix<index_t,-1,1> VecXi;

struct symm_props {
    int idx_pre_symm;
    int idx_post_symm;
    int R1, R2, R3;
};

template<typename Ostream>
Ostream& operator<<( Ostream& str, const symm_props& sp ) {
    str << "S:" << sp.idx_pre_symm << "->" << sp.idx_post_symm << "(" <<
        sp.R1 << "," << sp.R2 << "," << sp.R3 << ")";
    return str;
}

template<typename Ostream>
Ostream& operator<<( Ostream& str, const vector<symm_props>& vsp ) {
    int n = vsp.size();
    if (n == 0)
        return str;
    else if (n == 1) {
        str << vsp[0];
        return str;
    } else {
        str << vsp[0];
        for (int i=1; i<n; ++i) {
            str << "\n" << vsp[i];
        }
        return str;
    }
}

template<typename Ostream>
Ostream& operator<<( Ostream& str, const vector<index_t>& ksym ) {
    int n = ksym.size();
    if (n == 0)
        return str;
    else if (n == 1) {
        str << 0 << "->" << ksym[0];
        return str;
    } else {
        str << 0 << "->" << ksym[0];
        for (int i=1; i<n; ++i) {
            str << "\n" << i << "->" << ksym[i];
        }
        return str;
    }
}

static vector<symm_props> gen_symm_props( const vector<Vec3d>& basis,
        const array<Vec3d,3>& lattice, const Mat3d& symm );
static vector<index_t> k_symmap( const vector<Vec3d>& kmesh,
        const array<Vec3d,3>& reciprocal, const Mat3d& symm );

SymmetryTransform::~SymmetryTransform() {}

SymmetryTransform::SymmetryTransform(
                const vector<Mat3d>& symmetries,
                const vector<Vec3d>& basis,
                const array<Vec3d,3>& lattice,
                const array<Vec3d,3>& reciprocal,
                const vector<Vec3d>& kmesh ) {

    _symmetries = symmetries;
    _basis = basis;
    _lattice = lattice;
    _reciprocal = reciprocal;
    _kmesh = kmesh;

    nsym = _symmetries.size();
    _symm_props.resize( nsym );
    _k_symmap.resize( nsym );
    #pragma omp parallel for
    for (index_t i=0; i<nsym; ++i) {
        _symm_props[i] = gen_symm_props( _basis, _lattice, _symmetries[i] );
        _k_symmap[i] = k_symmap( _kmesh, _reciprocal, _symmetries[i] );
    }

    _symmetry_phases_P.resize(nsym);
    _symmetry_maps_P.resize(nsym);
    _symmetry_phases_C.resize(nsym);
    _symmetry_maps_C.resize(nsym);

    for (unsigned i=0; i<nsym; ++i) {
        _symmetry_phases_P[i].resize(0);
        _symmetry_maps_P[i].resize(0);
        _symmetry_phases_C[i].resize(0);
        _symmetry_maps_C[i].resize(0);
    }

    _symmetry_matrices.resize( nsym );
}

void SymmetryTransform::_reset_bond_symmprops(bool diverge_style) {
    for (vector<symm_props>& spv: _symm_props) {
        for (symm_props& s: spv) {
            s.idx_post_symm = s.idx_pre_symm;
            if (!diverge_style)
                s.R1 = s.R2 = s.R3 = 0;
        }
    }
}

void SymmetryTransform::transform_P( const index_t* k1m2, const index_t* k1p2,
        index_t qstart, index_t qstop, index_t nk, index_t nb, index_t symm ) {

    index_t LoopSize = (qstop-qstart)*nk*nb*nb*nb*nb;
    if (_symmetry_maps_P[symm].size() == 0 || _symmetry_phases_P[symm].size() == 0) {
        _symmetry_maps_P[symm].resize(LoopSize);
        _symmetry_phases_P[symm].resize(LoopSize);
    }
    #pragma omp parallel for
    for (index_t Loopindex_t=0; Loopindex_t<LoopSize; ++Loopindex_t) {
        index_t q = Loopindex_t/(nk*nb*nb*nb*nb) + qstart,
              k = (Loopindex_t%(nk*nb*nb*nb*nb))/(nb*nb*nb*nb),
              o1 = (Loopindex_t%(nb*nb*nb*nb))/(nb*nb*nb),
              o2 = (Loopindex_t%(nb*nb*nb))/(nb*nb),
              o3 = (Loopindex_t%(nb*nb))/nb,
              o4 = Loopindex_t%nb;

        index_t k1 = k,
              k2 = k1m2[q*nk+k];

        index_t k1_T = _k_symmap[symm][k1],
              o1_T = _symm_props[symm][o1].idx_post_symm,
              o3_T = _symm_props[symm][o3].idx_post_symm,
              k2_T = _k_symmap[symm][k2],
              o2_T = _symm_props[symm][o2].idx_post_symm,
              o4_T = _symm_props[symm][o4].idx_post_symm;

        index_t k_T = k1_T,
              q_T = k1p2[k2_T*nk+k1_T];

        Vec3d k1vT = _kmesh[k1_T];
        Vec3d k2vT = _kmesh[k2_T];
        Vec3d So1 = _lattice[0] * _symm_props[symm][o1].R1 +
                    _lattice[1] * _symm_props[symm][o1].R2 +
                    _lattice[2] * _symm_props[symm][o1].R3;
        Vec3d So2 = _lattice[0] * _symm_props[symm][o2].R1 +
                    _lattice[1] * _symm_props[symm][o2].R2 +
                    _lattice[2] * _symm_props[symm][o2].R3;
        Vec3d So3 = _lattice[0] * _symm_props[symm][o3].R1 +
                    _lattice[1] * _symm_props[symm][o3].R2 +
                    _lattice[2] * _symm_props[symm][o3].R3;
        Vec3d So4 = _lattice[0] * _symm_props[symm][o4].R1 +
                    _lattice[1] * _symm_props[symm][o4].R2 +
                    _lattice[2] * _symm_props[symm][o4].R3;

        complex128_t phase_first = exp( - I128 * k1vT.dot(So1 - So3) );
        complex128_t phase_second =exp( - I128 * k2vT.dot(So2 - So4) );

        index_t fullindex_T = q_T * nk*nb*nb*nb*nb + k_T * nb*nb*nb*nb + o1_T * nb*nb*nb + o2_T * nb*nb + o3_T * nb + o4_T;
        complex128_t fullPhase_T = phase_first * phase_second;

        _symmetry_maps_P[symm][Loopindex_t] = fullindex_T;
        _symmetry_phases_P[symm][Loopindex_t] = fullPhase_T;
    }
}

void SymmetryTransform::transform_C( const index_t* k1m2, const index_t*,
        index_t qstart, index_t qstop, index_t nk, index_t nb, index_t symm ) {

    index_t LoopSize = (qstop-qstart)*nk*nb*nb*nb*nb;
    if (_symmetry_maps_C[symm].size() == 0 || _symmetry_phases_C[symm].size() == 0) {
        _symmetry_maps_C[symm].resize(LoopSize);
        _symmetry_phases_C[symm].resize(LoopSize);
    }

    #pragma omp parallel for
    for (index_t Loopindex_t=0; Loopindex_t<LoopSize; ++Loopindex_t) {

        index_t q = Loopindex_t/(nk*nb*nb*nb*nb) + qstart,
              k = (Loopindex_t%(nk*nb*nb*nb*nb))/(nb*nb*nb*nb),
              O1 = (Loopindex_t%(nb*nb*nb*nb))/(nb*nb*nb),
              O2 = (Loopindex_t%(nb*nb*nb))/(nb*nb),
              O3 = (Loopindex_t%(nb*nb))/nb,
              O4 = Loopindex_t%nb;
        index_t o1 = O2, o2 = O3, o3 = O4, o4 = O1;

        index_t k1 = k,
              k2 = k1m2[k*nk+q];

        index_t k1_T = _k_symmap[symm][k1],
              o1_T = _symm_props[symm][o1].idx_post_symm,
              o3_T = _symm_props[symm][o3].idx_post_symm,
              k2_T = _k_symmap[symm][k2],
              o2_T = _symm_props[symm][o2].idx_post_symm,
              o4_T = _symm_props[symm][o4].idx_post_symm;

        index_t k_T = k1_T,
              q_T = k1m2[k1_T*nk+k2_T];

        Vec3d k1vT = _kmesh[k1_T];
        Vec3d k2vT = _kmesh[k2_T];
        Vec3d So1 = _lattice[0] * _symm_props[symm][o1].R1 +
                    _lattice[1] * _symm_props[symm][o1].R2 +
                    _lattice[2] * _symm_props[symm][o1].R3;
        Vec3d So2 = _lattice[0] * _symm_props[symm][o2].R1 +
                    _lattice[1] * _symm_props[symm][o2].R2 +
                    _lattice[2] * _symm_props[symm][o2].R3;
        Vec3d So3 = _lattice[0] * _symm_props[symm][o3].R1 +
                    _lattice[1] * _symm_props[symm][o3].R2 +
                    _lattice[2] * _symm_props[symm][o3].R3;
        Vec3d So4 = _lattice[0] * _symm_props[symm][o4].R1 +
                    _lattice[1] * _symm_props[symm][o4].R2 +
                    _lattice[2] * _symm_props[symm][o4].R3;

        complex128_t phase_first = exp( - I128 * k1vT.dot(So1 - So3) );
        complex128_t phase_second = exp( - I128 * k2vT.dot(So2 - So4) );

        index_t O1p = o4_T,
              O2p = o1_T,
              O3p = o2_T,
              O4p = o3_T;
        index_t fullindex_T = q_T * nk*nb*nb*nb*nb + k_T * nb*nb*nb*nb + O1p * nb*nb*nb + O2p * nb*nb + O3p * nb + O4p;
        complex128_t fullPhase_T = phase_first * phase_second;

        _symmetry_maps_C[symm][Loopindex_t] = fullindex_T;
        _symmetry_phases_C[symm][Loopindex_t] = fullPhase_T;
    }
}

void SymmetryTransform::clean_mem( index_t i ) {
    if (_symmetry_phases_P[i].size() > 0)
        _symmetry_phases_P[i].resize(0);
    if (_symmetry_phases_C[i].size() > 0)
        _symmetry_phases_C[i].resize(0);
    if (_symmetry_maps_P[i].size() > 0)
        _symmetry_maps_P[i].resize(0);
    if (_symmetry_maps_C[i].size() > 0)
        _symmetry_maps_C[i].resize(0);
}

vector<symm_props> gen_symm_props( const vector<Vec3d>& basis,
        const array<Vec3d,3>& lattice, const Mat3d& symm ) {
    vector<symm_props> result(basis.size());
    for (unsigned i=0; i<basis.size(); ++i) {
        result[i].idx_pre_symm = i;
        for (unsigned j=0; j<basis.size(); ++j) {
            for (int r1=-2; r1<=2; ++r1) for (int r2=-2; r2<=2; ++r2) for (int r3=-2; r3<=2; ++r3) {
                Vec3d A = lattice[0] * r1 + lattice[1] * r2 + lattice[2] * r3;
                if ( (symm * basis[i] + A - basis[j]).squaredNorm() <= 1.e-7 ) {
                    result[i].idx_post_symm = j;
                    result[i].R1 = r1;
                    result[i].R2 = r2;
                    result[i].R3 = r3;
                    goto endloop_symm_props;
                }
            }
        }
endloop_symm_props: {}
    }
    return result;
}

vector<index_t> k_symmap( const vector<Vec3d>& kmesh,
        const array<Vec3d,3>& reciprocal, const Mat3d& symm ) {
    vector<index_t> result( kmesh.size() );
    for (unsigned i=0; i<kmesh.size(); ++i) {
        Vec3d ksym = symm * kmesh[i];
        for (unsigned j=0; j<kmesh.size(); ++j) {
            Vec3d ksym_ij = ksym - kmesh[j];
            for (int g1=-2; g1<=2; ++g1) for (int g2=-2; g2<=2; ++g2) for (int g3=-2; g3<=2; ++g3) {
                Vec3d ksym_ij_g1g2 = ksym_ij + reciprocal[0] * g1 + reciprocal[1] * g2 + reciprocal[2] * g3;
                if (ksym_ij_g1g2.squaredNorm() <= 1.e-7) {
                    result[i] = j;
                    goto endloop_k_symmap;
                }
            }
        }
endloop_k_symmap: {}
    }
    return result;
}

int rank_from_index( const vector<index_t>& qstarts, const vector<index_t>& qstops,
        index_t L, index_t nk, index_t nb ) {
    index_t q = L/(nk*nb*nb*nb*nb);
    for (unsigned i=0; i<qstarts.size(); ++i) {
        if (q >= qstarts[i] && q < qstops[i])
            return i;
    }
    return -1;
}

SymmetryCommunicatorFactory::SymmetryCommunicatorFactory( SymmetryTransform* st, dist_info* mpi_info,
        const index_t* k1m2, const index_t* k1p2, index_t qstart, index_t qstop, index_t nk, index_t nb ) {
    _mpi = mpi_info;
    _st = st;
    _qstart = qstart;
    _qstop = qstop;
    vector<index_t> qss = _qstartstops();
    _qstarts.resize( qss.size()/2 );
    _qstops.resize( qss.size()/2 );
    for (unsigned i=0; i<_qstarts.size(); ++i) {
        _qstarts[i] = qss[2*i];
        _qstops[i] = qss[2*i+1];
    }

    int rank = diverge_mpi_comm_rank();

    _nk = nk;
    _nb = nb;

    _loop_size = (_qstop - _qstart) * _nk * _nb*_nb*_nb*_nb;

    _symops_P.resize( _st->get_nsym() );
    _symops_C.resize( _st->get_nsym() );

    index_t qfac = _nk * _nb * _nb * _nb * _nb;
    for (int i=0; i<_st->get_nsym(); ++i) {
        _st->transform_P( k1m2, k1p2, _qstart, _qstop, _nk, _nb, i );
        _st->transform_C( k1m2, k1p2, _qstart, _qstop, _nk, _nb, i );
        _symops_P[i] = new SymmetryOperation[_loop_size];
        _symops_C[i] = new SymmetryOperation[_loop_size];
        vector<complex128_t>& pP = _st->get_phases<'P'>( i );
        vector<complex128_t>& pC = _st->get_phases<'C'>( i );
        vector<index_t>& mP = _st->get_map<'P'>( i );
        vector<index_t>& mC = _st->get_map<'C'>( i );
        #pragma omp parallel for
        for (index_t l=0; l<_loop_size; ++l) {
            _symops_P[i][l].current_rank = rank;
            _symops_P[i][l].current_index = _qstart*qfac + l;
            _symops_P[i][l].phase = pP[l];
            _symops_P[i][l].target_index = mP[l];
            _symops_P[i][l].target_rank = rank_from_index( _qstarts, _qstops, mP[l], _nk, _nb );
            _symops_C[i][l].current_rank = rank;
            _symops_C[i][l].current_index = _qstart*qfac + l;
            _symops_C[i][l].phase = pC[l];
            _symops_C[i][l].target_index = mC[l];
            _symops_C[i][l].target_rank = rank_from_index( _qstarts, _qstops, mC[l], _nk, _nb );
        }
        _st->clean_mem(i);
        _CUSTOM_STABLE_SORT_( _symops_P[i], _symops_P[i] + _loop_size, [](const SymmetryOperation& a,
                    const SymmetryOperation& b)->bool{ return a.target_rank < b.target_rank; } );
        _CUSTOM_STABLE_SORT_( _symops_C[i], _symops_C[i] + _loop_size, [](const SymmetryOperation& a,
                    const SymmetryOperation& b)->bool{ return a.target_rank < b.target_rank; } );
    }
}

vector<index_t> SymmetryCommunicatorFactory::_qstartstops() {
    int nranks = diverge_mpi_comm_size();
    array<index_t,2> q = { _qstart, _qstop };
    vector<index_t> qstartstops( nranks*2 );
    diverge_mpi_allgather_index( q.data(), qstartstops.data(), 2 );
    return qstartstops;
}

SymmetryCommunicatorFactory::~SymmetryCommunicatorFactory() {
    for (index_t i=0; i<_st->get_nsym(); ++i) {
        delete[] _symops_P[i];
        delete[] _symops_C[i];
    }
}

template<typename Ostream> Ostream& operator<<( Ostream& str, const SymmetryOperation& op ) {
    str << "i:" << op.current_index << "(" << op.current_rank << ")->j:" <<
        op.target_index << "(" << op.target_rank << ")"; //::" << op.phase;
    return str;
}

SymmetryCommunication SymmetryCommunicatorFactory::gen_communication( index_t symm, char chan ) {

    const SymmetryOperation* symop = chan == 'P' ? get_symop<'P'>(symm) :
                                                   get_symop<'C'>(symm);
    int nranks = diverge_mpi_comm_size();
    int rank = diverge_mpi_comm_rank();
    vector<index_t> send_counts(nranks);
    for (index_t i=0; i<_loop_size; ++i)
        send_counts[symop[i].target_rank]++;
    vector<index_t> send_displs(nranks);
    send_displs[0] = 0;
    for (int i=1; i<nranks; ++i)
        send_displs[i] = send_displs[i-1] + send_counts[i-1];

    MatXi all_send_counts( nranks, nranks );
    diverge_mpi_allgather_index( send_counts.data(), all_send_counts.data(), nranks );

    VecXi recv_counts_ = all_send_counts.row( rank );
    vector<index_t> recv_counts( nranks );
    vector<index_t> recv_displs( nranks );
    for (int i=0; i<nranks; ++i)
        recv_counts[i] = recv_counts_[i];
    recv_displs[0] = 0;
    for (int i=1; i<nranks; ++i)
        recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];

    vector<SymmetryOperation> recv_op(_loop_size);

    diverge_mpi_alltoallv_bytes( symop, send_counts.data(), send_displs.data(),
                                 recv_op.data(), recv_counts.data(), recv_displs.data(),
                                 sizeof(SymmetryOperation) );

    SymmetryCommunication result;
    result.reorder_before_send.resize(_loop_size);
    result.phase_before_send.resize(_loop_size);
    result.send_counts.resize(nranks);
    result.send_displs.resize(nranks);
    result.recv_counts.resize(nranks);
    result.recv_displs.resize(nranks);
    result.reorder_after_recv.resize(_loop_size);

    for (int i=0; i<nranks; ++i) {
        result.send_counts[i] = send_counts[i];
        result.send_displs[i] = send_displs[i];
        result.recv_counts[i] = recv_counts[i];
        result.recv_displs[i] = recv_displs[i];
    }

    index_t qfac = _nk * _nb * _nb * _nb * _nb;
    #pragma omp parallel for
    for (index_t x=0; x<_loop_size; ++x) {
        result.reorder_before_send[x] = symop[x].current_index - _qstarts[symop[x].current_rank]*qfac;
        result.phase_before_send[x] = symop[x].phase;
        result.reorder_after_recv[x] = recv_op[x].target_index - _qstarts[recv_op[x].target_rank]*qfac;
    }

    result.chan = chan;
    result.use_orbital_matrix = _st->getMatrixUse();
    if (result.use_orbital_matrix)
        result.orbital_matrix = _st->getOrbitalMatrices()[symm];

    return result;
}

LoopSymmetrizer::LoopSymmetrizer( SymmetryCommunicatorFactory* factory, dist_info* mpi_info ) {
    if (factory == nullptr) {
        _can_transform = false;
        _nsym = 0;
    } else {
        _can_transform = true;
        _mpi = mpi_info;
        _nsym = factory->get_nsym();
        _loop_size = factory->get_loop_size();
        for (int i=0; i<_nsym; ++i) {
            _comms_P.push_back( (*factory)( i, 'P' ) );
            _comms_C.push_back( (*factory)( i, 'C' ) );
        }
        _send_buf.resize( _loop_size );
        _recv_buf.resize( _loop_size );
        _loop_buf.resize( _loop_size );
    }
}

/*
static array<index_t,4> unravel_index( index_t idx, const array<index_t,4>& shape ) {
    array<index_t,4> indices;
    index_t rem = idx;

    idx /= (shape[1]*shape[2]*shape[3]);
    rem %= (shape[1]*shape[2]*shape[3]);
    indices[0] = idx;
    idx = rem;

    idx /= (shape[2]*shape[3]);
    rem %= (shape[2]*shape[3]);
    indices[1] = idx;
    idx = rem;

    idx /= (shape[3]);
    rem %= (shape[3]);
    indices[2] = idx;
    idx = rem;

    indices[3] = idx;
    return indices;
}

static index_t ravel_index( const array<index_t,4>& indices, const array<index_t,4>& shape ) {
    index_t x = indices[0] * shape[1]*shape[2]*shape[3] +
              indices[1] * shape[2]*shape[3] +
              indices[2] * shape[3] +
              indices[3];
    return x;
}
*/

int LoopSymmetrizer::operator()( complex128_t* loop_in, complex128_t* loop_out, index_t sym, char chan ) {
    if (!_can_transform) return 1;

    const SymmetryCommunication& comm = chan == 'P' ? _comms_P[sym] : _comms_C[sym];

    const index_t* send_map = comm.reorder_before_send.data();
    const complex128_t* send_phase = comm.phase_before_send.data();

    const index_t* send_counts = comm.send_counts.data();
    const index_t* send_displs = comm.send_displs.data();
    const index_t* recv_counts = comm.recv_counts.data();
    const index_t* recv_displs = comm.recv_displs.data();

    const index_t* recv_map = comm.reorder_after_recv.data();

    if (!comm.use_orbital_matrix) {
        #pragma omp parallel for
        for (index_t x=0; x<_loop_size; ++x) {
            _send_buf[x] = loop_in[send_map[x]] * send_phase[x];
        }
    } else {
        const MatXcd orbital_matrix = comm.orbital_matrix;
        const index_t nb = orbital_matrix.rows();
        memset((void*)_send_buf.data(), 0, sizeof(complex128_t)*_loop_size);
        #pragma omp parallel for
        for (index_t qk=0; qk<_loop_size/(nb*nb*nb*nb); ++qk) {
            index_t x = qk*nb*nb*nb*nb;
            for (index_t o=0; o<nb*nb*nb*nb; ++o)
            for (index_t op=0; op<nb*nb*nb*nb; ++op) {

                index_t O1 = (o%(nb*nb*nb*nb))/(nb*nb*nb),
                        O2 = (o%(nb*nb*nb))/(nb*nb),
                        O3 = (o%(nb*nb))/nb,
                        O4 = o%nb;
                index_t O1p = (op%(nb*nb*nb*nb))/(nb*nb*nb),
                        O2p = (op%(nb*nb*nb))/(nb*nb),
                        O3p = (op%(nb*nb))/nb,
                        O4p =  op%nb;
                index_t o1,o2,o3,o4,
                      o1p,o2p,o3p,o4p;
                if (chan == 'P') {
                    o1 = O1, o2 = O2, o3 = O3, o4 = O4,
                    o1p = O1p, o2p = O2p, o3p = O3p, o4p = O4p;
                } else {
                    o1 = O2, o2 = O3, o3 = O4, o4 = O1,
                    o1p = O2p, o2p = O3p, o3p = O4p, o4p = O1p;
                }


                _send_buf[x+op] += loop_in[send_map[x+o]] *
                    std::conj(orbital_matrix(o1p,o1)) *
                    std::conj(orbital_matrix(o2p,o2)) *
                             (orbital_matrix(o3p,o3)) *
                             (orbital_matrix(o4p,o4)) *
                              send_phase[x+o];
            }
        }
    }

    diverge_mpi_alltoallv_bytes( _send_buf.data(), send_counts, send_displs,
            _recv_buf.data(), recv_counts, recv_displs, sizeof(complex128_t) );

    #pragma omp parallel for
    for (index_t x=0; x<_loop_size; ++x) {
        loop_out[recv_map[x]] += _recv_buf[x];
    }

    return 0;
}

bool LoopSymmetrizer::check( complex128_t* loop, char chan ) {
    if (!_can_transform) return true;
    complex128_t* buf = _loop_buf.data();
    bool errors = false;
    mpi_err_printf( "checking '%c'\n", chan );
    for (int i=0; i<_nsym; ++i) {
        memset( (void*)buf, 0, sizeof(complex128_t)*_loop_size );
        this->operator()( loop, buf, i, chan );
        for (index_t x=0; x<_loop_size; ++x) {
            double err = std::abs( buf[x] - loop[x] );
            if (err > 1.e-13) {
                mpi_wrn_printf( "chan '%c', sym %i loop not symmetric (%li; %.5e%+.5ej - %.5e%+.5ej)\n",
                        chan, i, x, buf[x].real(), buf[x].real(), loop[x].real(), loop[x].imag() );
                errors = true;
            }
        }
    }
    return errors;
}

complex128_t* LoopSymmetrizer::operator()( complex128_t* loop, char chan ) {
    if (!_can_transform) return loop;
    complex128_t* out = _loop_buf.data();
    memset( (void*)out, 0, _loop_size*sizeof(complex128_t) );
    for (int i=0; i<_nsym; ++i)
        if (this->operator()( loop, out, i, chan )) {
            return loop;
        }
    double norm = 1./(double)_nsym;
    #pragma omp parallel for
    for (int i=0; i<_loop_size; ++i)
        out[i] *= norm;

    return out;
}

}
