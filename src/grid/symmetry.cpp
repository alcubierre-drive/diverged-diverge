/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "symmetry.hpp"
#include "../misc/mergesort.h"
#include "../diverge_internals_struct.h"

namespace grid {

typedef Eigen::Matrix<index_t,-1,-1> MatXi;
typedef Eigen::Matrix<index_t,-1,1> VecXi;

struct symm_props {
    int idx_pre_symm;
    int idx_post_symm;
    int R1, R2, R3;
};

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
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
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
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
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

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
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

#undef MERGESORT_COMP
#define MERGESORT_COMP( x, y ) ( (x).target_rank - (y).target_rank )
MERGESORT_IMPL( mergesort, SymmetryOperation )

SymmetryCommunicatorFactory::SymmetryCommunicatorFactory( SymmetryTransform* st, const index_t* k1m2,
        const index_t* k1p2, index_t qstart, index_t qstop, index_t nk, index_t nb ) {
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
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
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
        mergesort( _symops_P[i], _loop_size, diverge_omp_num_threads(), 0 );
        mergesort( _symops_C[i], _loop_size, diverge_omp_num_threads(), 0 );
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
    index_t* send_counts = (index_t*)calloc(nranks, sizeof*send_counts);
    for (index_t i=0; i<_loop_size; ++i) send_counts[symop[i].target_rank]++;
    index_t* send_displs = (index_t*)calloc(nranks, sizeof*send_displs);
    for (int i=1; i<nranks; ++i) send_displs[i] = send_displs[i-1] + send_counts[i-1];

    index_t* all_send_counts = (index_t*)calloc(POW2(nranks), sizeof*all_send_counts);
    diverge_mpi_allgather_index( send_counts, all_send_counts, nranks );

    index_t* recv_counts = (index_t*)calloc(nranks, sizeof*recv_counts);
    index_t* recv_displs = (index_t*)calloc(nranks, sizeof*recv_displs);
    for (int i=0; i<nranks; ++i) recv_counts[i] = all_send_counts[IDX2(i,rank,nranks)];
    for (int i=1; i<nranks; ++i) recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];

    vector<SymmetryOperation> recv_op(_loop_size);

    diverge_mpi_alltoallv_bytes( symop, send_counts, send_displs, recv_op.data(),
            recv_counts, recv_displs, sizeof(SymmetryOperation) );

    SymmetryCommunication result;

    result.reorder_before_send = (index_t*)calloc(_loop_size, sizeof*result.reorder_before_send);
    result.phase_before_send = (complex128_t*)calloc(_loop_size, sizeof*result.phase_before_send);
    result.reorder_after_recv = (index_t*)calloc(_loop_size, sizeof*result.reorder_after_recv);
    result.send_counts = send_counts;
    result.send_displs = send_displs;
    result.recv_counts = recv_counts;
    result.recv_displs = recv_displs;

    free( all_send_counts );

    index_t qfac = _nk * _nb * _nb * _nb * _nb;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (index_t x=0; x<_loop_size; ++x) {
        result.reorder_before_send[x] = symop[x].current_index - _qstarts[symop[x].current_rank]*qfac;
        result.phase_before_send[x] = symop[x].phase;
        result.reorder_after_recv[x] = recv_op[x].target_index - _qstarts[recv_op[x].target_rank]*qfac;
    }

    result.chan = chan;
    if (_st->getMatrixUse()) {
        MatXcd M = _st->getOrbitalMatrices()[symm];
        result.nb = M.rows();
        result.orbital_matrix = (complex128_t*)calloc(POW2(result.nb), sizeof*result.orbital_matrix);
        Map<MatXcd>( result.orbital_matrix, result.nb, result.nb ) = M.transpose();
    } else {
        result.orbital_matrix = NULL;
        result.nb = 0;
    }

    return result;
}

void SymmetryCommunication_free( SymmetryCommunication S ) {
    free( S.reorder_before_send );
    free( S.phase_before_send );
    free( S.send_counts );
    free( S.send_displs );
    free( S.recv_counts );
    free( S.recv_displs );
    free( S.reorder_after_recv );
    free( S.orbital_matrix );
}

LoopSymmetrizer::LoopSymmetrizer( SymmetryCommunicatorFactory* factory ) {
    if (factory == nullptr) {
        _can_transform = false;
        _nsym = 0;
    } else {
        _can_transform = true;
        _nsym = factory->get_nsym();
        _loop_size = factory->get_loop_size();
        _comms_P = (SymmetryCommunication*)calloc(_nsym, sizeof*_comms_P);
        _comms_C = (SymmetryCommunication*)calloc(_nsym, sizeof*_comms_P);
        for (int i=0; i<_nsym; ++i) {
            _comms_P[i] = (*factory)( i, 'P' );
            _comms_C[i] = (*factory)( i, 'C' );
        }
        _send_buf.resize( _loop_size );
        _recv_buf.resize( _loop_size );
        _loop_buf.resize( _loop_size );
    }
}

LoopSymmetrizer::~LoopSymmetrizer() {
    for (int i=0; i<_nsym; ++i)
        SymmetryCommunication_free( _comms_P[i] ),
        SymmetryCommunication_free( _comms_C[i] );
    free( _comms_P );
    free( _comms_C );
}

int LoopSymmetrizer::operator()( complex128_t* loop_in, complex128_t* loop_out, index_t sym, char chan ) {
    if (!_can_transform) return 1;

    const SymmetryCommunication& comm = chan == 'P' ? _comms_P[sym] : _comms_C[sym];

    const index_t* send_map = comm.reorder_before_send;
    const complex128_t* send_phase = comm.phase_before_send;

    const index_t* send_counts = comm.send_counts;
    const index_t* send_displs = comm.send_displs;
    const index_t* recv_counts = comm.recv_counts;
    const index_t* recv_displs = comm.recv_displs;

    const index_t* recv_map = comm.reorder_after_recv;

    if (!comm.orbital_matrix) {
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t x=0; x<_loop_size; ++x)
            _send_buf[x] = loop_in[send_map[x]] * send_phase[x];
    } else {
        const index_t nb = comm.nb;
        const complex128_t* orbital_matrix = comm.orbital_matrix;
        ZERO_OUT( _send_buf.data(), _loop_size );
        #pragma omp parallel for num_threads(diverge_omp_num_threads())
        for (index_t qk=0; qk<_loop_size/(nb*nb*nb*nb); ++qk) {
            index_t x = qk*nb*nb*nb*nb;
            for (index_t o=0; o<nb*nb*nb*nb; ++o)
            for (index_t op=0; op<nb*nb*nb*nb; ++op) {

                const index_t dim[4] = {nb,nb,nb,nb};
                index_t O[4]; unravel( o, dim, 4, O );
                index_t Op[4]; unravel( op, dim, 4, Op );
                index_t o1,o2,o3,o4,
                        o1p,o2p,o3p,o4p;
                if (chan == 'P') {
                    o1 = O[0], o2 = O[1], o3 = O[2], o4 = O[3],
                    o1p = Op[0], o2p = Op[1], o3p = Op[2], o4p = Op[3];
                } else {
                    o1 = O[1], o2 = O[2], o3 = O[3], o4 = O[0],
                    o1p = Op[1], o2p = Op[2], o3p = Op[3], o4p = Op[0];
                }


                _send_buf[x+op] += loop_in[send_map[x+o]] *
                    std::conj(orbital_matrix[IDX2(o1p,o1,nb)]) *
                    std::conj(orbital_matrix[IDX2(o2p,o2,nb)]) *
                             (orbital_matrix[IDX2(o3p,o3,nb)]) *
                             (orbital_matrix[IDX2(o4p,o4,nb)]) * send_phase[x+o];
            }
        }
    }

    diverge_mpi_alltoallv_bytes( _send_buf.data(), send_counts, send_displs,
            _recv_buf.data(), recv_counts, recv_displs, sizeof(complex128_t) );

    #pragma omp parallel for num_threads(diverge_omp_num_threads())
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
        ZERO_OUT( buf, _loop_size );
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
    ZERO_OUT( out, _loop_size );
    for (int i=0; i<_nsym; ++i)
        if (this->operator()( loop, out, i, chan )) {
            return loop;
        }
    double norm = 1./(double)_nsym;
    #pragma omp parallel for num_threads(diverge_omp_num_threads())
    for (int i=0; i<_loop_size; ++i)
        out[i] *= norm;

    return out;
}

}
