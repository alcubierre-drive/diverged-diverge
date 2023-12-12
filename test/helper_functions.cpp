#ifndef DIVERGE_SKIP_TESTS

#include "helper_functions.hpp"
#include "../src/misc/mpi_log.h"
#include "../src/misc/mpi_functions.h"
#include <algorithm>
#include <vector>

bool check_hermicity(complex128_t* data, index_t linsize,index_t batches){
    complex128_t tmp = 0.;
    for (index_t k = 0; k < batches; ++k)
    for (index_t i = 0; i < linsize; ++i)
    for (index_t j = 0; j < linsize; ++j) {
        tmp += std::abs(data[i+linsize*(j+linsize*k)]
                    - std::conj(data[j+linsize*(i+linsize*k)]));
    }
    return std::abs(tmp) < 1e-10;
}


complex128_t trace(complex128_t* data, index_t linsize, index_t batches){
    complex128_t tmp = 0.;
    for (index_t k = 0; k < batches; ++k)
    for (index_t i = 0; i < linsize; ++i) {
        tmp += data[i+linsize*(i+linsize*k)];
    }
    diverge_mpi_allreduce_complex_sum_inplace(&tmp, 1);
    return tmp;
}
complex128_t trace_batch_lin(complex128_t* data, index_t linsize, index_t batches){
    complex128_t tmp = 0.;
    for (index_t k = 0; k < batches; ++k)
    for (index_t i = 0; i < linsize; ++i) {
        tmp += data[k+batches*(i+linsize*i)];
    }
    diverge_mpi_allreduce_complex_sum_inplace(&tmp, 1);
    return tmp;
}

double trace(double* data, index_t linsize, index_t batches){
    double tmp = 0.;
    for (index_t k = 0; k < batches; ++k)
    for (index_t i = 0; i < linsize; ++i) {
        tmp += data[i+linsize*(i+linsize*k)];
    }
    diverge_mpi_allreduce_complex_sum_inplace(&tmp, 1);
    return tmp;
}

template<typename T>
bool check_equal_T( T* data1, T* data2, index_t size, double errbnd ) {
    std::vector<index_t> wrong_indices;
    wrong_indices.reserve( size/10 ); // heuristic
    for (index_t i=0; i<size; ++i) {
        double diff = std::abs( data1[i] - data2[i] );
        if (diff > errbnd) wrong_indices.push_back( i );
    }
    int sz = wrong_indices.size();
    std::vector<int> counts( diverge_mpi_comm_size(), 0 );
    diverge_mpi_allgather_int( &sz, counts.data(), 1 );
    std::vector<int> displs( counts.size(), 0 );
    for (unsigned i=1; i<counts.size(); ++i)
        displs[i] = counts[i-1] + displs[i-1];

    if (diverge_mpi_comm_rank() == 0)
        wrong_indices.resize( displs.back() + counts.back() );
    diverge_mpi_gatherv_index( wrong_indices.data(), sz, wrong_indices.data(),
            counts.data(), displs.data(), 0 );
    if (diverge_mpi_comm_rank() == 0) {
        std::sort( wrong_indices.begin(), wrong_indices.end() );
        wrong_indices.erase( std::unique( wrong_indices.begin(), wrong_indices.end() ), wrong_indices.end() );
        for (int i=0; i<(int)wrong_indices.size(); ++i) {
            int source_rank = 0;
            while (displs[source_rank++] < wrong_indices[i]) {if (source_rank==(int)counts.size()) break;}
            mpi_log_printf( "rank %02d: error at index %d (%.4e)\n",
                    source_rank, wrong_indices[i], std::abs(data1[wrong_indices[i]] - data2[wrong_indices[i]]) );
        }
    }
    return (displs.back() + counts.back()) == 0;
}

bool check_equal(complex128_t* data1, complex128_t* data2, index_t size, double errbnd) {
    return check_equal_T(data1, data2, size, errbnd);
}

bool check_equal(double* data1, double* data2, index_t size, double errbnd) {
    return check_equal_T(data1, data2, size, errbnd);
}

bool check_abs_sorted_equal(double* data1, double* data2, index_t size) {
    std::sort( data1, data1+size );
    std::sort( data2, data2+size );
    return check_equal_T(data1, data2, size, 1e-10);
}

bool check_abs_sorted_equal(complex128_t* data1, complex128_t* data2, index_t size) {
    auto complex_sort_functor = [](complex128_t x, complex128_t y){
        return std::abs(x) < std::abs(y);
    };
    std::sort( data1, data1+size, complex_sort_functor );
    std::sort( data2, data2+size, complex_sort_functor );
    return check_equal_T(data1, data2, size, 1e-10);
}

#endif // DIVERGE_SKIP_TESTS
