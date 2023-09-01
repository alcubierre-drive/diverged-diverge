#include "helper_functions.hpp"
#include "../src/misc/mpi_log.h"
#include "../src/misc/mpi_functions.h"
#include <algorithm>

bool check_hermicity(complex128_t* data, index_t linsize,index_t batches){
    complex128_t tmp = 0.;
    for (index_t k = 0; k < batches; ++k)
    for (index_t i = 0; i < linsize; ++i)
    for (index_t j = 0; j < linsize; ++j) {
        tmp += std::abs(data[i+linsize*(j+linsize*k)]
                    - std::conj(data[j+linsize*(i+linsize*k)]));
    }
    return abs(tmp) < 1e-10;
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

bool check_equal(complex128_t* data1, complex128_t* data2, index_t size, double errbnd)
{
    double max_diff = 0.0;
    for (index_t i = 0; i < size;++i) {
        if (std::abs(data1[i] - data2[i]) > max_diff) {
            max_diff += abs(data1[i] - data2[i]);
            if (max_diff > errbnd) {
                mpi_log_printf_all("rank %2d: maximal difference is %f at %f and %f at idx %d\n",
                        diverge_mpi_comm_rank(), max_diff, ((double*)data1)[2*i], ((double*)data2)[2*i], (int)i );
            }
        }
    }
    return max_diff < errbnd;
}
bool check_equal(double* data1, double* data2, index_t size, double errbnd)
{
    double max_diff = 0.0;
    for (index_t i = 0; i < size;++i) {
        if (std::abs(data1[i] - data2[i]) > max_diff) {
            max_diff += abs(data1[i] - data2[i]);
            if (max_diff > errbnd) {
                mpi_log_printf_all("rank %2d: maximal difference is %f at idx %d\n",
                        diverge_mpi_comm_rank(), max_diff, (int)i );
            }
        }
    }
    return max_diff < errbnd;
}

bool check_abs_sorted_equal(double* data1, double* data2, index_t size) {
    std::sort( data1, data1+size );
    std::sort( data2, data2+size );
    double max_diff = 0.0;
    for (index_t i = 0; i < size;++i) {
        if (std::abs(data1[i] - data2[i]) > max_diff) {
            max_diff += std::abs(data1[i] - data2[i]);
            if (max_diff > 1e-10) {
                mpi_log_printf_all("rank %2d: maximal difference is %f at idx %d\n",
                        diverge_mpi_comm_rank(), max_diff,(int)i );
            }
        }
    }
    return max_diff < 1e-10;
}

bool check_abs_sorted_equal(complex128_t* data1, complex128_t* data2, index_t size) {
    auto complex_sort_functor = [](complex128_t x, complex128_t y){
        return std::abs(x) < std::abs(y);
    };
    std::sort( data1, data1+size, complex_sort_functor );
    std::sort( data2, data2+size, complex_sort_functor );
    double max_diff = 0.0;
    for (index_t i = 0; i < size;++i) {
        if (std::abs(std::abs(data1[i]) - std::abs(data2[i])) > max_diff) {
            max_diff += std::abs(std::abs(data1[i]) - std::abs(data2[i]));
            if (max_diff > 1e-10) {
                mpi_log_printf_all("rank %2d: maximal difference is %f at %f and %f at idx %d\n",
                        diverge_mpi_comm_rank(), max_diff, std::abs(data1[i]),  std::abs(data2[i]), (int)i );
            }
        }
    }
    return max_diff < 1e-10;
}
