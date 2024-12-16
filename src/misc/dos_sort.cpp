#include "dos_sort.h"
#include "mpi_functions.h"

#include <algorithm>
#ifdef USE_BOOST_SORT

// ignore boost warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <boost/sort/sort.hpp>
#pragma GCC diagnostic pop
#define _CUSTOM_SORT_(begin, end, comp) boost::sort::block_indirect_sort(begin, end, comp, diverge_omp_num_threads())
#else // USE_BOOST_SORT
#define _CUSTOM_SORT_(begin, end, comp) std::sort(begin, end, comp)
#endif // USE_BOOST_SORT
// no boost warning about negative size_t stuff on intel compilers
#ifdef __INTEL_COMPILER
#pragma warning disable 68
#endif

void dos_sort_doubles( double* start, index_t size ) {
    auto cmp = [](const double& A, const double& B)->bool{
        return A < B;
    };
    _CUSTOM_SORT_( start, start+size, cmp );
}

void dos_sort_doubles_indirect( index_t* start, index_t size, const double* vals ) {
    auto cmp = [vals](const index_t& A, const index_t& B)->bool{
        return vals[A] < vals[B];
    };
    _CUSTOM_SORT_( start, start+size, cmp );
}
