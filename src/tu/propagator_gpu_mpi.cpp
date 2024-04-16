#include "propagator_gpu_mpi.h"
#include "../misc/mpi_functions.h"

#include <algorithm>
#ifdef USE_BOOST_SORT

// ignore boost warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <boost/sort/sort.hpp>
#pragma GCC diagnostic pop
#define _CUSTOM_SORT_(begin, end, comp) boost::sort::block_indirect_sort(begin, end, comp, diverge_omp_num_threads())
#define _CUSTOM_STABLE_SORT_(begin, end, comp) boost::sort::parallel_stable_sort(begin, end, comp, diverge_omp_num_threads())

#else // USE_BOOST_SORT

#define _CUSTOM_SORT_(begin, end, comp) std::sort(begin, end, comp)
#define _CUSTOM_STABLE_SORT_(begin, end, comp) std::stable_sort(begin, end, comp)

#endif // USE_BOOST_SORT

// no boost warning about negative size_t stuff on intel compilers
#ifdef __INTEL_COMPILER
#pragma warning disable 68
#endif

void sort_pack( pack_t* start, pack_t* end ) {
    _CUSTOM_STABLE_SORT_( start, end, []( const pack_t& a, const pack_t& b ) {
            return a.dest_rank < b.dest_rank;
    } );
}
