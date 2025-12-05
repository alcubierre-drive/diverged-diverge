#define CT_SUITE_NAME qnan
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/misc/qnan.h"
#include "../../src/misc/mpi_functions.h"
#include "../../src/diverge_common.h"
#include <time.h>
#include <limits.h>

static void print_bits_64( const char* prefix, uint64_t x ) {
    char output[64+32] = {0};
    int output_idx = 0;
    for (int i = sizeof(x) * CHAR_BIT - 1; i >= 0; i--) {
        output[output_idx++] = (x & (1ULL << i)) ? '1' : '0';
        if (i % CHAR_BIT == 0) output[output_idx++] = ' ';
    }
    mpi_log_printf( "%s64bit pattern:\n%s\n", prefix, output );
}

static void print_bits_32( const char* prefix, uint32_t x ) {
    char output[32+32] = {0};
    int output_idx = 0;
    for (int i = sizeof(x) * CHAR_BIT - 1; i >= 0; i--) {
        output[output_idx++] = (x & (1U << i)) ? '1' : '0';
        if (i % CHAR_BIT == 0) output[output_idx++] = ' ';
    }
    mpi_log_printf( "%s32bit pattern:\n%s\n", prefix, output );
}

CT_TEST( int ) {
    uint32_t payload = clock();
    double qnan_with_payload = qnan_gen( payload );
    double regular_number = payload;

    uint64_t qnan_bits = 0; memcpy( &qnan_bits, &qnan_with_payload, sizeof qnan_bits );
    print_bits_64( "qnan ", qnan_bits );

    mpi_log_printf( "quiet NaN (%.1f) unsigned payload=%lu\n", qnan_with_payload, payload );
    CT_CHECK( qnan_isnan( qnan_with_payload ) );
    CT_CHECK( !qnan_isnan( regular_number ) );
    CT_CHECK( qnan_get( qnan_with_payload ) == payload );
}

CT_TEST( float ) {
    double payload = -(double)clock() / 1234.;
    double fnan_with_payload = fnan_gen( payload );
    double regular_number = payload;

    uint64_t fnan_bits = 0; memcpy( &fnan_bits, &fnan_with_payload, sizeof fnan_bits );
    print_bits_64( "fnan ", fnan_bits );

    mpi_log_printf( "quiet NaN (%.1f) floating point payload=%.5e\n", fnan_with_payload, payload );
    CT_CHECK( fnan_isnan( fnan_with_payload ) );
    CT_CHECK( !fnan_isnan( regular_number ) );

    double payload_trunc = fnan_get(fnan_with_payload);
    mpi_log_printf( "truncated vs full payload: %.18e %.18e\n", payload, payload_trunc );
    double denom = fabs(payload);
    if (denom < 1.e-2) denom = 1.;
    CT_CHECK( fabs(payload_trunc-payload)/denom < 1.e-1 );
}

CT_TEST( int32 ) {
    uint16_t payload = clock();
    float qnan_with_payload = qnanf_gen( payload );
    float regular_number = payload;

    uint32_t qnan_bits = 0; memcpy( &qnan_bits, &qnan_with_payload, sizeof qnan_bits );
    print_bits_32( "qnan ", qnan_bits );

    mpi_log_printf( "quiet NaNf (%.1f) unsigned payload=%u\n", qnan_with_payload, payload );
    CT_CHECK( qnanf_isnan( qnan_with_payload ) );
    CT_CHECK( !qnanf_isnan( regular_number ) );
    CT_CHECK( qnanf_get( qnan_with_payload ) == payload );
}

CT_TEST( float32 ) {
    float payload = -(float)clock() / 1234.;
    float fnan_with_payload = fnanf_gen( payload );
    float regular_number = payload;

    uint32_t fnan_bits = 0; memcpy( &fnan_bits, &fnan_with_payload, sizeof fnan_bits );
    print_bits_32( "fnan ", fnan_bits );

    mpi_log_printf( "quiet NaNf (%.1f) floating point payload=%.5e\n", fnan_with_payload, payload );
    CT_CHECK( fnanf_isnan( fnan_with_payload ) );
    CT_CHECK( !fnanf_isnan( regular_number ) );

    float payload_trunc = fnanf_get(fnan_with_payload);
    mpi_log_printf( "truncated vs full payload: %.18e %.18e\n", payload, payload_trunc );
    float denom = fabsf(payload);
    if (denom < 1.e-2) denom = 1.;
    CT_CHECK( fabsf(payload_trunc-payload)/denom < 1.e-1 );
}

#endif
