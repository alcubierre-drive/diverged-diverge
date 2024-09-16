/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "mpi_log.h"
#include "compilation_status.h"
#include "version.h"
#include "number_repr_checks.h"

#include <string.h>

#ifndef DIVERGE_MARK_OK
#define DIVERGE_MARK_OK "✓"
#endif

#ifndef DIVERGE_MARK_NOT_OK
#define DIVERGE_MARK_NOT_OK "✗"
#endif

#define MSG_STR_SZ 1024
static inline char* strxcat( char* dest, const char* src ) {
    return strncat( dest, src, MSG_STR_SZ-1 );
}

static void message( const char* msg, bool success, char* result ) {
    int use_color = mpi_log_get_colors();
    const char _mark_avail[16] = DIVERGE_MARK_OK;
    const char _mark_not_avail[16] = DIVERGE_MARK_NOT_OK;
    const char _color_bright_red[16] = "\x1b[31;1m";
    const char _color_bright_green[16] = "\x1b[32;1m";
    const char _color_reset[16] = "\x1b[0m";
    if (use_color) strxcat( result, success ? _color_bright_green : _color_bright_red );
    strxcat( result, msg );
    strxcat( result, " " );
    strxcat( result, success ? _mark_avail : _mark_not_avail );
    if (use_color) strxcat( result, _color_reset );
    strxcat( result, " " );
}

int diverge_compilation_status_mpi( void ) {
    #ifdef USE_MPI
    return true;
    #else
    return false;
    #endif
}

int diverge_compilation_status_cuda( char* str ) {
    #ifndef CUDA_VERSION_STR
    #define CUDA_VERSION_STR "CUDA"
    #endif
    if (str) strcpy( str, CUDA_VERSION_STR );
    #ifdef USE_CUDA
    return true;
    #else
    return false;
    #endif
}

int diverge_compilation_status_version( char* str ) {
    if (str) strcpy( str, get_version() );
    return !is_version_dirty();
}

int diverge_compilation_status_numbers( void ) {
    bool number_representation = check_iec_double() &&
                                 check_iec_float() &&
                                 check_little_endian() &&
                                 check_8bit_char();
    return number_representation;
}

void diverge_compilation_status( void ) {

    char version_string[256] = {0};
    char cuda_string[256] = {0};
    char compilation_status[4][MSG_STR_SZ] = {0};
    int version_status = diverge_compilation_status_version(version_string),
        cuda_status = diverge_compilation_status_cuda(cuda_string);
    message( version_string, version_status, compilation_status[0] );
    message( "MPI", diverge_compilation_status_mpi(), compilation_status[1] );
    message( cuda_string, cuda_status, compilation_status[2] );
    message( "numbers", diverge_compilation_status_numbers(), compilation_status[3] );
    mpi_ver_printf("");
    for (unsigned int i=0; i<sizeof(compilation_status)/sizeof(compilation_status[0]); ++i)
        mpi_eprintf("   %s", compilation_status[i]);
    mpi_eprintf("\n");

    if (!diverge_compilation_status_numbers()) {
        if (!check_iec_double())
            mpi_wrn_printf("double is not conformant to iec559. binary files may be corrupted.\n");
        if (!check_iec_float())
            mpi_wrn_printf("float is not conformant to iec559. binary files may be corrupted.\n");
        if (!check_little_endian())
            mpi_wrn_printf("int or int64_t is not little endian. binary files may be corrupted.\n");
        if (!check_8bit_char())
            mpi_wrn_printf("1 byte != 8 bit. implementation may fail.\n");
    }
    fflush(stderr);
}

