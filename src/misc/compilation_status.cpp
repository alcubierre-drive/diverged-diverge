/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "mpi_log.h"
#include "compilation_status.h"
#include "version.h"
#include "number_repr_checks.h"

#include <cstring>
#include <string>
#include <vector>

using std::string;
using std::vector;

static string message( string msg, bool success ) {
    int use_color = mpi_log_get_colors();
    char _mark_avail[] = "✓";
    char _mark_not_avail[] = "✗";
    char _escape = 0x001b;
    char _color_bright_red[] = "0[31;1m";
    char _color_bright_green[] = "0[32;1m";
    char _color_reset[] = "0[0m";
    _color_bright_green[0] = _escape;
    _color_bright_red[0] = _escape;
    _color_reset[0] = _escape;
    string result = "";
    if (use_color) result += success ? _color_bright_green : _color_bright_red;
    result += msg;
    result += " ";
    result += success ? _mark_avail : _mark_not_avail;
    if (use_color) result += _color_reset;
    result += " ";
    return result;
}


int diverge_compilation_status_mpi( void ) {
    #ifdef USE_MPI
    #define _VERSION_IS_MPI_USED true
    #else
    #define _VERSION_IS_MPI_USED false
    #endif

    return _VERSION_IS_MPI_USED;
}

int diverge_compilation_status_cuda( char* str ) {
    #ifndef CUDA_VERSION_STR
    #define CUDA_VERSION_STR "CUDA"
    #endif

    #ifdef USE_CUDA
    #define _VERSION_IS_CUDA_USED true
    #else
    #define _VERSION_IS_CUDA_USED false
    #endif

    if (str) strcpy( str, CUDA_VERSION_STR );
    return _VERSION_IS_CUDA_USED;
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
    vector<string> compilation_status = {
        message( version_string, diverge_compilation_status_version(version_string) ),
        message( "MPI", diverge_compilation_status_mpi() ),
        message( cuda_string, diverge_compilation_status_cuda(cuda_string) ),
        message( "numbers", diverge_compilation_status_numbers() ),
    };
    mpi_ver_printf("");
    for (string c: compilation_status)
        mpi_eprintf("   %s", c.c_str());
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

