/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

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

void diverge_compilation_status( void ) {

    #ifndef CUDA_VERSION_STR
    #define CUDA_VERSION_STR "CUDA"
    #endif

    #ifdef USE_CUDA
    #define _VERSION_IS_CUDA_USED true
    #else // USE_CUDA
    #define _VERSION_IS_CUDA_USED false
    #endif // USE_CUDA

    #ifdef USE_MPI
    #define _VERSION_IS_MPI_USED true
    #else
    #define _VERSION_IS_MPI_USED false
    #endif

    bool number_representation = check_iec_double() &&
                                 check_iec_float() &&
                                 check_little_endian() &&
                                 check_8bit_char();

    vector<string> compilation_status = {
        message( get_version(), !is_version_dirty() ),
        message("MPI", _VERSION_IS_MPI_USED),
        message(CUDA_VERSION_STR, _VERSION_IS_CUDA_USED),
        message("numbers", number_representation ),
    };
    mpi_ver_printf("");
    for (string c: compilation_status)
        mpi_eprintf("   %s", c.c_str());
    mpi_eprintf("\n");

    if (!number_representation) {
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

