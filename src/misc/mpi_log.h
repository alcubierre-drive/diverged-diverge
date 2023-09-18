/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

// Function: mpi_printf
// these functions only produce an output on rank 0
void mpi_printf( const char* fmt, ... );
// Function: mpi_eprintf
void mpi_eprintf( const char* fmt, ... );
// Function: mpi_fprintf
void mpi_fprintf( FILE* file, const char* fmt, ... );

void mpi_dbg_printf_all( const char* logname, int loglevel, const char* fname, int line, const char* fmt, ... );
void mpi_dbg_printf( const char* logname, int loglevel, const char* fname, int line, const char* fmt, ... );
// Function: mpi_tim_printf
// debug time measurement (loglevel -1)
#define mpi_tim_printf( ... ) mpi_dbg_printf( "time", -1, __FILE__, __LINE__, __VA_ARGS__ );
// Function: mpi_err_printf
// print error message (loglevel 0)
#define mpi_err_printf( ... ) mpi_dbg_printf( "error", 0, __FILE__, __LINE__, __VA_ARGS__ );
// Function: mpi_wrn_printf
// print warning message (loglevel 1)
#define mpi_wrn_printf( ... ) mpi_dbg_printf( "warn", 1, __FILE__, __LINE__, __VA_ARGS__ );
// Function: mpi_scc_printf
// print success message (loglevel 2)
#define mpi_scc_printf( ... ) mpi_dbg_printf( "success", 2, __FILE__, __LINE__, __VA_ARGS__ );
// Function: mpi_log_printf_all
// print log message (loglevel 3) from all ranks
#define mpi_log_printf_all( ... ) mpi_dbg_printf_all( "log", 3, __FILE__, __LINE__, __VA_ARGS__ );
// Function: mpi_log_printf
// print log message (loglevel 3)
#define mpi_log_printf( ... ) mpi_dbg_printf( "log", 3, __FILE__, __LINE__, __VA_ARGS__ );
// Function: mpi_vrb_printf_all
// print verbose message (loglevel 5) from all ranks
#define mpi_vrb_printf_all( ... ) mpi_dbg_printf_all( "verbose", 5, __FILE__, __LINE__, __VA_ARGS__ );
// Function: mpi_vrb_printf
// print verbose message (loglevel 5, off by default)
#define mpi_vrb_printf( ... ) mpi_dbg_printf( "verbose", 5, __FILE__, __LINE__, __VA_ARGS__ );
// Function: mpi_fil_printf
// print file message (loglevel 2)
#define mpi_fil_printf( ... ) mpi_dbg_printf( "file", 2, __FILE__, __LINE__, __VA_ARGS__ );
// Function: mpi_usr_printf
// print user message (loglevel -1)
#define mpi_usr_printf( ... ) mpi_dbg_printf( "user", 2, __FILE__, __LINE__, __VA_ARGS__ );
// Function: mpi_ver_printf
// print version message (loglevel -1)
#define mpi_ver_printf( ... ) mpi_dbg_printf( "version", -1, __FILE__, __LINE__, __VA_ARGS__ );

// Function: mpi_loglevel_set
// set loglevel. all messages below this level are discarded and not printed
void mpi_loglevel_set( int loglevel );

// Function: mpi_log_set_colors
// turn colors for all messages on/off
void mpi_log_set_colors( int colors );

// Function: mpi_log_get_colors
// are colors used in logging?
int mpi_log_get_colors( void );

// Enum: mpi_log_control_t
//
// Members:
// mpi_log_control_standard - logs to stderr, printf() to stdout
// mpi_log_control_modified - logs to stdout, printf() to stdout
// mpi_log_control_stderr - logs to stderr, printf() to stderr
// mpi_log_control_inverted - logs to stdout, printf() to stderr
typedef enum {
    mpi_log_control_standard = 0,
    mpi_log_control_modified,
    mpi_log_control_stderr,
    mpi_log_control_inverted,
} mpi_log_control_t;

// Function: mpi_log_control
// control the output of diverge using <mpi_log_control_t>
void mpi_log_control( mpi_log_control_t ctl );

// Function: mpi_py_print
// print to stdout from python including the diverge log prefix
void mpi_py_print( const char* str );
// Function: mpi_py_eprint
// print to stderr from python including the diverge log prefix
void mpi_py_eprint( const char* str );

// Function: mpi_py_print_all
// print to stdout from python from all ranks including the diverge log prefix
void mpi_py_print_all( const char* str );
// Function: mpi_py_eprint_all
// print to stderr from python from all ranks including the diverge log prefix
void mpi_py_eprint_all( const char* str );

#ifdef __cplusplus
}
#endif
