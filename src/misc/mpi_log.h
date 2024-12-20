/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

/** act like ``printf(fmt, ...)``, but only produce an output on rank 0 */
void mpi_printf( const char* fmt, ... );
/** act like ``fprintf(stderr, fmt, ...)``, but only produce an output on rank 0 */
void mpi_eprintf( const char* fmt, ... );
/** act like ``fprintf(file, fmt, ...)``, but only produce an output on rank 0 */
void mpi_fprintf( FILE* file, const char* fmt, ... );

void mpi_dbg_printf_all( const char* logname, int loglevel, const char* fname, int line, const char* fmt, ... );
void mpi_dbg_printf( const char* logname, int loglevel, const char* fname, int line, const char* fmt, ... );
/** debug time measurement (loglevel -1) */
#define mpi_tim_printf( ... ) mpi_dbg_printf( "time", -1, __FILE__, __LINE__, __VA_ARGS__ );
/** print error message (loglevel 0) */
#define mpi_err_printf( ... ) mpi_dbg_printf( "error", 0, __FILE__, __LINE__, __VA_ARGS__ );
/** print error message on all ranks (loglevel 0) */
#define mpi_err_printf_all( ... ) mpi_dbg_printf_all( "error", 0, __FILE__, __LINE__, __VA_ARGS__ );
/** print warning message (loglevel 1) */
#define mpi_wrn_printf( ... ) mpi_dbg_printf( "warn", 1, __FILE__, __LINE__, __VA_ARGS__ );
/** print success message (loglevel 2) */
#define mpi_scc_printf( ... ) mpi_dbg_printf( "success", 2, __FILE__, __LINE__, __VA_ARGS__ );
/** print log message (loglevel 3) from all ranks */
#define mpi_log_printf_all( ... ) mpi_dbg_printf_all( "log", 3, __FILE__, __LINE__, __VA_ARGS__ );
/** print log message (loglevel 3) */
#define mpi_log_printf( ... ) mpi_dbg_printf( "log", 3, __FILE__, __LINE__, __VA_ARGS__ );
/** print verbose message (loglevel 5, off by default) from all ranks */
#define mpi_vrb_printf_all( ... ) mpi_dbg_printf_all( "verbose", 5, __FILE__, __LINE__, __VA_ARGS__ );
/** print verbose message (loglevel 5, off by default) */
#define mpi_vrb_printf( ... ) mpi_dbg_printf( "verbose", 5, __FILE__, __LINE__, __VA_ARGS__ );
/** print file message (loglevel 2) */
#define mpi_fil_printf( ... ) mpi_dbg_printf( "file", 2, __FILE__, __LINE__, __VA_ARGS__ );
/** print user message (loglevel -1) */
#define mpi_usr_printf( ... ) mpi_dbg_printf( "user", 2, __FILE__, __LINE__, __VA_ARGS__ );
/** print version message (loglevel -1) */
#define mpi_ver_printf( ... ) mpi_dbg_printf( "version", -1, __FILE__, __LINE__, __VA_ARGS__ );
/** print backtrace (loglevel 5, off by default) */
#define mpi_btr_printf( ... ) mpi_dbg_printf( "trace", 5, __FILE__, __LINE__, __VA_ARGS__ );

/** set loglevel. all messages below this level are discarded and not printed */
void mpi_loglevel_set( int loglevel );

/** get current loglevel */
int mpi_loglevel_get( void );

/** turn colors for all messages on/off */
void mpi_log_set_colors( int colors );
/** are colors used in logging? */
int mpi_log_get_colors( void );

/** more control over logging behavior */
typedef enum {
    /** logs to stderr, printf() to stdout */
    mpi_log_control_standard = 0,
    /** logs to stdout, printf() to stdout */
    mpi_log_control_modified,
    /** logs to stderr, printf() to stderr */
    mpi_log_control_stderr,
    /** logs to stdout, printf() to stderr */
    mpi_log_control_inverted,
} mpi_log_control_t;

/** control the output of diverge using :c:enum:`mpi_log_control_t` */
void mpi_log_control( mpi_log_control_t ctl );

/** print to stdout from python including the diverge log prefix */
void mpi_py_print( const char* str );
/** print to stderr from python including the diverge log prefix */
void mpi_py_eprint( const char* str );

/** print to stdout from python from all ranks including the diverge log prefix */
void mpi_py_print_all( const char* str );
/** print to stderr from python from all ranks including the diverge log prefix */
void mpi_py_eprint_all( const char* str );

#ifdef __cplusplus
}
#endif
