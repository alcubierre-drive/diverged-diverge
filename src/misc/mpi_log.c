/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "mpi_log.h"
#include "mpi_functions.h"
#include "init_internal_libs.h"
#include "../diverge_internals_struct.h" // strdup_div
#include <stdarg.h>

#ifndef DIVERGE_LOG_PREFIX
#define DIVERGE_LOG_PREFIX "divERGe"
#endif

#ifdef DIVERGE_LOG_NAMELESS
#define DIVERGE_LOG_PREFIX_DIV ""
#else
#define DIVERGE_LOG_PREFIX_DIV ":"
#endif
// #define DIVERGE_LOG_COLORLESS

static mpi_loglevel_t h_diverge_mpi_loglevel = mpi_loglevel_log;
static int h_diverge_log_colors = 1;
static mpi_log_control_t h_diverge_log_ctl = mpi_log_control_standard;

static inline int zero_rank_or_no_init( void ) {
    if (!diverge_has_init_internal_libs()) return 1;
    return (diverge_mpi_comm_rank() == 0);
}

void mpi_log_control( mpi_log_control_t ctl ) {
    #pragma omp critical
    h_diverge_log_ctl = ctl;
}

static FILE* h_mpi_log_stdout( void ) {
    FILE* result = NULL;
    switch (h_diverge_log_ctl) {
        default:
        case mpi_log_control_standard:
        case mpi_log_control_modified: result = stdout; break;

        case mpi_log_control_stderr:
        case mpi_log_control_inverted: result = stderr; break;
    }
    return result;
}

static FILE* h_mpi_log_stderr( void ) {
    FILE* result = NULL;
    switch (h_diverge_log_ctl) {
        default:
        case mpi_log_control_standard:
        case mpi_log_control_stderr: result = stderr; break;

        case mpi_log_control_modified:
        case mpi_log_control_inverted: result = stdout; break;
    }
    return result;
}

void mpi_printf( const char* fmt, ... ) {
    va_list ap;
    va_start( ap, fmt );
    if (zero_rank_or_no_init())
        vfprintf( h_mpi_log_stdout(), fmt, ap );
    va_end( ap );
}

static void mpi_vprintf( const char* fmt, va_list ap ) {
    if (zero_rank_or_no_init())
        vfprintf( h_mpi_log_stdout(), fmt, ap );
}

void mpi_eprintf( const char* fmt, ... ) {
    va_list ap;
    va_start(ap, fmt);
    if (zero_rank_or_no_init())
        vfprintf( h_mpi_log_stderr(), fmt, ap );
    va_end(ap);
}

static void mpi_veprintf( const char* fmt, va_list ap ) {
    if (zero_rank_or_no_init())
        vfprintf( h_mpi_log_stderr(), fmt, ap );
}

void mpi_fprintf( FILE* file, const char* fmt, ... ) {
    va_list ap;
    va_start(ap, fmt);
    if (zero_rank_or_no_init())
        vfprintf( file, fmt, ap );
    va_end(ap);
}

#ifndef DIVERGE_LOG_COLORLESS
static const char h_log_color_black[] = "[30m";
static const char h_log_color_red[] = "[31m";
static const char h_log_color_green[] = "[32m";
static const char h_log_color_yellow[] = "[33m";
static const char h_log_color_blue[] = "[34m";
static const char h_log_color_magenta[] = "[35m";
static const char h_log_color_cyan[] = "[36m";
static const char h_log_color_white[] = "[37m";
static const char h_log_color_bright_black[] = "[30;1m";
static const char h_log_color_bright_red[] = "[31;1m";
static const char h_log_color_bright_green[] = "[32;1m";
static const char h_log_color_bright_yellow[] = "[33;1m";
static const char h_log_color_bright_blue[] = "[34;1m";
static const char h_log_color_bright_magenta[] = "[35;1m";
static const char h_log_color_bright_cyan[] = "[36;1m";
static const char h_log_color_bright_gray[] = "[38;5;248;1m";
#endif
static const char h_log_color_bright_white[] = "[37;1m";
static const char h_log_color_reset[] = "[0m";

#ifdef DIVERGE_LOG_GRAYSCALE
static const char h_log_color_gray0[] = "[38;5;248;1m";
static const char h_log_color_gray1[] = "[38;5;246;1m";
static const char h_log_color_gray2[] = "[38;5;244;1m";
static const char h_log_color_gray3[] = "[38;5;242;1m";
static const char h_log_color_gray4[] = "[38;5;240;1m";
static const char h_log_color_gray5[] = "[38;5;238;1m";
#endif

static char h_log_color_return[16] = { 0x001b };

#define h_log_color_RESET h_log_color_reset
#ifndef DIVERGE_LOG_GRAYSCALE
#define h_log_color_LOG h_log_color_bright_blue
#define h_log_color_VERBOSE h_log_color_bright_gray
#define h_log_color_SUCCESS h_log_color_bright_green
#define h_log_color_ERROR h_log_color_bright_red
#define h_log_color_WARN h_log_color_bright_yellow
#define h_log_color_TIME h_log_color_bright_magenta
#define h_log_color_FILE h_log_color_bright_cyan
#define h_log_color_USER h_log_color_bright_white
#define h_log_color_VERSION h_log_color_bright_green
#define h_log_color_log h_log_color_bright_blue
#define h_log_color_verbose h_log_color_bright_gray
#define h_log_color_success h_log_color_bright_green
#define h_log_color_error h_log_color_bright_red
#define h_log_color_warn h_log_color_bright_yellow
#define h_log_color_time h_log_color_bright_magenta
#define h_log_color_file h_log_color_bright_cyan
#define h_log_color_user h_log_color_bright_white
#define h_log_color_version h_log_color_bright_green
#define h_log_color_trace h_log_color_bright_white
#else // DIVERGE_LOG_GRAYSCALE
#define h_log_color_VERBOSE h_log_color_gray5
#define h_log_color_LOG     h_log_color_gray4
#define h_log_color_FILE    h_log_color_gray3
#define h_log_color_USER    h_log_color_gray3
#define h_log_color_WARN    h_log_color_gray2
#define h_log_color_SUCCESS h_log_color_gray3
#define h_log_color_ERROR   h_log_color_gray1
#define h_log_color_TIME    h_log_color_gray0
#define h_log_color_VERSION h_log_color_gray0
#define h_log_color_verbose h_log_color_gray5
#define h_log_color_log     h_log_color_gray4
#define h_log_color_file    h_log_color_gray3
#define h_log_color_user    h_log_color_gray3
#define h_log_color_warn    h_log_color_gray2
#define h_log_color_success h_log_color_gray3
#define h_log_color_error   h_log_color_gray1
#define h_log_color_time    h_log_color_gray0
#define h_log_color_version h_log_color_gray0
#define h_log_color_trace   h_log_color_gray0
#endif // DIVERGE_LOG_GRAYSCALE

static const char* h_log_color( const char* descr ) {
    if (!h_diverge_log_colors) {
        h_log_color_return[0] = '\0';
        return h_log_color_return;
    }
    strcpy( h_log_color_return+1, h_log_color_reset );
#ifdef DIVERGE_LOG_COLORLESS
    if (!(!strcmp( descr, "reset" ) || !strcmp( descr, "RESET" )))
        strcpy( h_log_color_return+1, h_log_color_user );
#else // DIVERGE_LOG_COLORLESS
#define H_RETPART( DESCR ) \
    if (!strcmp( descr, #DESCR )) { strcpy( h_log_color_return+1, h_log_color_##DESCR ); }
    H_RETPART( black )
    H_RETPART( red )
    H_RETPART( green )
    H_RETPART( yellow )
    H_RETPART( blue )
    H_RETPART( magenta )
    H_RETPART( cyan )
    H_RETPART( white )
    H_RETPART( bright_black )
    H_RETPART( bright_red )
    H_RETPART( bright_green )
    H_RETPART( bright_yellow )
    H_RETPART( bright_blue )
    H_RETPART( bright_magenta )
    H_RETPART( bright_cyan )
    H_RETPART( bright_white )
    H_RETPART( bright_gray )
    H_RETPART( reset )
    H_RETPART( RESET )
    H_RETPART( LOG )
    H_RETPART( VERBOSE )
    H_RETPART( SUCCESS )
    H_RETPART( ERROR )
    H_RETPART( WARN )
    H_RETPART( TIME )
    H_RETPART( FILE )
    H_RETPART( USER )
    H_RETPART( VERSION )
    H_RETPART( log )
    H_RETPART( verbose )
    H_RETPART( success )
    H_RETPART( error )
    H_RETPART( warn )
    H_RETPART( time )
    H_RETPART( file )
    H_RETPART( user )
    H_RETPART( version )
    H_RETPART( trace )
#endif // DIVERGE_LOG_COLORLESS
    return h_log_color_return;
}

void mpi_loglevel_set( mpi_loglevel_t loglevel ) {
    h_diverge_mpi_loglevel = loglevel;
}

mpi_loglevel_t mpi_loglevel_get( void ) {
    return h_diverge_mpi_loglevel;
}

void mpi_log_set_colors( int colors ) {
    h_diverge_log_colors = colors;
}

int mpi_log_get_colors( void ) {
    return h_diverge_log_colors;
}

void mpi_lll_printf( mpi_loglevel_t loglevel, const char* fmt, ... ) {
    if (h_diverge_mpi_loglevel < loglevel) return;
    va_list ap;
    va_start(ap, fmt);
    mpi_vprintf(fmt, ap);
    va_end(ap);
}

void mpi_dbg_printf( const char* logname_, mpi_loglevel_t loglevel, const char* fname,
        int line, const char* fmt, ... ) {

    if (h_diverge_mpi_loglevel < loglevel) return;

    va_list ap;
    va_start(ap, fmt);

#ifdef DIVERGE_LOG_NAMELESS
    const char logname[] = "";
#else
    const char* logname = logname_;
#endif

    #define behaviorIncludeLine 0
    #define behaviorDiscardLine 1
#ifdef NDEBUG
    #ifndef NDEBUG_SKIP_ERROR_LINES
    char behavior = (!strcmp(logname_,"error") || !strcmp(logname_,"ERROR")) ? behaviorIncludeLine : behaviorDiscardLine;
    #else // NDEBUG_SKIP_ERROR_LINES
    char behavior = behaviorDiscardLine;
    #endif // NDEBUG_SKIP_ERROR_LINES
#else // NDEBUG
    char behavior = behaviorIncludeLine;
#endif // NDEBUG

    switch (behavior) {
        default:
        case behaviorIncludeLine:
            mpi_eprintf( "%s[" DIVERGE_LOG_PREFIX "%s%s %s:%i] ", h_log_color( logname_ ), DIVERGE_LOG_PREFIX_DIV, logname, fname, line );
            break;
        case behaviorDiscardLine:
            mpi_eprintf( "%s[" DIVERGE_LOG_PREFIX "%s%s] ", h_log_color( logname_ ), DIVERGE_LOG_PREFIX_DIV, logname );
            break;
    }
    mpi_eprintf( "%s", h_log_color("RESET") );

    mpi_veprintf( fmt, ap );
    fflush(h_mpi_log_stderr());
    va_end(ap);
}

#ifdef DIVERGE_LOG_NAMELESS
static const char py_lg_name[] = "";
#else
static const char py_lg_name[] = "py";
#endif

void mpi_py_print( const char* str ) {
    mpi_printf( "%s[%s%s%s] ", h_log_color( "user" ),
            DIVERGE_LOG_PREFIX, DIVERGE_LOG_PREFIX_DIV, py_lg_name );
    mpi_printf( h_log_color("RESET") );
    mpi_printf( "%s\n", str );
}

void mpi_py_eprint( const char* str ) {
    mpi_eprintf( "%s[%s%s%s] ", h_log_color( "user" ),
            DIVERGE_LOG_PREFIX, DIVERGE_LOG_PREFIX_DIV, py_lg_name );
    mpi_eprintf( h_log_color("RESET") );
    mpi_eprintf( "%s\n", str );
}

static char** mpi_string_gather( const char* str ) {
    int nranks = diverge_mpi_comm_size(),
        rank = diverge_mpi_comm_rank();
    int* counts = calloc( nranks, sizeof(int) );
    int* displs = calloc( nranks, sizeof(int) );
    for (int r=0; r<nranks; ++r) {
        counts[r] = sizeof(int);
        displs[r] = sizeof(int)*r;
    }

    int* lengths = calloc( nranks, sizeof(int) );
    int length = strlen(str)+1;
    lengths[rank] = length;
    diverge_mpi_gatherv_bytes( &length, sizeof(int), lengths, counts, displs, 0 );

    int ltot = 0;
    for (int r=0; r<nranks; ++r) {
        ltot += lengths[r];
        counts[r] = lengths[r];
        displs[r] = r==0 ? 0 : counts[r-1]+displs[r-1];
    }
    char* strbuf = calloc( ltot, 1 );
    diverge_mpi_gatherv_bytes( str, lengths[rank], strbuf, counts, displs, 0 );

    if (rank == 0) {
        char** strings = (char**)calloc( nranks+1, sizeof(char*) );
        strings[nranks] = NULL;
        for (int r=0; r<nranks; ++r) {
            strings[r] = malloc( counts[r] );
            memcpy( strings[r], strbuf+displs[r], counts[r] );
        }
        free( counts );
        free( displs );
        free( lengths );
        free( strbuf );
        return strings;
    } else {
        free( counts );
        free( displs );
        free( lengths );
        free( strbuf );
        return NULL;
    }
}

static void mpi_dbg_vsprintf( char* buf, const char* logname_, const char* fname, int line, const char* fmt, va_list ap ) {
#ifdef DIVERGE_LOG_NAMELESS
    const char logname[] = "";
#else
    const char* logname = logname_;
#endif

#ifdef NDEBUG
    (void)fname;
    (void)line;
    sprintf( buf, "%s[" DIVERGE_LOG_PREFIX "%s%s] ", h_log_color( logname_ ),
            DIVERGE_LOG_PREFIX_DIV, logname );
    strcat( buf, h_log_color("RESET") );
#else
    sprintf( buf, "%s[" DIVERGE_LOG_PREFIX "%s%s %s:%i] ", h_log_color( logname_ ),
            DIVERGE_LOG_PREFIX_DIV, logname, fname, line );
    strcat( buf, h_log_color("RESET") );
#endif
    vsprintf( buf+strlen(buf), fmt, ap );
}

void mpi_dbg_printf_all( const char* logname, mpi_loglevel_t loglevel, const char* fname, int line, const char* fmt, ... ) {
    va_list ap;
    va_start( ap, fmt );
    char buf[2048];
    mpi_dbg_vsprintf( buf, logname, fname, line, fmt, ap );
    va_end( ap );

    if (h_diverge_mpi_loglevel < loglevel) return;

    if (!diverge_has_init_internal_libs()) {
        fprintf(h_mpi_log_stderr(), "%s", buf);
        return;
    }

    char** strings = mpi_string_gather( buf );
    if (diverge_mpi_comm_rank() == 0) {
        for (char** pstr = strings; *pstr; ++pstr) {
            fprintf(h_mpi_log_stderr(), "%s", *pstr);
            free(*pstr);
        }
        fflush(h_mpi_log_stderr());
        free(strings);
    }
}

void mpi_py_print_all( const char* str ) {
    char** strl = NULL;
    if (!diverge_has_init_internal_libs()) {
        strl = calloc(2, sizeof*strl);
        *strl = strdup_div(str);
    } else {
        strl = mpi_string_gather( str );
        if (diverge_mpi_comm_rank() != 0) return;
    }
    for (char** pstrl = strl; *pstrl; ++pstrl) {
        fprintf( h_mpi_log_stdout(), "%s[%s%s%s] ", h_log_color( "user" ),
                 DIVERGE_LOG_PREFIX, DIVERGE_LOG_PREFIX_DIV, py_lg_name );
        fprintf( h_mpi_log_stdout(), "%s", h_log_color("RESET") );
        fprintf( h_mpi_log_stdout(), "%s\n", *pstrl );
        free( *pstrl );
    }
    free(strl);
    fflush(h_mpi_log_stdout());
}

void mpi_py_eprint_all( const char* str ) {
    char** strl = NULL;
    if (!diverge_has_init_internal_libs()) {
        strl = calloc(2, sizeof*strl);
        *strl = strdup_div(str);
    } else {
        strl = mpi_string_gather( str );
        if (diverge_mpi_comm_rank() != 0) return;
    }
    for (char** pstrl = strl; *pstrl; ++pstrl) {
        fprintf( h_mpi_log_stderr(), "%s[%s%s%s] ", h_log_color( "user" ),
             DIVERGE_LOG_PREFIX, DIVERGE_LOG_PREFIX_DIV, py_lg_name );
        fprintf( h_mpi_log_stderr(), "%s", h_log_color("RESET") );
        fprintf( h_mpi_log_stderr(), "%s\n", *pstrl );
        free( *pstrl );
    }
    free(strl);
    fflush(h_mpi_log_stderr());
}

