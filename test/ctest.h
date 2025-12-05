/** ctest.h developed by Lennart Klebl (c) 2025, GPLv3 */

#pragma once

// XXX this is a divERGe "quirk". Maybe also for CUDA; dunno.
#ifdef USE_MPI
#define CT_NO_FORK
#endif
// XXX this also is a divERGe "quirk", more of a feature, though.
#include "../src/misc/mpi_log.h"
#define CT_PRINTF mpi_printf

#ifndef CT_NO_FORK
#include <sys/wait.h>
#include <unistd.h>
#include <stdlib.h>
#endif

#ifdef __cplusplus
#define CT_EXTERN extern "C"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#else
#define CT_EXTERN extern
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#endif

#ifndef CT_PRINTF
#define CT_PRINTF printf
#endif

typedef struct ct_test_t ct_test_t;
struct ct_test_t {
    void (*run)( void );
    const char* name;
    int hidden;
};

#ifndef CT_SUITE_MAX_FUNCS
#define CT_SUITE_MAX_FUNCS 128
#endif//CT_SUITE_MAX_FUNCS
#ifndef CT_SUITE_NAME_NCHAR
#define CT_SUITE_NAME_NCHAR 256
#endif//CT_SUITE_NAME_NCHAR

typedef struct ct_suite_t ct_suite_t;
struct ct_suite_t {
    ct_test_t tests[CT_SUITE_MAX_FUNCS];
    unsigned ntests;
    unsigned nasserts;
    unsigned nproblems;
    char name[CT_SUITE_NAME_NCHAR];
};

// this file's suite
#ifdef __cplusplus
static ct_suite_t ct_suite {};
#else
static ct_suite_t ct_suite = {0};
#endif

// all suites
#ifndef CT_MAX_SUITES
#define CT_MAX_SUITES 128
#endif//CT_MAX_SUITES
#ifndef CT_MAIN
CT_EXTERN ct_suite_t* ct_suites[CT_MAX_SUITES];
CT_EXTERN unsigned ct_nsuites;
#else
#ifdef __cplusplus
extern "C" {
#endif
ct_suite_t* ct_suites[CT_MAX_SUITES] = {0};
unsigned ct_nsuites = 0;
#ifdef __cplusplus
}
#endif
#endif

// stupid initializer code
#ifdef __cplusplus

    #define INITIALIZER(f) \
        static void f(void); \
        struct CT_PASTE_HELPER(f,_t_) { CT_PASTE_HELPER(f,_t_)(void) { f(); } }; \
        static CT_PASTE_HELPER(f,_t_) CT_PASTE_HELPER(f,_); \
        static void f(void)

#else // __cplusplus

    #ifndef CT_SKIP_INITIALIZERS

        #if defined(_MSC_VER)
            #pragma section(".CRT$XCU",read)
            #define INITIALIZER2_(f,p) \
                static void f(void); \
                __declspec(allocate(".CRT$XCU")) void (*CT_PASTE_HELPER(f,_))(void) = f; \
                __pragma(comment(linker,"/include:" p #f "_")) \
                static void f(void)
            #ifdef _WIN64
                #define INITIALIZER(f) INITIALIZER2_(f,"")
            #else
                #define INITIALIZER(f) INITIALIZER2_(f,"_")
            #endif
        #else
            #define INITIALIZER(f) \
                static void f(void) __attribute__((constructor)); \
                static void f(void)
        #endif

    #else // CT_SKIP_INITIALIZERS

        #define INITIALIZER(f) void f( void )

    #endif // CT_SKIP_INITIALIZERS

#endif // __cplusplus

#ifndef CT_SUITE_NAME
#define CT_SUITE_NAME _anon_
#endif

#define CT_PASTE_HELPER(X, Y) X##Y
#define CT_REGISTER_NAME(X) ct__register__suite__ ## X
#define INITIALIZER_PASTE(X) INITIALIZER(CT_REGISTER_NAME(X))
#define CT_STRINGIFY(X) #X
#define CT_EXPAND_AND_STRINGIFY(X) CT_STRINGIFY(X)

// append this file's suite to the list of all suites
INITIALIZER_PASTE(CT_SUITE_NAME) {
    if (ct_nsuites < CT_MAX_SUITES) {
        ct_suites[ct_nsuites++] = &ct_suite;
        strncpy( ct_suite.name, CT_EXPAND_AND_STRINGIFY(CT_SUITE_NAME), CT_SUITE_NAME_NCHAR-1 );
    } else {
        CT_PRINTF( "too many test suites (%u), redefine CT_MAX_SUITES\n", ct_nsuites+1 );
    }
}

static inline void ct_test_append( const char* name, void(*run)(void), int hidden ) {
    ct_test_t t = { run, name, hidden };
    if (ct_suite.ntests < CT_SUITE_MAX_FUNCS)
        ct_suite.tests[ct_suite.ntests++] = t;
    else
        CT_PRINTF( "too many tests (%u) in suite %s, redefine CT_SUITE_MAX_FUNCS\n",
                 ct_suite.ntests+1, CT_EXPAND_AND_STRINGIFY(CT_SUITE_NAME) );
}
// this is the standard macro to define test functions
#define CT_TEST( name ) \
    static void ct__test__##name( void ); \
    INITIALIZER( ct__register__test__##name ) { \
        ct_test_append( #name, ct__test__##name, 0 ); \
    } \
    static void ct__test__##name( void )
// and this is another one where the name can be different from the function
// name
#define CT_TEST_NAME( fname, tname ) \
    static void ct__test__##fname ( void ); \
    INITIALIZER( ct__register__test__##fname ) { \
        ct_test_append( tname, ct__test__##fname, 0 ); \
    } \
    static void ct__test__##fname ( void )
// this one is for hidden tests
#define CT_TEST_HIDDEN( name ) \
    static void ct__test__##name( void ); \
    INITIALIZER( ct__register__test__##name ) { \
        ct_test_append( #name, ct__test__##name, 1 ); \
    } \
    static void ct__test__##name( void )
// this one too
#define CT_TEST_NAME_HIDDEN( fname, tname ) \
    static void ct__test__##fname ( void ); \
    INITIALIZER( ct__register__test__##fname ) { \
        ct_test_append( tname, ct__test__##fname, 1 ); \
    } \
    static void ct__test__##fname ( void )

// we "need" colors
static const char ct_color_red[] = "\x1b[31;1m",
                  ct_color_green[] = "\x1b[32;1m",
                  ct_color_yellow[] = "\x1b[33;1m",
                  ct_color_blue[] = "\x1b[34;1m",
                  ct_color_reset[] = "\033[0m";

// and this is the macro to make assertions
#define CT_CHECK( expr ) { \
    ct_suite.nasserts++; \
    if (!(expr)) { \
        ct_suite.nproblems++; \
        CT_PRINTF( "%s!!! %s @%s:%i: CT_CHECK(" #expr ") failed!!!%s\n", ct_color_red, \
                __func__, __FILE__, __LINE__, ct_color_reset ); \
    } \
}

#ifndef CT_NAME_DELIM
#define CT_NAME_DELIM ","
#endif//CT_NAME_DELIM

/** substring iteration */
static inline void ct_strsit( const char* str_, const char* delim,
        void (*iter)(const char* str, void* ctx), void* ctx ) {
    if (!delim || !str_) return;
    char* str = (char*)malloc(strlen(str_)+1); strcpy(str, str_);
    for (char* tok=strtok(str,delim); tok!=NULL; tok=strtok(NULL,delim))
        (*iter)(tok, ctx);
    free(str);
}
/** filter substring iteration */
typedef struct { int nkeywords; int nmatches; const char* filter; } ct_strsit_iter_ctx_t;
static inline void ct_strsit_iter( const char* name, void* ctx_ ) {
    ct_strsit_iter_ctx_t* ctx = (ct_strsit_iter_ctx_t*)ctx_;
    ctx->nkeywords++;
    if (!strcmp(name, ctx->filter)) ctx->nmatches++;
}
typedef struct { int nkeywords, nmatches, is_neg, is_and; } ct_filter_eval_t;
static const ct_filter_eval_t ct_filter_eval_skip = { -1, -1, -1, -1 };
static inline int ct_filter_eval_is_skip( ct_filter_eval_t fe ) {
    return (fe.nkeywords == ct_filter_eval_skip.nkeywords) &&
           (fe.nmatches == ct_filter_eval_skip.nmatches) &&
           (fe.is_neg == ct_filter_eval_skip.is_neg) &&
           (fe.is_and == ct_filter_eval_skip.is_and);
}

/**
 * returns whether suite name or test keywords (of index i) matches with the
 * filters given in the zero terminated array. If -1 is returned, this means we
 * don't have any filters or only negative filters that don't match
 */
static inline int ct_matches_filters( char** filters, ct_suite_t* s, int i ) {

    if (!*filters) return -1; // no filters

    int nfilters = 0;
    for (char** f=filters; *f; f++) nfilters++;

    ct_filter_eval_t* fev = (ct_filter_eval_t*)calloc(nfilters, sizeof*fev);

    nfilters = 0;
    for (char** f=filters; *f; f++) {
        char* filt = *f;
        int len = strlen(filt);
        int is_neg = (len > 1) && (*f[0] == '~' || *f[0] == '!'),
            is_and = (len > 1) && (*f[0] == '&' || *f[0] == '+');
        if (is_neg || is_and) filt++;

        // case keywords
        ct_strsit_iter_ctx_t ctx = { 0, 0, filt };
        ct_strsit( s->tests[i].name, CT_NAME_DELIM, ct_strsit_iter, (void*)&ctx );

        // suite name
        ctx.nkeywords++;
        if (!strcmp(s->name, filt)) ctx.nmatches++;

        ct_filter_eval_t fev_cur = { ctx.nkeywords, ctx.nmatches, is_neg, is_and };
        fev[nfilters++] = fev_cur;
    }

    int result = 0;
    // prevent goto initialization crossing
    int nfilters_and = 0,
        nmatches_and = 0,
        nfilters_neg = 0;

    // check for any negative matches and early return
    for (int f=0; f<nfilters; ++f) {
        if (!ct_filter_eval_is_skip(fev[f]) && fev[f].is_neg) {
            if (fev[f].nmatches > 0) goto ct_matches_filters_return;
            nfilters_neg++;
            fev[f] = ct_filter_eval_skip; // flag skipped
        }
    }

    // early return in case *just* negative filters are applied
    if (nfilters_neg == nfilters) {
        result = -1;
        goto ct_matches_filters_return;
    }

    // now check for the number of AND matches
    for (int f=0; f<nfilters; ++f) {
        if (!ct_filter_eval_is_skip(fev[f]) && fev[f].is_and) {
            if (fev[f].nmatches) nmatches_and++;
            nfilters_and++;
            fev[f] = ct_filter_eval_skip; // flag skipped
        }
    }
    if ((nfilters_and == nmatches_and) && (nfilters_and != 0)) {
        // we have a match for all AND filters (in case there are any) so we can return
        result = 1;
        goto ct_matches_filters_return;
    }

    // last check for the OR filters, if any matches we're good and return
    for (int f=0; f<nfilters; ++f) {
        if (!ct_filter_eval_is_skip(fev[f])) {
            if (fev[f].nmatches) {
                result = 1;
                goto ct_matches_filters_return;
            }
        }
    }

    ct_matches_filters_return: free( fev ); return result;
}

/**
 * runs a test suite and puts result into pointers. applies filters using
 * ct_matches_filters. ignores suites with zero tests.
 */
static inline void ct_suite_run( ct_suite_t* s, unsigned* nassert,
        unsigned* npassed, unsigned* ncases, unsigned* ncases_run,
        char** filters ) {
    if (s->ntests == 0) return;

    CT_PRINTF( "%s=== ct suite '%s' ===%s\n", ct_color_yellow, s->name, ct_color_reset );

    unsigned nc = 0, nc_run = 0;

    for (unsigned i=0; i<s->ntests; ++i) {
        int nmatches = ct_matches_filters( filters, s, i );
        if (nmatches) {
            if (s->tests[i].hidden) {
                if (nmatches > 0) {
                    CT_PRINTF( "... run test '%s'\n", s->tests[i].name );
                    s->tests[i].run();
                    nc++;
                    nc_run++;
                }
            } else {
                CT_PRINTF( "... run test '%s'\n", s->tests[i].name );
                s->tests[i].run();
                nc_run++;
            }
        }
        if (!s->tests[i].hidden) nc++;
    }
    *nassert = s->nasserts;
    *npassed = s->nasserts-s->nproblems;
    s->nasserts = s->nproblems = 0; // reset
    *ncases = nc;
    *ncases_run = nc_run;
}

/**
 * lists all tests of all suites that are active given a list of filters.
 * ignores suites with zero tests.
 */
static inline int ct_list( char** filters ) {
    for (char** f = filters; *f; f++)
        CT_PRINTF( "%s=== filter '%s' ===%s\n", ct_color_yellow, *f, ct_color_reset );
    unsigned nc_total = 0, nc_total_run = 0;
    unsigned ct_nsuites_nz = 0;
    for (unsigned s=0; s<ct_nsuites; ++s) {
        ct_suite_t* ss = ct_suites[s];
        if (ss->ntests == 0) continue;
        ct_nsuites_nz++;

        CT_PRINTF( "%s>suite<%s %s\n", ct_color_yellow, ct_color_reset, ss->name );
        unsigned nc = 0, nc_run = 0;
        for (unsigned i=0; i<ss->ntests; ++i) {
            int nmatches = ct_matches_filters( filters, ss, i );
            if (nmatches) {
                CT_PRINTF( "%s->%ccase%s %s\n", ss->tests[i].hidden ? ct_color_red : ct_color_green,
                        ss->tests[i].hidden ? '.' : ' ', ct_color_reset, ss->tests[i].name );
                nc_run++;
            }
            nc++;
        }
        if (nc != nc_run) {
            const char up_line[16] = "\033[F";
            const char no_up[16] = "";
            CT_PRINTF( "%s%s>suite<%s %s found %u/%u cases\n", nc_run == 0 ? up_line : no_up,
                    ct_color_blue, ct_color_reset, ss->name, nc_run, nc );
        }
        nc_total += nc;
        nc_total_run += nc_run;
    }
    if (nc_total != nc_total_run)
        CT_PRINTF( "%s=== found %u/%u cases in %u suites ===%s\n", ct_color_yellow,
                nc_total_run, nc_total, ct_nsuites_nz, ct_color_reset );
    return 0;
}

/**
 * runs all suites that match the given filters
 */
static inline int ct_run( char** filters ) {
    unsigned npassed_total = 0, nassert_total = 0,
             ncases_total = 0, ncases_run_total = 0,
             ncrashed = 0;
    unsigned successes = 0;
    unsigned ct_nsuites_nz = 0;
    for (unsigned s=0; s<ct_nsuites; ++s) {
        if (ct_suites[s]->ntests == 0) continue;
        ct_nsuites_nz++;
        unsigned nums[4] = {0}; // nassert, npassed, ncases, ncases_run
        int retval = 0;

        #ifndef CT_NO_FORK
        int pipefd[2] = {0};

        if (pipe(pipefd) == -1) {
            CT_PRINTF("cannot setup pipe\n");
            exit(EXIT_FAILURE);
        }

        pid_t p = fork();
        if (p == 0) { // child
            ct_suite_run( ct_suites[s], &nums[0], &nums[1], &nums[2], &nums[3], filters );
            close(pipefd[0]);
            write(pipefd[1], nums, sizeof nums);
            close(pipefd[1]);
            exit(EXIT_SUCCESS);
        } else {
            close(pipefd[1]);
            read(pipefd[0], nums, sizeof nums);
            close(pipefd[0]);
            waitpid( p, &retval, 0 );
        }
        #else // CT_NO_FORK
        ct_suite_run( ct_suites[s], &nums[0], &nums[1], &nums[2], &nums[3], filters );
        #endif // CT_NO_FORK

        unsigned nassert = nums[0], npassed = nums[1],
                 ncases = nums[2], ncases_run = nums[3];

        nassert_total += nassert;
        npassed_total += npassed;
        ncases_total += ncases;
        ncases_run_total += ncases_run;
        unsigned suite_success = (npassed == nassert) && !retval;
        if (retval) {
            ncrashed++;
            CT_PRINTF( "%s=== ct suite '%s' CRASHED (%i) ===%s\n", ct_color_red,
                ct_suites[s]->name, retval, ct_color_reset );
        } else {
            if (ncases_run > 0) {
                CT_PRINTF( "%s=== ct suite '%s' passed %u/%u assertions in %u/%u cases ===%s\n",
                    suite_success ? ct_color_green : ct_color_red, ct_suites[s]->name, npassed,
                    nassert, ncases_run, ncases, ct_color_reset );
            } else {
                CT_PRINTF( "\033[F%s=== ct suite '%s' filtered out ===%s\n", ct_color_yellow,
                    ct_suites[s]->name, ct_color_reset );
            }
        }
        successes += suite_success;
    }

    CT_PRINTF( "%s=== ct report ===%s\n", ct_color_yellow, ct_color_reset );
    CT_PRINTF( "%s=== crashed %u/%u suites with %u/%u failed assertions in %u/%u cases ===%s\n",
        successes == ct_nsuites_nz ? ct_color_green : ct_color_red,
        ncrashed, ct_nsuites_nz, nassert_total - npassed_total, nassert_total,
        ncases_run_total, ncases_total, ct_color_reset );

    return !(successes == ct_nsuites_nz);
}

static inline int ct_list_or_run( int argc, char** argv ) {
    argv[argc] = NULL;
    if (argc <= 1) return ct_run(argv+argc);

    if (!strcmp(argv[1],"-l")) return ct_list(argv+2);
    else                       return ct_run(argv+1);
}
