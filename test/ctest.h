#pragma once

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
#ifndef CT_SKIP_INITIALIZERS

#ifdef __cplusplus
    #define INITIALIZER(f) \
        static void f(void); \
        struct CT_PASTE_HELPER(f,_t_) { CT_PASTE_HELPER(f,_t_)(void) { f(); } }; \
        static CT_PASTE_HELPER(f,_t_) CT_PASTE_HELPER(f,_); \
        static void f(void)
#elif defined(_MSC_VER)
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

    #ifdef __cplusplus
        #define CT_INITIALIZER_PREFIX extern "C"
    #else
        #define CT_INITIALIZER_PREFIX 
    #endif

    #define INITIALIZER(f) \
        CT_INITIALIZER_PREFIX void f( void )

#endif // CT_SKIP_INITIALIZERS

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
typedef struct { int* matches_case_filters; const char* filter; } ct_strsit_iter_ctx_t;
static inline void ct_strsit_iter( const char* name, void* ctx_ ) {
    ct_strsit_iter_ctx_t* ctx = (ct_strsit_iter_ctx_t*)ctx_;
    if (!strcmp(name, ctx->filter)) *ctx->matches_case_filters += 1;
}

/**
 * returns nfilters, checks suite name and test (index i) name and puts result
 * into ptrs matches_case_filters and matches_suite_filters
 */
static inline int ct_matches_filters( char** filters, ct_suite_t* s, int i,
        int* matches_case_filters, int* matches_suite_filters ) {
    int nfilters = 0;
    *matches_case_filters = *matches_suite_filters = 0;
    for (char** f = filters; *f; f++) {
        nfilters++;
        ct_strsit_iter_ctx_t ctx = { matches_case_filters, *f };
        ct_strsit( s->tests[i].name, CT_NAME_DELIM, ct_strsit_iter, (void*)&ctx );
        if (!strcmp(s->name, *f)) *matches_suite_filters += 1;
    }
    return nfilters;
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
        int matches_case_filters = 0;
        int matches_suite_filters = 0;
        int nfilters = ct_matches_filters( filters, s, i, &matches_case_filters, &matches_suite_filters );
        if (matches_suite_filters || matches_case_filters || nfilters==0) {
            if (s->tests[i].hidden) {
                if (matches_case_filters || matches_suite_filters) {
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
            int matches_case_filters = 0;
            int matches_suite_filters = 0;
            int nfilters = ct_matches_filters( filters, ss, i, &matches_case_filters, &matches_suite_filters );
            if (matches_suite_filters || matches_case_filters || nfilters==0) {
                CT_PRINTF( "%s->%ccase%s %s\n", ss->tests[i].hidden ? ct_color_red : ct_color_green,
                        ss->tests[i].hidden ? '.' : ' ', ct_color_reset, ss->tests[i].name );
                nc_run++;
            }
            nc++;
        }
        if (nc != nc_run)
            CT_PRINTF( "%s>suite<%s %s filtered %3u/%3u cases\n", ct_color_yellow,
                    ct_color_reset, ss->name, nc_run, nc );
        nc_total += nc;
        nc_total_run += nc_run;
    }
    if (nc_total != nc_total_run)
        CT_PRINTF( "%s=== filtered %3u/%3u cases in %u suites ===%s\n", ct_color_yellow,
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
                CT_PRINTF( "%s=== ct suite '%s' filtered out ===%s\n", ct_color_yellow,
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
