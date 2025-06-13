#include "../diverge_common.h"

#ifdef __cplusplus
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" {
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#endif

#ifndef DIVERGE_GETLINE_MAX
#define DIVERGE_GETLINE_MAX 8192
#endif

static inline index_t diverge_getline( char* line, FILE* stream ) {
    if (!line) return -1;
    if (!fgets(line, DIVERGE_GETLINE_MAX-1, stream)) return -1;
    return feof(stream) ? -1 : (index_t)strlen(line);
}

static inline char* diverge_chomp( char* str, index_t len ) {
    if (len && str[len-1] == '\n') str[len-1] = 0;
    char* l = str; while (*l == ' ' || *l == '\t') l++;
    return l;
}

#ifdef __cplusplus
}
#endif
