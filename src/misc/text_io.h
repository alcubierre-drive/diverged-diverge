#include "../diverge_common.h"

#ifdef __cplusplus
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
extern "C" {
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
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
    char* l = str; while (isspace((unsigned char)*l)) l++;
    return l;
}

static inline void diverge_chomp_inplace(char* str) {
    if (str == NULL) return;
    char *dst = str;
    char *src = str;

    // strip front
    while (*src && isspace((unsigned char)*src)) src++;

    // compress
    int in_space = 0;
    while (*src) {
        if (isspace((unsigned char)*src)) {
            if (!in_space) {
                *dst++ = ' ';
                in_space = 1;
            }
        } else {
            *dst++ = *src;
            in_space = 0;
        }
        src++;
    }

    // strip back
    if (dst > str && *(dst - 1) == ' ') {
        *(dst - 1) = '\0';
    } else {
        *dst = '\0';
    }
}


#ifdef __cplusplus
}
#endif
