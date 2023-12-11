/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "version.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static char h_git_version[512];
static char h_tag_version[512];

#ifndef GIT_VERSION
#define NO_GIT_VERSION
#else
#undef NO_GIT_VERSION
#endif

#ifndef GIT_VERSION_BRANCH
#define GIT_VERSION_BRANCH "git"
#endif

#if defined(TAG_VERSION) & defined(NO_GIT_VERSION)
#define VERSION TAG_VERSION
#endif

#if defined(TAG_VERSION) & !defined(NO_GIT_VERSION)
#define VERSION TAG_VERSION "," GIT_VERSION_BRANCH ":" GIT_VERSION
#endif

#if !defined(TAG_VERSION) & defined(NO_GIT_VERSION)
#define VERSION GIT_VERSION_BRANCH ":undef"
#endif

#if !defined(TAG_VERSION) & !defined(NO_GIT_VERSION)
#define VERSION GIT_VERSION_BRANCH ":" GIT_VERSION
#endif

const char* get_version( void ) {
    strncpy( h_git_version, VERSION, 511 );
    h_git_version[511] = '\0';
    return h_git_version;
}

const char* tag_version( void ) {

    #ifdef TAG_VERSION
    strncpy( h_tag_version, TAG_VERSION, 511 );
    #else
    strcpy( h_tag_version, "dev" );
    #endif

    h_tag_version[511] = '\0';
    return h_tag_version;
}

int is_version_dirty( void ) {
    #ifndef NO_GIT_VERSION
    char* sv = calloc(1, strlen(GIT_VERSION)+1);
    int nread = sscanf( GIT_VERSION, "%[0-9,a-f]", sv );
    int versions_equal = nread == 0 ? 0 : !strcmp( sv, GIT_VERSION );
    free( sv );
    return !versions_equal;
    #else

    #ifdef TAG_VERSION
    return 0;
    #else
    return 1;
    #endif

    #endif
}
