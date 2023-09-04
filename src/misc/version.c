#include "version.h"
#include <string.h>

static char h_git_version[512];

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

int is_version_dirty( void ) {
    #ifndef NO_GIT_VERSION
    return strlen( GIT_VERSION ) != 7;
    #else

    #ifdef TAG_VERSION
    return 0;
    #else
    return 1;
    #endif

    #endif
}
