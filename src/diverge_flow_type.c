/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_flow_type.h"
#include "diverge_internals_struct.h"

diverge_flow_type diverge_flow_type_fromstr( const char* mode ) {
    diverge_flow_type t = diverge_flow_type_invalid;
    // DEVELOPERS
    //
    // For a new backend, you must add a selection to your new flow_type in the
    // below if statements.
    char* method = strdup_div(mode);
    strlwr_div(method); for (char* m=method; *m; ++m) if (*m == '-') *m = '_';

    if (!strcmp(method, "grid"))            t = diverge_grid;
    else if (!strcmp(method, "patch"))      t = diverge_patch;
    else if (!strcmp(method, "tu"))         t = diverge_tu;
    else if (!strcmp(method, "tu_phonons")) t = diverge_tu_phonons;
    free(method);
    return t;
}
