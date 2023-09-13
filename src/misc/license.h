/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Function: diverge_license
// returns license information as string
const char* diverge_license( void );
// Function: diverge_license_print
// prints out license information to stdout
void diverge_license_print( void );

#ifdef __cplusplus
}
#endif
