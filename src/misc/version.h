/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

const char* tag_version( void );
const char* get_version( void );
int is_version_dirty( void );

#ifdef __cplusplus
}
#endif
