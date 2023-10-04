/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

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
