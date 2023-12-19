/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void shared_malloc_init( void );
void shared_malloc_finalize( void );

void* shared_malloc( long long sz );
void* shared_calloc( long long cnt, long long sz );
void shared_free( void* ptr );

int shared_exclusive_enter( void* ptr );
void shared_exclusive_wait( void* ptr );

#ifdef __cplusplus
}
#endif

