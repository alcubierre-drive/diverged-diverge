/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#else
#include <stddef.h>
#include <stdbool.h>
#endif

typedef enum {
    Memory_CPU,
    Memory_GPU_Device,
    Memory_GPU_Unified,
    Memory_FFT,
} MemoryType;
typedef struct memory_pool_t memory_pool_t;

int memory_pool_malloc( memory_pool_t* p, size_t size, MemoryType type, int dev );
int memory_pool_calloc( memory_pool_t* p, size_t size, MemoryType type, int dev );
void* memory_pool_ptr( memory_pool_t* p, int id );
size_t memory_pool_size( memory_pool_t* p, int id );
MemoryType memory_pool_type( memory_pool_t* p, int id );
int memory_pool_dev( memory_pool_t* p, int id );
bool memory_pool_valid( memory_pool_t* p, int id );
void memory_pool_free( memory_pool_t* p, int id );

memory_pool_t* memory_pool_init();
void memory_pool_delete( memory_pool_t* p );
void* memory_pool_cxx( memory_pool_t* p );

#ifdef __cplusplus
}
#endif
