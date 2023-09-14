/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "memory_pool.hpp"
#include "cuda_error.h"
#include <algorithm>
#include <cstdlib>
#include <fftw3.h>
#include <cstring>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
static void cudaMalloc( void** ptr, size_t size ) { *ptr = std::malloc( size ); }
static void cudaSetDevice( int dev ) { (void)dev; }
#define cudaMallocManaged cudaMalloc
#define cudaFree std::free
#define cudaMemset memset
#endif

MemoryPool::MemoryPool() {
    _id = 0;
}

MemoryPool::~MemoryPool() {
    vector<memory_t> pool_copy = _pool;
    for (memory_t m: pool_copy)
        free(m.id);
}

int MemoryPool::malloc( size_t size, MemoryType type, int dev ) {
    _lock.lock();

    memory_t mem;
    mem.dev = dev;
    mem.type = type;
    mem.size = size;

    mem.id = _id;
    _id++;
    switch (type) {
        case Memory_CPU:
            mem.ptr = std::malloc(size);
            break;
        case Memory_FFT:
            mem.ptr = fftw_malloc(size);
            break;
        case Memory_GPU_Device:
            CUDA_CHECK( cudaSetDevice( dev ) );
            CUDA_CHECK( cudaMalloc( &mem.ptr, size ) );
            break;
        case Memory_GPU_Unified:
            CUDA_CHECK( cudaSetDevice( dev ) );
            CUDA_CHECK( cudaMallocManaged( &mem.ptr, size ) );
            break;
    }
    _pool.push_back(mem);

    _lock.unlock();
    return mem.id;
}

int MemoryPool::calloc( size_t size, MemoryType type, int dev ) {
    int id = malloc( size, type, dev );
    memory_t mem = _get(id);
    _lock.lock();
    switch (mem.type) {
        case Memory_CPU:
        case Memory_FFT:
        case Memory_GPU_Unified:
            memset(mem.ptr, 0, mem.size);
            break;
        case Memory_GPU_Device:
            CUDA_CHECK( cudaSetDevice(mem.dev) );
            CUDA_CHECK( cudaMemset(mem.ptr, 0, mem.size) );
            break;
    }
    _lock.unlock();
    return id;
}

int MemoryPool::_find( int id ) {
    auto pos = std::find_if( _pool.begin(), _pool.end(),
            [id]( const memory_t& mem )->bool{ return mem.id == id; } );
    if (pos == _pool.end())
        return -1;
    else
        return std::distance( _pool.begin(), pos );
}

memory_t MemoryPool::_get( int id ) {
    _lock.lock();
    memory_t result;
    int idx = _find(id);
    if (idx == -1) {
        result.ptr = nullptr;
        result.id = -1;
        result.size = 0;
    } else {
        result = _pool[idx];
    }
    _lock.unlock();
    return result;
}

void* MemoryPool::ptr( int id ) {
    return _get(id).ptr;
}

size_t MemoryPool::size( int id ) {
    return _get(id).size;
}

MemoryType MemoryPool::type( int id ) {
    return _get(id).type;
}

int MemoryPool::dev( int id ) {
    return _get(id).id;
}

bool MemoryPool::valid( int id ) {
    memory_t mem = _get(id);
    return (mem.id != -1 && mem.ptr != nullptr && mem.size != 0);
}

void MemoryPool::free( int id ) {
    _lock.lock();
    int idx = _find(id);
    if (idx != -1) {
        memory_t mem = _pool[idx];
        _pool.erase( _pool.begin() + idx );
        _free( mem );
    }
    _lock.unlock();
}

void MemoryPool::_free( memory_t mem ) {
    switch (mem.type) {
        case Memory_CPU:
            std::free(mem.ptr);
            break;
        case Memory_FFT:
            fftw_free(mem.ptr);
            break;
        case Memory_GPU_Device:
        case Memory_GPU_Unified:
            CUDA_CHECK( cudaFree( mem.ptr ) );
            break;
    }
}

struct memory_pool_t {
    MemoryPool* P;
};

memory_pool_t* MemoryPool::c_obj( void ) {
    memory_pool_t* result = (memory_pool_t*)std::malloc(sizeof(memory_pool_t));
    result->P = this;
    return result;
}

int memory_pool_malloc( memory_pool_t* p, size_t size, MemoryType type, int dev ) {
    return p->P->malloc( size, type, dev );
}
int memory_pool_calloc( memory_pool_t* p, size_t size, MemoryType type, int dev ) {
    return p->P->calloc( size, type, dev );
}
void* memory_pool_ptr( memory_pool_t* p, int id ) {
    return p->P->ptr( id );
}
size_t memory_pool_size( memory_pool_t* p, int id ) {
    return p->P->size( id );
}
MemoryType memory_pool_type( memory_pool_t* p, int id ) {
    return p->P->type( id );
}
int memory_pool_dev( memory_pool_t* p, int id ) {
    return p->P->dev( id );
}
bool memory_pool_valid( memory_pool_t* p, int id ) {
    return p->P->valid( id );
}
void memory_pool_free( memory_pool_t* p, int id ) {
    p->P->free( id );
}

memory_pool_t* memory_pool_init() {
    memory_pool_t* p = (memory_pool_t*)malloc(sizeof(memory_pool_t));
    p->P = new MemoryPool();
    return p;
}
void memory_pool_delete( memory_pool_t* p ) {
    delete p->P;
    free(p);
}
void* memory_pool_cxx( memory_pool_t* p ) {
    return (void*)(p->P);
}
