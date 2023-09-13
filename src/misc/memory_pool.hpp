/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "memory_pool.h"

#include <vector>
#include <mutex>

using std::vector;
using std::mutex;

typedef struct {
    int id;
    void* ptr;
    size_t size;
    MemoryType type;
    int dev;
} memory_t;

class MemoryPool {
    public:
        MemoryPool();
        ~MemoryPool();
        int calloc( size_t size, MemoryType type, int dev=0 );
        int malloc( size_t size, MemoryType type, int dev=0 );

        void* ptr( int id );
        size_t size( int id );
        MemoryType type( int id );
        int dev( int id );

        bool valid( int id );

        void free( int id );

        memory_pool_t* c_obj(void);
    private:
        void _free( memory_t mem );
        memory_t _get( int id );
        int _find( int id );
        vector<memory_t> _pool;
        mutex _lock;
        int _id;
};
