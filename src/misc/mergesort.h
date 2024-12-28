#pragma once

#ifdef __cplusplus
extern "C" {
#else
#include <stddef.h>
#endif

void mergesort( void* base, size_t nel, size_t width, int (*cmp)(const void*, const void*),
        int nthr, size_t tasksize );
void mergesort_r( void* base, size_t nel, size_t width, int (*cmp)(const void*, const void*, void*),
        void* ctx, int nthr, size_t tasksize );

#ifdef __cplusplus
}
#endif
