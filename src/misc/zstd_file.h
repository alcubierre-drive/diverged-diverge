#pragma once

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#else
#include <stddef.h>
#endif

typedef struct zstd_file_t zstd_file_t;
zstd_file_t* zstd_fopen(const char *path, const char *mode, int compressionLevel);
size_t zstd_fwrite(const void *ptr, size_t size, size_t nmemb, zstd_file_t *zf);
int zstd_fflush(zstd_file_t *zf);
int zstd_fclose(zstd_file_t *zf);

#ifndef DIVERGE_OUTPUT_COMPRESSION_LEVEL
#define DIVERGE_OUTPUT_COMPRESSION_LEVEL 3
#endif//DIVERGE_OUTPUT_COMPRESSION_LEVEL

#ifdef __cplusplus
}
#endif
