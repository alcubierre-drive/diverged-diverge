#pragma once

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#else
#include <stddef.h>
#endif

typedef struct diverge_zstd_file_t diverge_zstd_file_t;

diverge_zstd_file_t* diverge_zstd_fopen_default(const char* path, const char* mode);
diverge_zstd_file_t* diverge_zstd_fopen(const char* path, const char* mode, int level);
size_t diverge_zstd_fwrite(const void* ptr, size_t size, size_t nmemb, diverge_zstd_file_t* zf);
int diverge_zstd_fprintf(diverge_zstd_file_t* zf, const char* fmt, ...);
int diverge_zstd_fflush(diverge_zstd_file_t* zf);
int diverge_zstd_fclose(diverge_zstd_file_t* zf);

/** return previous compression level; set new value iff level >= 0. */
int diverge_zstd_file_set_default_compression_level(int level);

#ifdef __cplusplus
}
#endif
