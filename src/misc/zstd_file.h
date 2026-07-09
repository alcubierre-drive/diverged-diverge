#pragma once

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#else
#include <stddef.h>
#endif

typedef struct diverge_zstd_file_t diverge_zstd_file_t;

diverge_zstd_file_t* diverge_zstd_fopen_default(const char* path, const char* mode);
diverge_zstd_file_t* diverge_zstd_fopen(const char* path, const char* mode, int compressionLevel);
size_t diverge_zstd_fwrite(const void* ptr, size_t size, size_t nmemb, diverge_zstd_file_t* zf);
int diverge_zstd_fprintf(diverge_zstd_file_t* zf, const char* fmt, ...);
int diverge_zstd_fflush(diverge_zstd_file_t* zf);
int diverge_zstd_fclose(diverge_zstd_file_t* zf);

void diverge_zstd_file_set_default_compression_level(int compressionLevel);
int diverge_zstd_file_get_default_compression_level(void);

#ifdef __cplusplus
}
#endif
