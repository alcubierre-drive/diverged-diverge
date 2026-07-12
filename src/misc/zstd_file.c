/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "zstd_file.h"
#include "../diverge_common.h"
#include <stdio.h>
#include <stdarg.h>

static int DEFAULT_COMPRESSION_LEVEL = 3;
int diverge_zstd_file_set_default_compression_level(int level) {
    if (level >= 0) return XORSWAP(level, DEFAULT_COMPRESSION_LEVEL);
    return DEFAULT_COMPRESSION_LEVEL;
}
diverge_zstd_file_t* diverge_zstd_fopen_default(const char* path, const char* mode) {
    return diverge_zstd_fopen(path, mode, DEFAULT_COMPRESSION_LEVEL);
}

#ifndef USE_COMPRESSED_OUTPUT

diverge_zstd_file_t* diverge_zstd_fopen(const char* path, const char* mode, int level) {
    (void)level;
    return (diverge_zstd_file_t*)fopen(path, mode);
}
size_t diverge_zstd_fwrite(const void* ptr, size_t size, size_t nmemb, diverge_zstd_file_t* zf) {
    return fwrite(ptr, size, nmemb, (FILE*)zf);
}
int diverge_zstd_fprintf(diverge_zstd_file_t* zf, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int result = vfprintf((FILE*)zf, fmt, ap);
    va_end(ap);
    return result;
}
int diverge_zstd_fflush(diverge_zstd_file_t* zf) {
    return fflush((FILE*)zf);
}
int diverge_zstd_fclose(diverge_zstd_file_t* zf) {
    return fclose((FILE*)zf);
}

#else

#include <stdlib.h>
#include <zstd.h>

struct diverge_zstd_file_t {
    FILE* fp;
    ZSTD_CCtx* cctx;
    void* outBuf;
    size_t outBufSize;
    int isNormalFile;

    FILE* tmpfile;
    void* tmpfile_buf;
};

diverge_zstd_file_t* diverge_zstd_fopen(const char* path, const char* mode, int level) {
    diverge_zstd_file_t* zf = calloc(1, sizeof*zf);
    if (level == 0) {
        zf->fp = fopen(path, mode);
        zf->isNormalFile = 1;
        return zf;
    }
    if (!zf) goto error;

    zf->fp = fopen(path, mode);
    if (!zf->fp) goto error;

    zf->cctx = ZSTD_createCCtx();
    if (!zf->cctx) goto error;

    size_t rc = ZSTD_CCtx_setParameter(zf->cctx, ZSTD_c_compressionLevel, level);
    if (ZSTD_isError(rc)) goto error;

    zf->outBufSize = ZSTD_CStreamOutSize();
    zf->outBuf = malloc(zf->outBufSize);
    if (!zf->outBuf) goto error;

    zf->tmpfile = tmpfile();
    if (!zf->tmpfile) goto error;

    return zf;
error:
    if (zf) {
        if (zf->cctx) ZSTD_freeCCtx(zf->cctx);
        if (zf->fp) fclose(zf->fp);
        free(zf->outBuf);
        if (zf->tmpfile) fclose(zf->tmpfile);
        free(zf);
    }
    return NULL;
}

size_t diverge_zstd_fwrite(const void* ptr, size_t size, size_t nmemb, diverge_zstd_file_t* zf) {
    if (zf->isNormalFile) return fwrite(ptr, size, nmemb, zf->fp);
    size_t totalBytes = size * nmemb;

    ZSTD_inBuffer input = {
        .src = ptr,
        .size = totalBytes,
        .pos = 0
    };

    while (input.pos < input.size) {
        ZSTD_outBuffer output = {
            .dst = zf->outBuf,
            .size = zf->outBufSize,
            .pos = 0
        };

        size_t rc = ZSTD_compressStream2(zf->cctx, &output, &input, ZSTD_e_continue);
        if (ZSTD_isError(rc)) return 0;

        if (output.pos) {
            size_t written = fwrite(zf->outBuf, 1, output.pos, zf->fp);
            if (written != output.pos) return 0;
        }
    }
    return nmemb;
}

int diverge_zstd_fprintf(diverge_zstd_file_t* zf, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    size_t nbytes = vfprintf(zf->tmpfile, fmt, ap);
    va_end(ap);

    rewind(zf->tmpfile);
    zf->tmpfile_buf = realloc(zf->tmpfile_buf, nbytes);
    fread(zf->tmpfile_buf, 1, nbytes, zf->tmpfile);
    rewind(zf->tmpfile);
    return diverge_zstd_fwrite(zf->tmpfile_buf, 1, nbytes, zf);
}

int diverge_zstd_fflush(diverge_zstd_file_t* zf) {
    if (zf->isNormalFile) return fflush(zf->fp);
    size_t remaining = -1;
    do {
        ZSTD_outBuffer output = {
            .dst = zf->outBuf,
            .size = zf->outBufSize,
            .pos = 0
        };

        remaining = ZSTD_compressStream2(zf->cctx, &output, &(ZSTD_inBuffer){NULL, 0, 0}, ZSTD_e_flush);

        if (ZSTD_isError(remaining)) return -1;
        if (output.pos && fwrite(zf->outBuf, 1, output.pos, zf->fp) != output.pos) return -1;
    } while (remaining != 0);
    return fflush(zf->fp);
}

int diverge_zstd_fclose(diverge_zstd_file_t* zf) {
    int result = 0;
    if (!zf) return EOF;
    if (zf->isNormalFile) {
        result = fclose(zf->fp);
        free(zf);
        return result;
    }

    size_t remaining = -1;
    do {
        ZSTD_outBuffer output = {
            .dst = zf->outBuf,
            .size = zf->outBufSize,
            .pos = 0
        };

        remaining = ZSTD_compressStream2(zf->cctx, &output, &(ZSTD_inBuffer){NULL, 0, 0}, ZSTD_e_end);

        if (ZSTD_isError(remaining)) {
            result = EOF;
            break;
        }

        if (output.pos && fwrite(zf->outBuf, 1, output.pos, zf->fp) != output.pos) {
            result = EOF;
            break;
        }
    } while (remaining != 0);

    if (fflush(zf->fp)) result = EOF;
    if (fclose(zf->fp)) result = EOF;

    ZSTD_freeCCtx(zf->cctx);
    free(zf->outBuf);
    fclose(zf->tmpfile);
    free(zf->tmpfile_buf);
    free(zf);

    return result;
}

#endif
