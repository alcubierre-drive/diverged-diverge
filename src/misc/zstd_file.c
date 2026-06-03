/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "zstd_file.h"
#include <stdio.h>

#ifndef USE_COMPRESSED_OUTPUT

zstd_file_t* zstd_fopen(const char *path, const char *mode, int compressionLevel) {
    (void)compressionLevel;
    return (zstd_file_t*)fopen(path, mode);
}
size_t zstd_fwrite(const void *ptr, size_t size, size_t nmemb, zstd_file_t *zf) {
    return fwrite(ptr, size, nmemb, (FILE*)zf);
}
int zstd_fflush(zstd_file_t *zf) {
    return fflush((FILE*)zf);
}
int zstd_fclose(zstd_file_t *zf) {
    return fclose((FILE*)zf);
}

#else

#include <stdlib.h>
#include <zstd.h>

struct zstd_file_t {
    FILE *fp;
    ZSTD_CCtx *cctx;
    void *outBuf;
    size_t outBufSize;
};

zstd_file_t* zstd_fopen(const char *path, const char *mode, int compressionLevel) {
    zstd_file_t *zf = calloc(1, sizeof(*zf));
    if (!zf) goto error;

    zf->fp = fopen(path, mode);
    if (!zf->fp) goto error;

    zf->cctx = ZSTD_createCCtx();
    if (!zf->cctx) goto error;

    size_t rc = ZSTD_CCtx_setParameter(zf->cctx, ZSTD_c_compressionLevel, compressionLevel);
    if (ZSTD_isError(rc)) goto error;

    zf->outBufSize = ZSTD_CStreamOutSize();
    zf->outBuf = malloc(zf->outBufSize);
    if (!zf->outBuf) goto error;

    return zf;
error:
    if (zf) {
        if (zf->cctx) ZSTD_freeCCtx(zf->cctx);
        if (zf->fp) fclose(zf->fp);
        free(zf->outBuf);
        free(zf);
    }
    return NULL;
}

size_t zstd_fwrite(const void *ptr, size_t size, size_t nmemb, zstd_file_t *zf) {
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

int zstd_fflush(zstd_file_t *zf) {
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

int zstd_fclose(zstd_file_t *zf) {
    int result = 0;
    if (!zf) return EOF;

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
    free(zf);

    return result;
}

#endif
