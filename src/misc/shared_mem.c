/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "shared_mem.h"

#if defined(USE_SHARED_MEM) && defined(USE_MPI)

#ifdef SHARED_MEM_MPI_IMPL

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "hashtable.h"
#include "mpi_functions.h"

static int status = 0;
static char error[MPI_MAX_ERROR_STRING+1] = {0};
static int errorlen = 0;

#define MPICALL( fun, ... ) { \
    if ((status = MPI_##fun( __VA_ARGS__ )) != MPI_SUCCESS) { \
        MPI_Error_string( status, error, &errorlen ); \
        error[errorlen] = '\0'; \
        printf( __FILE__ ":%i MPIERROR: %s\n", __LINE__, error ); \
    } \
}

typedef struct {
    char name[MPI_MAX_PROCESSOR_NAME+1];
    int rank;
} mpi_proc_info;

static int mpi_proc_info_compar( const void* pa, const void* pb ) {
    const mpi_proc_info* a = pa;
    const mpi_proc_info* b = pb;
    int result = strcmp(a->name, b->name);
    if (!result)
        return a->rank > b->rank;
    else
        return result;
}

typedef struct {
    void* ptr;
    size_t sz;

    int rank, size;
    MPI_Win win;
    MPI_Comm comm;
} shmem_t;

typedef struct {
    int rank, size;
    MPI_Comm comm;
    int shrank, shsize;
    MPI_Comm shcomm;

    hashtable_t* pool;
} shmem_pool_t;

static int shmem_pool_more_colors = 0;

static shmem_pool_t* shmem_pool_init( void ) {
    shmem_pool_t* p = calloc(1, sizeof(shmem_pool_t));
    p->comm = diverge_mpi_get_comm();
    p->rank = diverge_mpi_comm_rank();
    p->size = diverge_mpi_comm_size();

    MPICALL(Barrier, p->comm);
    // color setup
    mpi_proc_info* procnams = calloc(p->size, sizeof(mpi_proc_info));
    int len = -1;
    MPICALL(Get_processor_name, procnams[p->rank].name, &len);
    procnams[p->rank].rank = p->rank;
    MPICALL(Allgather, MPI_IN_PLACE, 0, 0, procnams, sizeof(mpi_proc_info), MPI_BYTE, MPI_COMM_WORLD);
    qsort( procnams, p->size, sizeof(mpi_proc_info), mpi_proc_info_compar );
    int color = 0;
    for (int r=1; r<=p->rank; ++r)
        if (strcmp( procnams[r-1].name, procnams[r].name )) color++;
    free( procnams );

    if (shmem_pool_more_colors) {
        int* colors = calloc(p->size, sizeof(int));
        diverge_mpi_allgather_int( &color, colors, 1 );

        int ncolors = colors[p->size-1] + 1;
        int *color_idx = calloc(p->size, sizeof(int));
        int *color_cnt = calloc(ncolors, sizeof(int));

        for (int c=0; c<ncolors; ++c)
        for (int r=0; r<p->size; ++r) {
            color_idx[r] = color_cnt[c];
            color_cnt[c] += colors[r] == c;
        }

        for (int r=0; r<p->size; ++r) {
            int c = colors[r],
                i = color_idx[r];
            colors[r] = i%2 ? c+ncolors : c;
        }

        free( color_idx );
        free( color_cnt );

        color = colors[p->rank];
        free( colors );
    }

    // split the communicator according to color
    MPICALL(Comm_split, p->comm, color, 0, &p->shcomm);
    MPICALL(Comm_rank, p->shcomm, &p->shrank);
    MPICALL(Comm_size, p->shcomm, &p->shsize);
    MPICALL(Barrier, p->comm);
    return p;
}

static void shmem_pool_destroy( shmem_pool_t* p ) {
    free( p );
}

static shmem_t* shmem_alloc( shmem_pool_t* p, size_t sz ) {
    shmem_t* m = calloc(1, sizeof(shmem_t));

    m->sz = sz;
    m->rank = p->shrank;
    m->size = p->shsize;
    m->comm = p->shcomm;

    if (m->rank == 0) {
        MPICALL(Win_allocate_shared, sz, 1, MPI_INFO_NULL, m->comm, &m->ptr, &m->win );
    } else {
        int disp_unit = 0;
        MPI_Aint ssize = 0;
        MPICALL(Win_allocate_shared, 0, 1, MPI_INFO_NULL, m->comm, &m->ptr, &m->win );
        MPICALL(Win_shared_query, m->win, 0, &ssize, &disp_unit, &m->ptr );
    }
    MPICALL(Barrier, m->comm);

    return m;
}

static void shmem_free( shmem_t* m ) {
    MPICALL(Win_free, &m->win );

    free( m );
}

static shmem_pool_t* shared_mem_pool = NULL;

void shared_malloc_init( int more_colors ) {
    shmem_pool_more_colors = more_colors;
    shared_mem_pool = shmem_pool_init();
    shared_mem_pool->pool = hashtable_init( 128, 8 );
}

static inline void hashtable_iter_free_shmem( index_t key, void* val, void* ctx ) {
    (void)ctx; (void)key; shmem_free( val );
}
void shared_malloc_finalize( void ) {
    if (!shared_mem_pool) return;

    hashtable_iterate( shared_mem_pool->pool, &hashtable_iter_free_shmem, NULL );
    hashtable_free( shared_mem_pool->pool );

    shmem_pool_destroy( shared_mem_pool );
}

void* shared_calloc( long long cnt, long long sz ) {
    void* ptr = shared_malloc( cnt*sz );
    if (shared_exclusive_enter(ptr) == 1) memset( ptr, 0, cnt*sz );
    shared_exclusive_wait(ptr);
    return ptr;
}

void* shared_malloc( long long sz ) {
    if (!shared_mem_pool) return NULL;
    shmem_t* memptr = shmem_alloc( shared_mem_pool, sz );
    hashtable_insert( shared_mem_pool->pool, (index_t)memptr->ptr, memptr );
    return memptr->ptr;
}

void shared_free( void* ptr ) {
    if (!shared_mem_pool) return;
    if (!ptr) return;
    shmem_t* memptr = hashtable_pop( shared_mem_pool->pool, (index_t)ptr );
    if (!memptr) return;
    shmem_free( memptr );
}

int shared_exclusive_enter( void* ptr ) {
    if (!shared_mem_pool) return -1;
    if (!ptr) return -1;
    shmem_t* memptr = hashtable_find( shared_mem_pool->pool, (index_t)ptr );
    if (!memptr) return -1;

    MPICALL(Win_lock, MPI_LOCK_SHARED, 0, MPI_MODE_NOCHECK, memptr->win);
    if (memptr->rank == 0)
        return 1;
    return 0;
}

void shared_exclusive_wait( void* ptr ) {
    if (!shared_mem_pool) return;
    if (!ptr) return;
    shmem_t* memptr = hashtable_find( shared_mem_pool->pool, (index_t)ptr );
    if (!memptr) return;

    MPICALL(Win_unlock, 0, memptr->win);
    shared_malloc_barrier();
}

int shared_malloc_rank( void ) {
    if (!shared_mem_pool) return -1;
    return shared_mem_pool->shrank;
}

int shared_malloc_size( void ) {
    if (!shared_mem_pool) return -1;
    return shared_mem_pool->shsize;
}

static inline void shared_mem_pool_win_lock( index_t key, void* val, void* ctx ) {
    (void)key; (void)ctx;
    MPICALL(Win_lock, MPI_LOCK_SHARED, 0, MPI_MODE_NOCHECK, ((shmem_t*)val)->win);
}
static inline void shared_mem_pool_win_unlock( index_t key, void* val, void* ctx ) {
    (void)key; (void)ctx;
    MPICALL(Win_unlock, 0, ((shmem_t*)val)->win);
}
void shared_malloc_barrier( void ) {
    if (!shared_mem_pool) return;
    MPICALL(Barrier, shared_mem_pool->shcomm);
    hashtable_iterate(shared_mem_pool->pool, &shared_mem_pool_win_lock, NULL);
    hashtable_iterate(shared_mem_pool->pool, &shared_mem_pool_win_unlock, NULL);
    MPICALL(Barrier, shared_mem_pool->shcomm);
}

#else // SHARED_MEM_MPI_IMPL

#define _GNU_SOURCE
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>
#include <time.h>

#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "hashtable.h"
#include "mpi_functions.h"
#include "md5.h"

#ifndef SHARED_MEM_POSIX_IMPL

#include <linux/memfd.h>
#ifdef MEMFD_NOT_SUPPORTED
#include <sys/syscall.h>
#include <err.h>
static int memfd_create_(const char* fname, unsigned flags) {
    int fd;
    if ( (fd = syscall(SYS_memfd_create, fname, flags)) == -1 )
        err(1, "memfd_create");
    return fd;
}
#define memfd_create memfd_create_
#endif // MEMFD_NOT_SUPPORTED

#else // SHARED_MEM_POSIX_IMPL

#include <errno.h>

static inline void shmem_alloc_name( char* out, size_t alloc_sz, int shrank, int shsize, int rank, int size ) {
    // buffer for all that goes into the md5sum
    char buf[2048] = {0};
    snprintf( buf, sizeof(buf), "%zu@%li:%i/%i:%i/%i", alloc_sz, clock(), shrank, shsize, rank, size );

    // calculate the md5sum
    uint128_t md5 = md5sum_div_stack( buf, sizeof(buf) );

    // output
    sprintf( out, "/diverge.shm." );
    for (int i=0; i<16; ++i) {
        char mbuf[16];
        sprintf( mbuf, "%02X", (int)md5.digest[i] );
        strcat( out, mbuf );
    }
}

#endif // SHARED_MEM_POSIX_IMPL

static int status = 0;
static char error[MPI_MAX_ERROR_STRING+1] = {0};
static int errorlen = 0;

#define MPICALL( fun, ... ) { \
    if ((status = MPI_##fun( __VA_ARGS__ )) != MPI_SUCCESS) { \
        MPI_Error_string( status, error, &errorlen ); \
        error[errorlen] = '\0'; \
        printf( __FILE__ ":%i MPIERROR: %s\n", __LINE__, error ); \
    } \
}

typedef struct {
    char name[MPI_MAX_PROCESSOR_NAME+1];
    int rank;
} mpi_proc_info;

static int mpi_proc_info_compar( const void* pa, const void* pb ) {
    const mpi_proc_info* a = pa;
    const mpi_proc_info* b = pb;
    int result = strcmp(a->name, b->name);
    if (!result)
        return a->rank > b->rank;
    else
        return result;
}

typedef struct {
    void* ptr;
    sem_t* sem;
    size_t sz;
    int fd;

    int rank, size;
    MPI_Comm comm;

#ifdef SHARED_MEM_POSIX_IMPL
    char name[256];
#endif
} shmem_t;

typedef struct {
    int rank, size;
    MPI_Comm comm;
    int shrank, shsize;
    MPI_Comm shcomm;

    hashtable_t* pool;
} shmem_pool_t;

static int shmem_pool_more_colors = 0;

static shmem_pool_t* shmem_pool_init( void ) {
    shmem_pool_t* p = calloc(1, sizeof(shmem_pool_t));
    p->comm = diverge_mpi_get_comm();
    p->rank = diverge_mpi_comm_rank();
    p->size = diverge_mpi_comm_size();

    MPICALL(Barrier, p->comm);
    // color setup
    mpi_proc_info* procnams = calloc(p->size, sizeof(mpi_proc_info));
    int len = -1;
    MPICALL(Get_processor_name, procnams[p->rank].name, &len);
    procnams[p->rank].rank = p->rank;
    MPICALL(Allgather, MPI_IN_PLACE, 0, 0, procnams, sizeof(mpi_proc_info), MPI_BYTE, MPI_COMM_WORLD);
    qsort( procnams, p->size, sizeof(mpi_proc_info), mpi_proc_info_compar );
    int color = 0;
    for (int r=1; r<=p->rank; ++r)
        if (strcmp( procnams[r-1].name, procnams[r].name )) color++;
    free( procnams );

    if (shmem_pool_more_colors) {
        int* colors = calloc(p->size, sizeof(int));
        diverge_mpi_allgather_int( &color, colors, 1 );

        int ncolors = colors[p->size-1] + 1;
        int *color_idx = calloc(p->size, sizeof(int));
        int *color_cnt = calloc(ncolors, sizeof(int));

        for (int c=0; c<ncolors; ++c)
        for (int r=0; r<p->size; ++r) {
            color_idx[r] = color_cnt[c];
            color_cnt[c] += colors[r] == c;
        }

        for (int r=0; r<p->size; ++r) {
            int c = colors[r],
                i = color_idx[r];
            colors[r] = i%2 ? c+ncolors : c;
        }

        free( color_idx );
        free( color_cnt );

        color = colors[p->rank];
        free( colors );
    }

    // split the communicator according to color
    MPICALL(Comm_split, p->comm, color, 0, &p->shcomm);
    MPICALL(Comm_rank, p->shcomm, &p->shrank);
    MPICALL(Comm_size, p->shcomm, &p->shsize);
    MPICALL(Barrier, p->comm);
    return p;
}

static void shmem_pool_destroy( shmem_pool_t* p ) {
    free( p );
}

static shmem_t* shmem_alloc( shmem_pool_t* p, size_t sz ) {
    shmem_t* m = calloc(1, sizeof(shmem_t));

    m->sz = sz + sizeof(sem_t);
    m->rank = p->shrank;
    m->size = p->shsize;
    m->comm = p->shcomm;

    int errors = 0;

#ifndef SHARED_MEM_POSIX_IMPL

    char fd_path[64];
    if (p->shrank == 0) {
        // tame the beast (i.e. create the memfd) if we're shared rank 0
        errors += ((m->fd = memfd_create( "666", MFD_CLOEXEC|MFD_ALLOW_SEALING )) == -1);
        snprintf( fd_path, sizeof(fd_path), "/proc/%d/fd/%d", getpid(), m->fd );
        errors += ((ftruncate( m->fd, m->sz ) == -1));
    }
    MPICALL(Bcast, fd_path, sizeof(fd_path), MPI_BYTE, 0, p->shcomm);

    MPICALL(Reduce, p->rank == 0 ? MPI_IN_PLACE : &errors, &errors, 1, MPI_INT, MPI_SUM, 0, p->comm);
    if (errors)
        mpi_err_printf( "could not open/truncate shm file %s\n", fd_path );

    errors = 0;
    if (p->shrank != 0)
        errors += ((m->fd = open(fd_path, O_RDWR)) == -1);

    errors += ((m->ptr = mmap( NULL, m->sz, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_SHARED, m->fd, 0 )) == (void*)-1);

    MPICALL(Reduce, p->rank == 0 ? MPI_IN_PLACE : &errors, &errors, 1, MPI_INT, MPI_SUM, 0, p->comm);
    if (errors)
        mpi_err_printf( "could not open, or mmap\n" );

#else // SHARED_MEM_POSIX_IMPL

    m->fd = -1;
    if (p->shrank == 0) {
        shmem_alloc_name( m->name, m->sz, p->shrank, p->shsize, p->rank, p->size );
        errors += ((m->fd = shm_open( m->name, O_RDWR|O_CREAT|O_EXCL|O_SYNC, 0777 )) == -1);
        errors += (ftruncate(m->fd, m->sz) == -1);
    }
    MPICALL(Bcast, m->name, sizeof(m->name), MPI_BYTE, 0, p->shcomm);

    MPICALL(Reduce, p->rank == 0 ? MPI_IN_PLACE : &errors, &errors, 1, MPI_INT, MPI_SUM, 0, p->comm);
    if (errors)
        mpi_err_printf( "could not open/truncate shm file %s\n", m->name );

    errors = 0;
    if (p->shrank != 0)
        errors += ((m->fd = shm_open( m->name, O_RDWR|O_SYNC, 0777 )) == -1);

    errors += ((m->ptr = mmap( NULL, m->sz, PROT_EXEC|PROT_READ|PROT_WRITE, MAP_SHARED|MAP_SYNC, m->fd, 0 )) == (void*)-1);
    errors += ((close( m->fd ) == -1));

    MPICALL(Reduce, p->rank == 0 ? MPI_IN_PLACE : &errors, &errors, 1, MPI_INT, MPI_SUM, 0, p->comm);
    if (errors)
        mpi_err_printf( "could not shm_open, or mmap/close\n" );

#endif // SHARED_MEM_POSIX_IMPL

    errors = 0;

    m->sem = (sem_t*)((char*)m->ptr + sz);
    if (p->shrank == 0)
        errors += (sem_init( m->sem, 1, 0 ) == -1);
    MPICALL(Reduce, p->rank == 0 ? MPI_IN_PLACE : &errors, &errors, 1, MPI_INT, MPI_SUM, 0, p->comm);
    if (errors)
        mpi_err_printf( "could not perform semaphore init\n" );

    MPICALL(Barrier, p->shcomm);
    return m;
}

static shmem_pool_t* shared_mem_pool = NULL;

static void shmem_free( shmem_t* m ) {
    int errors = 0;
    if (m->rank == 0)
        errors += (sem_destroy( m->sem ) == -1);

    errors += (munmap( m->ptr, m->sz ) == -1);

#ifndef SHARED_MEM_POSIX_IMPL
    errors += (close( m->fd ) == -1);
#else
    if (m->rank == 0)
        errors += (shm_unlink( m->name ) == -1);
#endif // SHARED_MEM_POSIX_IMPL

    MPICALL(Reduce, shared_mem_pool->rank == 0 ? MPI_IN_PLACE : &errors,
            &errors, 1, MPI_INT, MPI_SUM, 0, shared_mem_pool->comm);
    if (errors)
        mpi_err_printf( "could not munmap/close shared memory\n" );

    free( m );
}

void shared_malloc_init( int more_colors ) {
    shmem_pool_more_colors = more_colors;
    shared_mem_pool = shmem_pool_init();
    shared_mem_pool->pool = hashtable_init( 128, 8 );
}

static inline void pool_free_func( index_t key, void* val, void* ctx ) {
    (void)key; (void)ctx;
    shmem_free( val );
}
void shared_malloc_finalize( void ) {
    if (!shared_mem_pool) return;
    hashtable_iterate(shared_mem_pool->pool, &pool_free_func, NULL);
    hashtable_free(shared_mem_pool->pool);
    shmem_pool_destroy( shared_mem_pool );
}

void* shared_calloc( long long cnt, long long sz ) {
    void* ptr = shared_malloc( cnt*sz );
    if (shared_exclusive_enter(ptr) == 1) memset( ptr, 0, cnt*sz );
    shared_exclusive_wait(ptr);
    return ptr;
}

void* shared_malloc( long long sz ) {
    if (!shared_mem_pool) return NULL;
    shmem_t* memptr = shmem_alloc( shared_mem_pool, sz );
    hashtable_insert( shared_mem_pool->pool, (index_t)memptr->ptr, memptr );
    return memptr->ptr;
}

void shared_free( void* ptr ) {
    if (!shared_mem_pool) return;
    if (!ptr) return;
    shmem_t* memptr = hashtable_pop( shared_mem_pool->pool, (index_t)ptr );
    if (!memptr) return;
    shmem_free( memptr );
}

int shared_exclusive_enter( void* ptr ) {
    if (!shared_mem_pool) return -1;
    if (!ptr) return -1;
    shmem_t* memptr = hashtable_find( shared_mem_pool->pool, (index_t)ptr );
    if (!memptr) return -1;

    if (memptr->rank == 0)
        return 1;
    return 0;
}

void shared_exclusive_wait( void* ptr ) {
    if (!shared_mem_pool) return;
    if (!ptr) return;
    shmem_t* memptr = hashtable_find( shared_mem_pool->pool, (index_t)ptr );
    if (!memptr) return;

    if (memptr->rank == 0)
        for (int i=0; i<memptr->size; ++i) sem_post( memptr->sem );
    sem_wait( memptr->sem );
    shared_malloc_barrier();
}

int shared_malloc_rank( void ) {
    if (!shared_mem_pool) return -1;
    return shared_mem_pool->shrank;
}

int shared_malloc_size( void ) {
    if (!shared_mem_pool) return -1;
    return shared_mem_pool->shsize;
}

static inline void pool_sem_post( index_t key, void* val, void* ctx ) {
    (void)key; (void)ctx; if (!val) return;
    shmem_t* m = val;
    if (m->rank == 0) for (int i=0; i<m->size; ++i) sem_post( m->sem );
}
static inline void pool_sem_wait( index_t key, void* val, void* ctx ) {
    (void)key; (void)ctx; if (!val) return;
    shmem_t* m = val;
    sem_wait( m->sem );
}
void shared_malloc_barrier( void ) {
    if (!shared_mem_pool) return;

    MPICALL(Barrier, shared_mem_pool->shcomm);
    hashtable_iterate(shared_mem_pool->pool, &pool_sem_post, NULL);
    hashtable_iterate(shared_mem_pool->pool, &pool_sem_wait, NULL);
    MPICALL(Barrier, shared_mem_pool->shcomm);
}

#endif // SHARED_MEM_MPI_IMPL

#else // USE_SHARED_MEM && USE_MPI

#include <stdlib.h>

void shared_malloc_init( int more_colors ) { (void)more_colors; }
void shared_malloc_finalize( void ) {}
void* shared_malloc( long long sz ) { return malloc(sz); }
void* shared_calloc( long long cnt, long long sz ) { return calloc(cnt, sz); }
void shared_free( void* ptr ) { free(ptr); }

int shared_exclusive_enter( void* ptr ) { (void)ptr; return 1; }
void shared_exclusive_wait( void* ptr ) { (void)ptr; }
int shared_malloc_rank( void ) { return 0; }
int shared_malloc_size( void ) { return 1; }
void shared_malloc_barrier( void ) {}

#endif // USE_SHARED_MEM && USE_MPI

/* EXAMPLE

// init lib
shared_malloc_init();

size_t N = 1024ul*1024ul*128ul;
double* ptr = shared_malloc( sizeof(double)*N );

if (shared_exclusive_enter(ptr) == 1) {
    // do sth exclusively
}
shared_exclusive_wait(ptr); // wait for exclusive region

shared_free( ptr );

// finalize lib
shared_malloc_finalize();

*/
