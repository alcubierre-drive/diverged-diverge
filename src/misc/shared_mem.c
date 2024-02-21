/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

// TODO(LK) append an interface to this MPI shared memory to the memory pool

#include "shared_mem.h"

#if defined(USE_SHARED_MEM) && defined(USE_MPI)

#ifdef SHARED_MEM_MPI_IMPL

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <uthash.h>

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

    UT_hash_handle hh;
} shmem_t;

typedef struct {
    int rank, size;
    MPI_Comm comm;
    int shrank, shsize;
    MPI_Comm shcomm;

    shmem_t* pool;
} shmem_pool_t;

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

void shared_malloc_init( void ) {
    shared_mem_pool = shmem_pool_init();
}
void shared_malloc_finalize( void ) {
    if (!shared_mem_pool) return;
    shmem_t *mem = NULL, *tmp = NULL;
    HASH_ITER(hh, shared_mem_pool->pool, mem, tmp) {
        HASH_DEL(shared_mem_pool->pool, mem);
        shmem_free( mem );
    }
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
    HASH_ADD_PTR( shared_mem_pool->pool, ptr, memptr );
    return memptr->ptr;
}

void shared_free( void* ptr ) {
    if (!shared_mem_pool) return;
    if (!ptr) return;
    shmem_t* memptr = NULL;
    HASH_FIND_PTR( shared_mem_pool->pool, &ptr, memptr );
    if (!memptr) return;
    HASH_DEL( shared_mem_pool->pool, memptr );
    shmem_free( memptr );
}

int shared_exclusive_enter( void* ptr ) {
    if (!shared_mem_pool) return -1;
    if (!ptr) return -1;
    shmem_t* memptr = NULL;
    HASH_FIND_PTR( shared_mem_pool->pool, &ptr, memptr );
    if (!memptr) return -1;

    MPICALL(Win_lock, MPI_LOCK_SHARED, 0, MPI_MODE_NOCHECK, memptr->win);
    if (memptr->rank == 0)
        return 1;
    return 0;
}

void shared_exclusive_wait( void* ptr ) {
    if (!shared_mem_pool) return;
    if (!ptr) return;
    shmem_t* memptr = NULL;
    HASH_FIND_PTR( shared_mem_pool->pool, &ptr, memptr );
    if (!memptr) return;

    MPICALL(Win_unlock, 0, memptr->win);
}

int shared_malloc_rank( void ) {
    if (!shared_mem_pool) return -1;
    return shared_mem_pool->shrank;
}

int shared_malloc_size( void ) {
    if (!shared_mem_pool) return -1;
    return shared_mem_pool->shsize;
}

void shared_malloc_barrier( void ) {
    if (!shared_mem_pool) return;

    shmem_t* m = NULL;
    for (m = shared_mem_pool->pool; m != NULL; m = m->hh.next) {
        shared_exclusive_enter( m->ptr );
        shared_exclusive_wait( m->ptr );
    }
    MPICALL(Barrier, shared_mem_pool->shcomm);
}

#else // SHARED_MEM_MPI_IMPL

#define _GNU_SOURCE
#include <sys/mman.h>
#include <linux/memfd.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>

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

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <uthash.h>

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
    union {
        void* ptr;
        int64_t id;
    };
    sem_t* sem;
    size_t sz;
    int fd;

    int rank, size;
    MPI_Comm comm;

    UT_hash_handle hh;
} shmem_t;

typedef struct {
    int rank, size;
    MPI_Comm comm;
    int shrank, shsize;
    MPI_Comm shcomm;

    shmem_t* pool;
} shmem_pool_t;

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
    char fd_path[64];
    if (p->shrank == 0) {
        // tame the beast (i.e. create the memfd) if we're shared rank 0
        m->fd = memfd_create( "666", MFD_CLOEXEC|MFD_ALLOW_SEALING );
        snprintf( fd_path, sizeof(fd_path), "/proc/%d/fd/%d", getpid(), m->fd );
        ftruncate( m->fd, m->sz );
    }
    MPICALL(Bcast, fd_path, sizeof(fd_path), MPI_BYTE, 0, p->shcomm);
    if (p->shrank != 0)
        m->fd = open(fd_path, O_RDWR);
    m->ptr = mmap( NULL, m->sz, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_SHARED, m->fd, 0 );
    m->sem = (sem_t*)((char*)m->ptr + sz);
    if (p->shrank == 0) sem_init( m->sem, 1, 0 );
    MPICALL(Barrier, p->shcomm);
    return m;
}

static void shmem_free( shmem_t* m ) {
    if (m->rank == 0) sem_destroy( m->sem );
    munmap( m->ptr, m->sz );
    close( m->fd );
    free( m );
}

static shmem_pool_t* shared_mem_pool = NULL;

void shared_malloc_init( void ) {
    shared_mem_pool = shmem_pool_init();
}
void shared_malloc_finalize( void ) {
    if (!shared_mem_pool) return;
    shmem_t *mem = NULL, *tmp = NULL;
    HASH_ITER(hh, shared_mem_pool->pool, mem, tmp) {
        HASH_DEL(shared_mem_pool->pool, mem);
        shmem_free( mem );
    }
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
    HASH_ADD_INT( shared_mem_pool->pool, id, memptr );
    return memptr->ptr;
}

void shared_free( void* ptr ) {
    if (!shared_mem_pool) return;
    if (!ptr) return;
    int64_t id = (int64_t)ptr;
    shmem_t* memptr = NULL;
    HASH_FIND_INT( shared_mem_pool->pool, &id, memptr );
    if (!memptr) return;
    HASH_DEL( shared_mem_pool->pool, memptr );
    shmem_free( memptr );
}

int shared_exclusive_enter( void* ptr ) {
    if (!shared_mem_pool) return -1;
    if (!ptr) return -1;
    int64_t id = (int64_t)ptr;
    shmem_t* memptr = NULL;
    HASH_FIND_INT( shared_mem_pool->pool, &id, memptr );
    if (!memptr) return -1;

    if (memptr->rank == 0)
        return 1;
    return 0;
}

void shared_exclusive_wait( void* ptr ) {
    if (!shared_mem_pool) return;
    if (!ptr) return;
    int64_t id = (int64_t)ptr;
    shmem_t* memptr = NULL;
    HASH_FIND_INT( shared_mem_pool->pool, &id, memptr );
    if (!memptr) return;

    if (memptr->rank == 0)
        for (int i=0; i<memptr->size; ++i) sem_post( memptr->sem );
    sem_wait( memptr->sem );
}

int shared_malloc_rank( void ) {
    if (!shared_mem_pool) return -1;
    return shared_mem_pool->shrank;
}

int shared_malloc_size( void ) {
    if (!shared_mem_pool) return -1;
    return shared_mem_pool->shsize;
}

void shared_malloc_barrier( void ) {
    if (!shared_mem_pool) return;

    shmem_t* m = NULL;
    for (m = shared_mem_pool->pool; m != NULL; m = m->hh.next) {
        shared_exclusive_enter( m->ptr );
        shared_exclusive_wait( m->ptr );
    }
    MPICALL(Barrier, shared_mem_pool->shcomm);
}

#endif // SHARED_MEM_MPI_IMPL

#else // USE_SHARED_MEM && USE_MPI

#include <stdlib.h>

void shared_malloc_init( void ) {}
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
