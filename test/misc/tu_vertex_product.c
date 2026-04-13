#define CT_SUITE_NAME tu_vertex_product
#include "../ctest.h"

#ifndef DIVERGE_SKIP_TESTS

#include "../../src/diverge_model.h"
#include "../../src/diverge_flow_step.h"
#include "../../src/diverge_model_internals.h"
#include "../../src/misc/mpi_functions.h"
#include "../../src/tu/vertex.h"
#include "../../src/tu/cuda_init.h"
#include "../helper_functions.h"

CT_TEST(tu_vertex_product) {
    index_t N = 12;
    index_t b = 5;
    complex128_t* B = (complex128_t*)malloc(N*N*b*sizeof(complex128_t));
    complex128_t* C = (complex128_t*)malloc(N*N*b*sizeof(complex128_t));
    complex128_t* r1 = (complex128_t*)malloc(N*N*b*sizeof(complex128_t));
    complex128_t* r2 = (complex128_t*)malloc(N*N*b*sizeof(complex128_t));
    complex128_t* help = (complex128_t*)malloc(N*N*b*sizeof(complex128_t));

    for (index_t j=0; j<N*N*b; ++j) {
        B[j] = random_cmplx();
        C[j] = random_cmplx();
    }
    char* buf[32][3] = {0};

    // TODO there shouldn't be an explicit USE_CUDA in the tests
    #ifdef USE_CUDA
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    for(int i = 0; i<devices;++i) {
        cudaSetDevice( i );
        char* p1;
        char* p2;
        char* p3;

        CUDA_CHECK( cudaMalloc((void**)&p1, N*N*sizeof(cuDoubleComplex)) );
        CUDA_CHECK( cudaMalloc((void**)&p2, N*N*sizeof(cuDoubleComplex)) );
        CUDA_CHECK( cudaMalloc((void**)&p3, N*N*sizeof(cuDoubleComplex)) );
        buf[i][2] = p3;buf[i][0] = p1;buf[i][1] = p2;
    }
    #endif

    tu_vertex_product(r1,B,C,help,N,b,1.0,true,buf);
    #pragma omp parallel for collapse(3) schedule(static) num_threads(diverge_omp_num_threads())
    for(index_t k = 0; k<b;++k)
    for(index_t o1 = 0; o1<N;++o1)
    for(index_t o2 = 0; o2<N;++o2){
        complex128_t tmp = 0.0;
        for(index_t o3 = 0; o3<N;++o3)
        for(index_t o4 = 0; o4<N;++o4) {
            tmp += B[o1+N*(o3+N*k)]*
                    C[o3+N*(o4+N*k)]*
                    B[o4+N*(o2+N*k)];
        }
        r2[o1+N*(o2+N*k)] = tmp;
    }
    double max_diff = 0.0;
    for (index_t j=0; j<N*N*b; ++j) {
        if (cabs(r1[j]-r2[j]) > max_diff) max_diff = cabs(r1[j]-r2[j]);
    }
    CT_CHECK(max_diff < 1e-11);

    free(r1);
    free(r2);
    free(C);
    free(B);
    free(help);

    // TODO there shouldn't be an explicit USE_CUDA in the tests
    #ifdef USE_CUDA
    int devices2 = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices2));
    for(int i = 0; i<devices2;++i) {
        cudaSetDevice( i );
        CUDA_CHECK( cudaFree((buf[i][0])) );
        CUDA_CHECK( cudaFree((buf[i][1])) );
        CUDA_CHECK( cudaFree((buf[i][2])) );
    }
    #endif
}

#endif // DIVERGE_SKIP_TESTS
