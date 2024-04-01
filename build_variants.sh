#!/bin/bash

Makefile_local_CPU='
DEFINES := -DNDEBUG -DEIGEN_NO_DEBUG -DDIVERGE_OPENBLAS_LIMIT_THREADS=40
DEFINES += -DDIVERGE_LOG_GRAYSCALE -DUSE_FPLO_READER

EXEFLAGS := -L. -Wl,-rpath=. -l:libdivERGe.so
DEFINE_VERSION := -DTAG_VERSION=\"$(shell git tag --list "v*.*" | tail -n1)\"
INCLUDES += -Istatic/include -Istatic/include/openblas
PREFLAGS = -include static/include/force_link_glibc_2.17.h
CXXFLAGS += -Ofast -mtune=native
CFLAGS += -Ofast -mtune=native
LIBS := -Lstatic/lib -Wl,-Bstatic -lfftw3 -lfftw3_omp -lopenblas -Wl,-Bdynamic \
	-Wl,--as-needed -static-libgcc -static-libstdc++
'

Makefile_local_CUDA10_1='
USE_CUDA := true
CUDA_ROOT := /usr/local/cuda-10.1
CUCC := $(CUDA_ROOT)/bin/nvcc
DEFINES += -DCUDA_CIRCUMVENT_CONSTEXPR -DCUDA_VERSION_STR="\"CUDA $(shell cat $(CUDA_ROOT)/version.txt)\""
LIBPREFIX_CUDA := -L$(CUDA_ROOT)/lib64 -l
LIBPREFIX_CUBLAS := -L/usr/lib64/lib -l
CULIBS := $(LIBPREFIX_CUDA)cudart_static \
	  $(LIBPREFIX_CUDA)cufft_static \
	  $(LIBPREFIX_CUBLAS)cublas_static \
	  $(LIBPREFIX_CUBLAS)cublasLt_static \
	  $(LIBPREFIX_CUDA)cusolver_static \
	  $(LIBPREFIX_CUDA)cudart_static \
	  $(LIBPREFIX_CUDA)culibos \
	  $(LIBPREFIX_CUDA)lapack_static \
	  $(LIBPREFIX_CUDA)metis_static \
	  $(LIBPREFIX_CUBLAS)cublas_static \
	  $(LIBPREFIX_CUDA)cusparse_static \
	  -ldl -lrt
CULDFLAGS += $(LIBPREFIX_CUDA)cudart_static $(LIBPREFIX_CUDA)cufft_static \
	     $(LIBPREFIX_CUBLAS)cublas_static $(LIBPREFIX_CUBLAS)cublasLt_static
CUFLAGS = $(DEFINES) -DSKIP_VERTEX_RECO_INCLUDE $(INCLUDES) $(DEVICES) \
	  -Xcompiler="-fPIC -fopenmp -std=c++11" -std=c++11 -rdc=true \
	  --expt-relaxed-constexpr --use_fast_math -O9
INCLUDES += -I$(CUDA_ROOT)/include
DEVICES := -arch=compute_75 \
-gencode arch=compute_60,code=sm_60 \
-gencode arch=compute_61,code=sm_61 \
-gencode arch=compute_62,code=sm_62 \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_72,code=sm_72 \
-gencode arch=compute_75,code=sm_75
'

Makefile_local_CUDA10_2='
USE_CUDA := true
CUDA_ROOT := /usr/local/cuda-10.2
CUCC := $(CUDA_ROOT)/bin/nvcc
DEFINES += -DCUDA_CIRCUMVENT_CONSTEXPR -DCUDA_VERSION_STR="\"CUDA $(shell cat $(CUDA_ROOT)/version.txt)\""
LIBPREFIX_CUDA := -L$(CUDA_ROOT)/lib64 -l
CULIBS := $(LIBPREFIX_CUDA)cudart_static \
	  $(LIBPREFIX_CUDA)cufft_static \
	  $(LIBPREFIX_CUDA)cublas_static \
	  $(LIBPREFIX_CUDA)cublasLt_static \
	  $(LIBPREFIX_CUDA)cusolver_static \
	  $(LIBPREFIX_CUDA)cudart_static \
	  $(LIBPREFIX_CUDA)culibos \
	  $(LIBPREFIX_CUDA)lapack_static \
	  $(LIBPREFIX_CUDA)metis_static \
	  $(LIBPREFIX_CUDA)cublas_static \
	  $(LIBPREFIX_CUDA)cusparse_static \
	  -ldl -lrt
CULDFLAGS += $(LIBPREFIX_CUDA)cudart_static $(LIBPREFIX_CUDA)cufft_static \
	     $(LIBPREFIX_CUDA)cublas_static $(LIBPREFIX_CUDA)cublasLt_static
CUFLAGS = $(DEFINES) -DSKIP_VERTEX_RECO_INCLUDE $(INCLUDES) $(DEVICES) \
	  -Xcompiler="-fPIC -fopenmp -std=c++11" -std=c++11 -rdc=true \
	  --expt-relaxed-constexpr --use_fast_math -O9
INCLUDES += -I$(CUDA_ROOT)/include
DEVICES := -arch=compute_75 \
-gencode arch=compute_60,code=sm_60 \
-gencode arch=compute_61,code=sm_61 \
-gencode arch=compute_62,code=sm_62 \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_72,code=sm_72 \
-gencode arch=compute_75,code=sm_75
'

Makefile_local_CUDA11_0='
USE_CUDA := true
CUDA_ROOT := /usr/local/cuda-11.0
CUCC := $(CUDA_ROOT)/bin/nvcc
DEFINES += -DCUDA_VERSION_STR="\"$(shell cat $(CUDA_ROOT)/version.txt)\""
LIBPREFIX_CUDA := -L$(CUDA_ROOT)/lib64 -l
CULIBS := $(LIBPREFIX_CUDA)cufft_static \
	  $(LIBPREFIX_CUDA)cudart_static \
	  $(LIBPREFIX_CUDA)cublas_static \
	  $(LIBPREFIX_CUDA)cublasLt_static \
	  $(LIBPREFIX_CUDA)cusolver_static \
	  $(LIBPREFIX_CUDA)cudart_static \
	  $(LIBPREFIX_CUDA)culibos \
	  $(LIBPREFIX_CUDA)lapack_static \
	  $(LIBPREFIX_CUDA)metis_static \
	  $(LIBPREFIX_CUDA)cublas_static \
	  $(LIBPREFIX_CUDA)cusparse_static \
	  -ldl -lrt
CULDFLAGS += $(LIBPREFIX_CUDA)cudart_static $(LIBPREFIX_CUDA)cufft_static \
	     $(LIBPREFIX_CUDA)cublas_static $(LIBPREFIX_CUDA)cublasLt_static
CUFLAGS = $(DEFINES) $(INCLUDES) $(DEVICES) -Xcompiler="-fPIC -fopenmp -std=c++17" -std=c++17 \
	  -rdc=true --expt-relaxed-constexpr --use_fast_math -O9
INCLUDES += -I$(CUDA_ROOT)/include
DEVICES := -arch=compute_80 \
-gencode arch=compute_60,code=sm_60 \
-gencode arch=compute_61,code=sm_61 \
-gencode arch=compute_62,code=sm_62 \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_72,code=sm_72 \
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_80,code=sm_80
'

Makefile_local_CUDA11_8='
USE_CUDA := true
CUDA_ROOT := /usr/local/cuda-11.8
CUCC := $(CUDA_ROOT)/bin/nvcc
DEFINES += -DCUDA_VERSION_STR="\"CUDA Version $(shell cat $(CUDA_ROOT)/version.json |\
	   grep cccl -n2 | grep version | tail -n1 | cut -d":" -f2 | cut -c 3- | rev | cut -c 2- | rev)\""
LIBPREFIX_CUDA := -L$(CUDA_ROOT)/lib64 -l
CULIBS := $(LIBPREFIX_CUDA)cufft_static \
	  $(LIBPREFIX_CUDA)cudart_static \
	  $(LIBPREFIX_CUDA)cublas_static \
	  $(LIBPREFIX_CUDA)cublasLt_static \
	  $(LIBPREFIX_CUDA)cusolver_static \
	  $(LIBPREFIX_CUDA)cudart_static \
	  $(LIBPREFIX_CUDA)culibos \
	  $(LIBPREFIX_CUDA)lapack_static \
	  $(LIBPREFIX_CUDA)metis_static \
	  $(LIBPREFIX_CUDA)cublas_static \
	  $(LIBPREFIX_CUDA)cusparse_static \
	  -ldl -lrt
CULDFLAGS += $(LIBPREFIX_CUDA)cudart_static $(LIBPREFIX_CUDA)cufft_static \
	     $(LIBPREFIX_CUDA)cublas_static $(LIBPREFIX_CUDA)cublasLt_static
CUFLAGS = $(DEFINES) $(INCLUDES) $(DEVICES) -Xcompiler="-fPIC -fopenmp -std=c++17" -std=c++17 \
	  -rdc=true --expt-relaxed-constexpr --use_fast_math -O9 -diag-suppress=20236,20012
INCLUDES += -I$(CUDA_ROOT)/include
DEVICES := -arch=compute_80 \
-gencode arch=compute_60,code=sm_60 \
-gencode arch=compute_61,code=sm_61 \
-gencode arch=compute_62,code=sm_62 \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_72,code=sm_72 \
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_86,code=sm_86 \
-gencode arch=compute_87,code=sm_87 \
-gencode arch=compute_89,code=sm_89 \
-gencode arch=compute_90,code=sm_90
'

Makefile_local_CUDA12_2='
USE_CUDA := true
CUDA_ROOT := /usr/local/cuda-12.2
CUCC := $(CUDA_ROOT)/bin/nvcc
DEFINES += -DCUDA_VERSION_STR="\"CUDA Version $(shell cat $(CUDA_ROOT)/version.json |\
	   grep core -i -n1 | tail -n1 | cut -d":" -f2 | cut -c 3- | rev | cut -c 2- | rev)\""
LIBPREFIX_CUDA := -L$(CUDA_ROOT)/lib64 -l
CULIBS := $(LIBPREFIX_CUDA)cudart_static \
	  $(LIBPREFIX_CUDA)cufft_static \
	  $(LIBPREFIX_CUDA)cublas_static \
	  $(LIBPREFIX_CUDA)cublasLt_static \
	  $(LIBPREFIX_CUDA)cusolver_static \
	  $(LIBPREFIX_CUDA)cudart_static \
	  $(LIBPREFIX_CUDA)culibos \
	  $(LIBPREFIX_CUDA)metis_static \
	  $(LIBPREFIX_CUDA)cublas_static \
	  $(LIBPREFIX_CUDA)cusparse_static \
	  -ldl -lrt
CULDFLAGS += $(LIBPREFIX_CUDA)cudart_static $(LIBPREFIX_CUDA)cufft_static \
	     $(LIBPREFIX_CUDA)cublas_static $(LIBPREFIX_CUDA)cublasLt_static
CUFLAGS = $(DEFINES) -DSKIP_VERTEX_RECO_INCLUDE $(INCLUDES) $(DEVICES) \
	  -Xcompiler="-fPIC -fopenmp -std=c++17" -rdc=true --expt-relaxed-constexpr -std=c++17 \
	  -diag-suppress=20012 --use_fast_math -O9
INCLUDES += -I$(CUDA_ROOT)/include
DEVICES := -arch=compute_80 \
-gencode arch=compute_60,code=sm_60 \
-gencode arch=compute_61,code=sm_61 \
-gencode arch=compute_62,code=sm_62 \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_72,code=sm_72 \
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_86,code=sm_86 \
-gencode arch=compute_87,code=sm_87 \
-gencode arch=compute_89,code=sm_89 \
-gencode arch=compute_90,code=sm_90 \
-gencode arch=compute_90a,code=sm_90a
'

function gen_Makefile_local() {
    # $1: combination: CPU, CUDAXX
    echo "$Makefile_local_CPU"
    if [ "$1" != "CPU" ]
    then
        varname=Makefile_local_$1
        echo "${!varname}"
    fi
}

function build_variant() {
    # build variant $1
    # output to $2

    # back up Makefile.local
    mv Makefile.local Makefile.local.bak

    mkdir -p "$2"
    gen_Makefile_local "$1" > Makefile.local
    make clean
    make -j $(nproc) dist
    make -j $(npric) dist-src
    cp libdivERGe.so divERGe "$2"

    mv Makefile.local.bak Makefile.local
}

version="$(git tag | tail -n1)"
prefix="public/releases/${version}/"
build_variant CPU "${prefix}cpu/"
cp divERGe.tar.gz ${prefix}
cp divERGe_src.tar.gz ${prefix}

if [ $1 = 'CPU' ]
then
    cp "${prefix}cpu/libdivERGe.so" .
    exit 0
fi

# build_variant CUDA10_1 "${prefix}/cuda10-1/"
# build_variant CUDA10_2 "${prefix}/cuda10-2/"
# build_variant CUDA11_0 "${prefix}/cuda11-0/"
# build_variant CUDA11_8 "${prefix}/cuda11-8/"
# build_variant CUDA12_2 "${prefix}/cuda12-2/"
