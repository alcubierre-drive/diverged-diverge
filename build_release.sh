#!/bin/bash

START_LINE_COMPILE_SCRIPT=22
make clean
mkdir -p build_release
tail -n+${START_LINE_COMPILE_SCRIPT} $0 > build_release/compile.sh
chmod +x build_release/compile.sh
mkdir -p build_release/diverge
cp -r src/ build_release/diverge
cp Makefile build_release/diverge
cp -r test/ build_release/diverge
cp libdivERGe.map build_release/diverge
cp -r examples/ build_release/diverge
cp -r util/ build_release/diverge
TAG_VERSION="$(git tag --list "v*.*" | tail -n1)"
echo 'DEFINE_VERSION := -DTAG_VERSION=\"'${TAG_VERSION}'\"' > build_release/diverge/Makefile.local
docker run -t -i --rm -v `pwd`/build_release:/io \
    ghcr.io/phusion/holy-build-box/hbb-64 \
    /hbb_exe/activate-exec bash /io/compile.sh
exit 0

#!/bin/bash
set -e
# Activate Holy Build Box environment
source /hbb_exe/activate
set -x
ROOT_PATH="$(pwd)"

BUILD_SOFTWARE=false
OPENBLAS_VERSION=0.3.28
EIGEN_VERSION=3.4.0
FFTW_VERSION=3.3.10

# libraries
SOFTWARE_PATH=/io/software
mkdir -p ${SOFTWARE_PATH}
if $BUILD_SOFTWARE; then
cd ${SOFTWARE_PATH}
mkdir -p src
# OPENBLAS
cd ${SOFTWARE_PATH}/src
curl -LO https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v${OPENBLAS_VERSION}.tar.gz
tar xvf v${OPENBLAS_VERSION}.tar.gz
cd "OpenBLAS-${OPENBLAS_VERSION}"
mkdir -p build
cd build
env CFLAGS="-fPIC $STATICLIB_CFLAGS" CXXFLAGS="-fPIC $STATICLIB_CXXFLAGS" \
    cmake .. -DCMAKE_INSTALL_PREFIX=${SOFTWARE_PATH} -DUSE_OPENMP=1 -DNOFORTRAN=1
make -j12
make install
# EIGEN
cd ${SOFTWARE_PATH}/src
curl -LO https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz
tar xvf eigen-${EIGEN_VERSION}.tar.gz
cd ${SOFTWARE_PATH}/include
ln -s ../src/eigen-${EIGEN_VERSION} eigen3
# FFTW
cd ${SOFTWARE_PATH}/src
curl -LO https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz
tar xvf fftw-${FFTW_VERSION}.tar.gz
cd fftw-${FFTW_VERSION}
env CFLAGS="-fPIC $STATICLIB_CFLAGS" CXXFLAGS="-fPIC $STATICLIB_CXXFLAGS" \
    ./configure --enable-float --prefix=${SOFTWARE_PATH}
make -j12
make install
env CFLAGS="-fPIC $STATICLIB_CFLAGS" CXXFLAGS="-fPIC $STATICLIB_CXXFLAGS" \
    ./configure --enable-float --enable-openmp --prefix=${SOFTWARE_PATH}
make -j12
make install
env CFLAGS="-fPIC $STATICLIB_CFLAGS" CXXFLAGS="-fPIC $STATICLIB_CXXFLAGS" \
    ./configure --prefix=${SOFTWARE_PATH}
make -j12
make install
env CFLAGS="-fPIC $STATICLIB_CFLAGS" CXXFLAGS="-fPIC $STATICLIB_CXXFLAGS" \
    ./configure --enable-openmp --prefix=${SOFTWARE_PATH}
make -j12
make install
fi # BUILD_SOFTWARE

# diverge
DIVERGE_PATH=/io/diverge
cd ${DIVERGE_PATH}
cat >> Makefile.local <<EOF
INCLUDES += -I${SOFTWARE_PATH}/include -I${SOFTWARE_PATH}/include/openblas
LIBS := ${SOFTWARE_PATH}/lib64/libopenblas.a ${SOFTWARE_PATH}/lib/libfftw3.a ${SOFTWARE_PATH}/lib/libfftw3_omp.a
FLAGS += -Ofast -mtune=native -flto -DNDEBUG -DEIGEN_NO_DEBUG
LDFLAGS += -flto=12 -static-libstdc++ -Wno-lto-type-mismatch -Ofast -mtune=native
EOF
make -j12 test
