NAME := divERGe
NAME_GPU := divERGe_gpu.a
NAME_GPU_LINK := divERGe_gpu_link.a
NAME_LIB := libdivERGe.so

CC := gcc
LD := g++
CXX := g++
CUCC := nvcc
AR := ar
STRIP := strip

CXXFLAGS = $(FLAGS) -std=c++17 -fPIC -fopenmp -MMD -Wall -Wextra -pedantic $(DEFINES) $(INCLUDES)
CFLAGS = $(FLAGS) -std=c11 -fPIC -fopenmp -MMD -Wall -Wextra -pedantic $(DEFINES) $(INCLUDES)
LDFLAGS = -fPIC -fopenmp -Wall -Wextra -pedantic $(LIBS)
CUFLAGS = $(DEFINES) $(INCLUDES) $(DEVICES) -Xcompiler="-fPIC -fopenmp" -rdc=true
CULDFLAGS = -Xcompiler="-fPIC"

# for linkage:
# EXEFLAGS: linking of the main executable
# SOFLAGS: linking of the library
# LDSOFLAGS: LDFLAGS used for linking of the library
EXEFLAGS = -L. -l:$(NAME_LIB) -Wl,-rpath=$(shell pwd)
SOFLAGS = -Wl,--version-script=libdivERGe.map -shared
LDSOFLAGS = $(LDFLAGS)

FLAGS = 
INCLUDES = 
DEFINES = 
LIBS = -llapacke -lcblas -lfftw3 -lfftw3_omp
CULIBS = -lcudart -lcuda -lcufft -lcublas -lcusolver -lcusolverMg
DEVICES = -gencode arch=compute_70,code=sm_70 \
	  -gencode arch=compute_75,code=sm_75 \
	  -gencode arch=compute_80,code=sm_80

DEFINE_VERSION := -DTAG_VERSION=\"$(shell git tag --list "v*.*" | tail -n1)\" \
		  -DGIT_VERSION=\"$(shell git rev-parse --short HEAD)$(shell (git diff --quiet && git diff --cached --exit-code --quiet) || echo ':dirty')\" \
		  -DGIT_VERSION_BRANCH=\"$(shell git rev-parse --abbrev-ref HEAD)\"

SRC_C := $(shell find ./src/ -name '*.c') \
         $(shell find ./test/ -name '*.c')
SRC_CPP := $(shell find ./src/ -name '*.cpp') \
           $(shell find ./test/ -name '*.cpp')
SRC_CU := $(shell find ./src/ -name '*.cu')

IGNORED_DIRS := $(shell find . -type d -name ignore)
ifneq ($(strip $(IGNORED_DIRS)),)
IGNORED_SRC_C := $(shell find $(IGNORED_DIRS) -name '*.c')
IGNORED_SRC_CPP := $(shell find $(IGNORED_DIRS) -name '*.cpp')
ifneq ($(strip $(IGNORED_SRC_C)),)
SRC_C := $(filter-out $(IGNORED_SRC_C),$(SRC_C))
endif
ifneq ($(strip $(IGNORED_SRC_CPP)),)
SRC_CPP := $(filter-out $(IGNORED_SRC_CPP),$(SRC_CPP))
endif
endif

OBJ_C := $(patsubst %.c,%.c.o,$(SRC_C))
OBJ_CPP := $(patsubst %.cpp,%.cpp.o,$(SRC_CPP))
OBJ_CU := $(patsubst %.cu,%.cu.o,$(SRC_CU))

DEP_C := $(patsubst %.c,%.c.d,$(SRC_C))
DEP_CPP := $(patsubst %.cpp,%.cpp.d,$(SRC_CPP))

HEADERS := $(shell find src/ -name '*.h') \
           $(shell find src/ -name '*.hpp') \
           $(shell find test/ -name '*.h') \
           $(shell find test/ -name '*.hpp')

-include Makefile.inc
-include Makefile.local

.PHONY: lib test clean types dist dist-src todos doc man pdf $(NAME_GPU) $(NAME_GPU_LINK) src/misc/version.c.o

# resolves nasty circular dependency bug for .cu(.o) files
.SUFFIXES: 
.SUFFIXES: .cu .cu.o

lib: types $(NAME_LIB)

test: types $(NAME)

-include $(DEP_C) $(DEP_CPP)

$(NAME): src/main.c.o $(NAME_LIB)
	$(LD) $(EXEFLAGS) $< $(LDFLAGS) -o $@

ifdef USE_CUDA
DEFINES += -DUSE_CUDA
LIBS += $(CULIBS)
$(NAME_GPU_LINK): $(NAME_GPU)
	$(CUCC) $^ -o $@ -dlink $(CULDFLAGS)
$(NAME_LIB): $(OBJ_C) $(OBJ_CPP) $(NAME_GPU) $(NAME_GPU_LINK) libdivERGe.map
	$(LD) $(SOFLAGS) $(filter-out src/main.c.o libdivERGe.map,$^) $(LDSOFLAGS) -o $@
	$(STRIP) $(NAME_LIB)
else
$(NAME_LIB): $(OBJ_C) $(OBJ_CPP) libdivERGe.map
	$(LD) $(SOFLAGS) $(filter-out src/main.c.o libdivERGe.map,$^) $(LDSOFLAGS) -o $@
	$(STRIP) $(NAME_LIB)
endif

src/misc/version.c.o: src/misc/version.c src/misc/version.h
	$(CC) -c $< -o $@ $(CFLAGS) $(DEFINE_VERSION)

%.c.o: %.c %.h Makefile $(wildcard Makefile.local Makefile.inc)
	$(CC) -c $< -o $@ $(CFLAGS)
%.c.o: %.c Makefile $(wildcard Makefile.local Makefile.inc)
	$(CC) -c $< -o $@ $(CFLAGS)

%.cpp.o: %.cpp %.h Makefile $(wildcard Makefile.local Makefile.inc)
	$(CXX) -c $< -o $@ $(CXXFLAGS)
%.cpp.o: %.cpp Makefile $(wildcard Makefile.local Makefile.inc)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

%.cu.o: %.cu %.cu.h Makefile $(wildcard Makefile.local Makefile.inc)
	$(CUCC) -c $< -o $@ $(CUFLAGS)
%.cu.o: %.cu Makefile $(wildcard Makefile.local Makefile.inc)
	$(CUCC) -c $< -o $@ $(CUFLAGS)

types: types.vim
types.vim: $(SRC_C) $(SRC_CPP) $(HEADERS) Makefile $(wildcard Makefile.local Makefile.inc)
	@echo "adding ctypes to vim..."
	@ctags --c-kinds=t -o- $(SRC_C) $(SRC_CPP) $(HEADERS) |\
		awk '{print "syntax keyword Type " $$1}' | uniq > $@

clean:
	-@$(RM) -rf $(NAME) $(NAME_GPU) $(NAME_GPU_LINK) $(NAME_LIB) types.vim
	-@$(RM) -rf $(shell find . -type f -name '*.o') $(shell find . -type f -name '*.d')
	-@$(RM) -rf divERGe.tar.gz divERGe_src.tar
	make -C sphinx clean

$(NAME_GPU): $(OBJ_CU)
	$(AR) -crs $@ $^

todos:
	grep -R TODO src/ test/

DIVERGE_TAR ?= tar
DIVERGE_GZIP ?= gzip
DIVERGE_MV ?= mv
DIVERGE_CP ?= cp
DIVERGE_MKDIR ?= mkdir
DIVERGE_RM ?= rm
export DIVERGE_TAR
export DIVERGE_GZIP
export DIVERGE_MV
export DIVERGE_CP
export DIVERGE_MKDIR
export DIVERGE_RM
dist: $(NAME) $(NAME_LIB)
	$(MAKE) $(MFLAGS) -C util

EXAMPLES := $(shell find examples/ -iname "*.cpp") \
	$(shell find examples/ -iname "*.c") \
	$(shell find examples/ -iname "*.py") \
	$(shell find examples/ -iname Makefile)
UTIL := $(shell find util/ -iname "*.py") \
	setup.py setup.cfg libdivERGe.map LICENSE.md README.md TODO.md
dist-src: $(SRC_C) $(SRC_CPP) $(SRC_CU) $(HEADERS) Makefile make.local/ $(EXAMPLES) $(UTIL)
	$(DIVERGE_TAR) -c util/libdivERGe.so $^ -f divERGe_src.tar
	$(DIVERGE_GZIP) -f divERGe_src.tar

doc:
	make -C sphinx html
	@bash sphinx/publish.sh
man:
	make -C sphinx man
pdf:
	make -C sphinx latexpdf
