.PHONY: pack

EXAMPLES_PY := $(wildcard ../examples/python/*.py ../examples/python/*.dat)
EXAMPLES_TUT := $(wildcard ../examples/python_tutorial/*.py ../examples/python_tutorial/*.dat)
EXAMPLES_C := $(wildcard ../examples/c_cpp/*.c ../examples/c_cpp/*.cpp ../examples/c_cpp/Makefile)
PYTHON_FILES := $(wildcard *.py)
BINARIES := ../divERGe ../libdivERGe.so

EXAMPLES := $(EXAMPLES_PY) $(EXAMPLES_TUT) $(EXAMPLES_C)
LIBS := $(PYTHON_FILES) $(BINARIES)

INCLUDES_REL := diverge.h $(shell ./list_includes_rel.sh)
INCLUDES := $(patsubst %,../src/%,$(INCLUDES_REL)) ../src/diverge_Eigen3.hpp

DIVERGE_TAR ?= tar
DIVERGE_GZIP ?= gzip -9
DIVERGE_MV ?= mv
DIVERGE_CP ?= cp
DIVERGE_MKDIR ?= mkdir
DIVERGE_RM ?= rm
DIVERGE_LN ?= ln
pack:
	$(DIVERGE_RM) -rf pack/
	$(DIVERGE_MKDIR) -p pack/examples/ pack/include/ pack/util/
	# examples
	$(DIVERGE_CP) $(EXAMPLES) pack/examples/
	# libs
	$(DIVERGE_CP) $(PYTHON_FILES) pack/util/
	$(DIVERGE_CP) $(BINARIES) pack/
	$(DIVERGE_CP) ../setup.py ../setup.cfg ../README.md ../LICENSE.md pack/
	$(DIVERGE_CP) Makefile.local_examples pack/examples/Makefile.local
	# headers
	$(DIVERGE_CP) --parents $(INCLUDES) pack/include/
	$(DIVERGE_MV) pack/src/* pack/include/
	$(DIVERGE_RM) -rf pack/src
	# and pack it up
	cd pack/util; $(DIVERGE_LN) -s ../libdivERGe.so .
	cd pack/; $(DIVERGE_TAR) -f ../divERGe.tar -c *
	$(DIVERGE_GZIP) divERGe.tar
	$(DIVERGE_MV) divERGe.tar.gz ..
	$(DIVERGE_RM) -rf pack/

