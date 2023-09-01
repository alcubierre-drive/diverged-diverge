CC := gcc
LD := gcc
CXX := g++

CXXFLAGS = $(PREFLAGS) -std=c++17 -fopenmp -MMD -Wall -Wextra -pedantic $(DEFINES) $(INCLUDES)
CFLAGS = $(PREFLAGS) -std=c11 -fopenmp -MMD -Wall -Wextra -pedantic $(DEFINES) $(INCLUDES)
LDFLAGS = $(LIBS)

PREFLAGS = 
INCLUDES = 
LIBS = -ldivERGe -lm -lstdc++
DEFINES = 

SRC_C := $(shell find ./ -name '*.c')
SRC_CPP := $(shell find ./ -name '*.cpp')

BIN_C := $(patsubst %.c,%.x,$(SRC_C))
BIN_CPP := $(patsubst %.cpp,%.x,$(SRC_CPP))

DEP_C := $(patsubst %.c,%.c.d,$(SRC_C))
DEP_CPP := $(patsubst %.cpp,%.cpp.d,$(SRC_CPP))

-include Makefile.inc
-include Makefile.local

.PHONY: all clean

all: $(BIN_C) $(BIN_CPP)

-include $(DEP_C) $(DEP_CPP)

%.x: %.c Makefile
	$(CC) $< -o $@ $(CFLAGS) $(LDFLAGS)

%.x: %.cpp Makefile
	$(CXX) $< -o $@ $(CXXFLAGS) $(LDFLAGS)

clean:
	-@$(RM) -rf $(shell find . -name '*.x') $(shell find . -name '*.d')
