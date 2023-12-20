CC := gcc
LDFLAGS += -L.. -Wl,-rpath=$(shell pwd)/.. -ldivERGe -lm -lfftw3 -lfftw3_mpi
CFLAGS += -I../src -Ofast -march=native -Wall -Wextra -pedantic -std=c11 -fopenmp -DUSE_MPI

-include Makefile.local
-include Makefile.inc

NAME := t2g_model.x

all: $(NAME)

%.x: %.c Makefile
	$(CC) $< -o $@ $(CFLAGS) $(LDFLAGS)

clean:
	@rm -f $(NAME)
