CC := gcc
LDFLAGS += -L.. -Wl,-rpath=$(shell pwd)/.. -ldivERGe -lm -lfftw3 -lfftw3_mpi
CFLAGS += -I../src -Ofast -march=native -Wall -Wextra -pedantic -std=c11 -fopenmp -DUSE_MPI

-include Makefile.local

.PHONY: Makefile Makefile.local

t2g_model.x: t2g_model.c Makefile Makefile.local
	$(CC) $< -o $@ $(CFLAGS) $(LDFLAGS)
clean:
	@rm -f t2g_model.x
