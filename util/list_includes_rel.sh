#!/bin/sh
grep "#include" ../src/diverge.h | cut -d" " -f2 | cut -c 2- | rev | cut -c 2- | rev
