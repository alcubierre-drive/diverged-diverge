#!/bin/bash

# for ctypesgen
PATH=$PATH:$HOME/.local/bin/

included_files_self=( $(grep -o "^#include \"[a-z,_,/]*.h\"" src/diverge.h | sed 's|#include "||g' | sed 's|"||g') )
included_files=( ${included_files_self[@]/#/src/} )
echo "-------------------------------------------------------------------------"
echo ctypesgen -ldivERGe -L. ${included_files[@]} -o util/diverge.py
echo "-------------------------------------------------------------------------"
ctypesgen -ldivERGe -L. ${included_files[@]} -o util/diverge.py

