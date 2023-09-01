#!/bin/bash

# for ctypesgen
PATH=$PATH:$HOME/.local/bin/

included_files_self=( $(grep -o "[a-z,_,/]*.h" src/diverge.h) )
included_files=( ${included_files_self[@]/#/src/} )
echo "-------------------------------------------------------------------------"
echo ctypesgen -ldivERGe -L. ${included_files[@]} -o util/diverge.py
echo "-------------------------------------------------------------------------"
ctypesgen -ldivERGe -L. ${included_files[@]} -o util/diverge.py

