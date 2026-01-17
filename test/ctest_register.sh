#!/bin/bash

registry=( $(find . ! -name ctest_register.c.o -name '*.o' \
    -exec nm --format=just-symbols {} \; | grep ct__register__ | sort) )

echo "#include \"ctest_register.h\""
echo "#ifdef CT_SKIP_INITIALIZERS"
for ((i=0; i<${#registry[@]}; i++)); do
    echo "void ${registry[$i]}(void);"
done
echo "#endif"

echo "void ct__register__all(void) {"
echo "#ifdef CT_SKIP_INITIALIZERS"
for ((i=0; i<${#registry[@]}; i++)); do
    echo "${registry[$i]}();"
done
echo "#endif"
echo "}"

