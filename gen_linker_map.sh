#!/bin/bash

VERSION=0.3
SYMBOLS=DIVERGE_$VERSION
OUTFILE=libdivERGe.map

included_files_self=( $(grep "^#" src/diverge.h | grep -o "[a-z,_,/,A-Z,0-9]*.h") )
included_files=( ${included_files_self[@]/#/src/} )

exported_symbols=( $(ctags -o- --kinds-C=p ${included_files[@]} |\
    grep -v "^!" | awk '{print $1}' |\
    grep -v "diverge_patching_find_fs_pts$" |\
    grep -v "diverge_read_W90$" ) )
exported_cpp_symbols=$(cat <<EOF
        extern "C++" {
            "diverge_patching_find_fs_pts(diverge_model_t*, double const*, long, long, long)";
            "diverge_read_W90(char const*, long)";
        };
EOF
)

echo "$SYMBOLS {
    global:" > $OUTFILE
for sym in ${exported_symbols[@]}
do
    echo "        $sym;" >> $OUTFILE
done
echo "$exported_cpp_symbols" >> $OUTFILE
echo "local:
        *;
};" >> $OUTFILE
