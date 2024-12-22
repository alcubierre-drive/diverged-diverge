#!/bin/bash

TAG_VERSION="$(git tag --list "v*.*" | tail -n1)"
VERSION=${TAG_VERSION:1}
SYMBOLS=DIVERGE_$VERSION
OUTFILE=libdivERGe.map

included_files_self=( $(grep "^#" src/diverge.h | grep -o "[a-z,_,/,A-Z,0-9]*.h") )
included_files=( ${included_files_self[@]/#/src/} )

misc_included_files_self=( $(grep "^#" src/diverge_misc.h | grep -o "[a-z,_,/,A-Z,0-9]*.h") )
misc_included_files=( ${misc_included_files_self[@]/#/src/} )

exported_symbols=( $(ctags -o- --kinds-C=p ${included_files[@]} |\
    grep -v "^!" | awk '{print $1}' |\
    grep -v "diverge_patching_find_fs_pts$" |\
    grep -v "shared_malloc_init$" |\
    grep -v "shared_malloc_finalize$") )

exported_cpp_symbols=$(cat <<EOF
        extern "C++" {
            "diverge_patching_find_fs_pts(diverge_model_t*, double const*, long, long, long)";
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

exported_unstable_symbols=( $(ctags -o- --kinds-C=p ${misc_included_files[@]} |\
    grep -v "^!" | awk '{print $1}') )

echo "DIVERGE_UNSTABLE {" >> $OUTFILE
echo "    global:" >> $OUTFILE
for sym in ${exported_unstable_symbols[@]}
do
    echo "        $sym;" >> $OUTFILE
done
echo "};" >> $OUTFILE
