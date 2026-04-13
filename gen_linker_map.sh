#!/bin/bash

TAG_VERSION="$(git tag --list | tac | head -n-1 | sort -V | tail -n1)"
VERSION=${TAG_VERSION:1}
SYMBOLS=DIVERGE_$VERSION
OUTFILE=libdivERGe.map

echo $SYMBOLS

included_files_self=( $(grep "^#" src/diverge.h | grep -o "[a-z,_,/,A-Z,0-9]*.h") )
included_files=( ${included_files_self[@]/#/src/} )

misc_included_files_self=( $(grep "^#" src/diverge_misc.h | grep -o "[a-z,_,/,A-Z,0-9]*.h") )
misc_included_files=( ${misc_included_files_self[@]/#/src/} )

phonon_included_files_self=( $(grep "^#" src/diverge_phonons.h | grep -o "[a-z,_,/,A-Z,0-9]*.h") \
    diverge_phonons.h )
phonon_included_files=( ${phonon_included_files_self[@]/#/src/} )

exported_symbols=( $(ctags -o- --kinds-C=p ${included_files[@]} |\
    grep -v "^!" | grep -v VECTOR | awk '{print $1}' |\
    grep -v "shared_malloc_init$" |\
    grep -v "shared_malloc_finalize$") )

exported_cpp_symbols=$(cat <<EOF
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

exported_unstable_symbols=( $(ctags -o- --kinds-C=p ${misc_included_files[@]} ${phonon_included_files[@]} |\
    grep -v "^!" | grep -v VECTOR  | awk '{print $1}' |\
    grep -v "_dos_sort_doubles_$" |\
    grep -v "_dos_sort_doubles_indirect_$") \
    diverge_tu_vertex_reco
    diverge_tu_vertex_reco_ft
)

echo "DIVERGE_UNSTABLE {" >> $OUTFILE
echo "    global:" >> $OUTFILE
for sym in ${exported_unstable_symbols[@]}
do
    echo "        $sym;" >> $OUTFILE
done
echo "};" >> $OUTFILE
