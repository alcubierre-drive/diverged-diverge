#!/bin/bash

VERSION=0.5
SYMBOLS=DIVERGE_$VERSION
OUTFILE=libdivERGe.map

included_files_self=( $(grep "^#" src/diverge.h | grep -o "[a-z,_,/,A-Z,0-9]*.h") )
included_files=( ${included_files_self[@]/#/src/} )

exported_symbols=( $(ctags -o- --kinds-C=p ${included_files[@]} |\
    grep -v "^!" | awk '{print $1}' |\
    grep -v "diverge_patching_find_fs_pts$" |\
    grep -v "diverge_read_W90$" |\
    grep -v "shared_malloc_init$" |\
    grep -v "shared_malloc_finalize$") )
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

exported_unstable_symbols=$(cat << EOF
DIVERGE_UNSTABLE {
    global:
        batched_eigen;
        batched_eigen_r;
        batched_svd;
        batched_zgemd_r;
        eigen_solve;
        singular_value_decomp;
        eigensolution_sort;
        batched_gemm_overlapping;
        batched_gemm;
        batched_gemm_with_buffers;
        batched_gemm_vertex_loop;
        batched_gemm_loop_vertex;
        batched_gemm_cublas_init;
        batched_gemm_cublas_clear;
        batched_gemm_cublas_w_buffer_clear;
        batched_gemm_N_ll_num;
        gemm;
        batched_gemm_small_init;
        batched_gemm_small_destroy;
        batched_gemm_small_mode;
        batched_gemm_small;
};
EOF
)

echo "$exported_unstable_symbols" >> $OUTFILE
