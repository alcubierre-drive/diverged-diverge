#!/bin/bash

#SBATCH --job-name=diverge
#SBATCH --time=1200
#SBATCH --account=enhancerg
#SBATCH --mail-type=ALL
#SBATCH --mail-user=klebl@physik.rwth-aachen.de
#SBATCH --array=0-0

### GPU
### #SBATCH --partition=dc-gpu

### CPU
#SBATCH --partition=dc-cpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4

# SET 1
nodes=( 1 2 4 8 )
executable="./t2g_model.x 16 100 64"
prefix="16_100_64"

# SET 2
nodes=( 4 8 )
executable="./t2g_model.x 24 200 128"
prefix="24_200_128"

# SET 3
nodes=( 8 16 )
executable="./t2g_model.x 32 400 256"
prefix="32_400_256"

tasks_per_node=( 2 4 )
cpus=( 16 32 64 )

cmds=()

mkdir -p $prefix

function run_missing() {
missing_config_files="$(wc -l *.dat | grep "^\s*1" | rev | cut -d" " -f1 | rev)"
missing_configs=( )
for cfg in ${missing_configs[@]}; do
    bifs="$IFS"
    IFS=","
    ary=( $cfg )
    IFS="$bifs"

    nodes=${ary[0]}
    tasks=${ary[1]}
    cpus=${ary[2]}

    export OMP_NUM_THREADS=${cpus}
    export DIVERGE_OMP_NUM_THREADS=${OMP_NUM_THREADS}
    export DIVERGE_OMP_IMPLICIT_NUM_THREADS=${OMP_NUM_THREADS}

    fname=$(printf "$prefix/n%02d_t%02d_c%02d.dat" ${nodes} ${tasks} ${cpus})
    cmd="srun \
        --nodes=${nodes} \
        --ntasks-per-node=${tasks} \
        --cpus-per-task=${cpus} \
        --account=enhancerg \
        --time=600 \
        --partition=dc-cpu \
        $executable"
    cmds[${#cmds[@]}]="$cmd"
    echo "# $cmd" > $fname
    $cmd >> $fname &
done
wait $(jobs -p)
}

# run_missing
# exit 0

function run_all() {
for ((n=0; n<${#nodes[@]}; n++)); do
for ((t=0; t<${#tasks_per_node[@]}; t++)); do
for ((c=0; c<${#cpus[@]}; c++)); do

    export OMP_NUM_THREADS=${cpus[$c]}
    export DIVERGE_OMP_NUM_THREADS=${OMP_NUM_THREADS}
    export DIVERGE_OMP_IMPLICIT_NUM_THREADS=${OMP_NUM_THREADS}

    fname=$(printf "$prefix/n%02d_t%02d_c%02d.dat" ${nodes[$n]} ${tasks_per_node[$t]} ${cpus[$c]})
    cmd="srun \
        --nodes=${nodes[$n]} \
        --ntasks-per-node=${tasks_per_node[$t]} \
        --cpus-per-task=${cpus[$c]} \
        --account=enhancerg \
        --time=600 \
        --partition=dc-cpu \
        $executable"
    cmds[${#cmds[@]}]="$cmd"
    echo "$cmd"
    echo "# $cmd" > $fname
    $cmd >> $fname &
done
done
done

wait $(jobs -p)
}

run_all
exit 0

srun --cpus-per-task=64 --nodes=1 ./t2g_model.x 16 100 256

if true; then
# CPU:
# srun --cpus-per-task=16 --threads-per-core=1 ./t2g_model.x 32 200
srun --cpus-per-task=64 ./t2g_model.x 32 400 256

# other CPU:
# export OMP_NUM_THREADS=32
# export DIVERGE_OMP_NUM_THREADS=${OMP_NUM_THREADS}
# export DIVERGE_OMP_IMPLICIT_NUM_THREADS=${OMP_NUM_THREADS}
# 
# srun --cpus-per-task=32 --threads-per-core=2 ./t2g_model.x 16 200
else
# GPU:
LD_PRELOAD=./libdivERGe.so ./t2g_model.x 32 400 256
fi
