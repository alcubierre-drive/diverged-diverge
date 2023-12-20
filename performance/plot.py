#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import glob
import matplotlib.pyplot as plt
import cmasher as cm
import matplotlib
matplotlib.rc('font',**dict(family='sans',size=9))
# matplotlib.rc('font',**dict(family='serif',size=9))

def data_entry( file ):
    system, compute = file.split('/')
    gpu = False
    if "GPU" in file:
        sysparam = [int(x) for x in system.split('_')[:-1]]
        sysparam.append( 1 )
        comparam = [1, int(compute[1:-4]), 64]
        gpu = True
    else:
        sysparam = [int(x) for x in system.split('_')]
        comparam = [int(x[1:]) for x in compute.split('.')[0].split('_')]
    time = -1
    with open(file, "r") as f:
        for l in f:
            if (l[0] != '#'):
                time = float(l[:-2])
                break

    return dict(nkc=sysparam[0]*sysparam[0], nkf=sysparam[1]*sysparam[1],
                nodes=comparam[0], tasks_per_node=comparam[1], cpus_per_task=comparam[2],
                chunksize=sysparam[2], time=time, gpu=gpu)

def sciform( num, prec=2 ):
    snum = ("%%.%de" % prec) % num
    nbase, npow = snum.split('e')
    nnpow = npow.replace('+','')
    if (nnpow == '0'):
        return nbase
    else:
        return nbase + '\\cdot 10^{' + nnpow.replace('0','') + '}'

data_all = [data_entry(f) for f in glob.glob( '*/*.dat' )]

data_cpu = []
data_gpu = []
for d in data_all:
    if d['gpu']:
        data_gpu.append(d)
    else:
        data_cpu.append(d)

def get_plot_data( data ):
    kpts = [d['nkc']*d['nkf'] for d in data]
    tasks = [d['nodes']*d['tasks_per_node'] for d in data]
    cpus = [d['cpus_per_task'] for d in data]
    times = [d['time'] for d in data]

    K = np.array(kpts)
    T = np.array(tasks)
    C = np.array(cpus)
    t = np.array(times)

    uK = np.unique(K)

    return K, T, C, t, uK

K, T, C, t, uK = get_plot_data( data_cpu )
gK, gT, gC, gt, guK = get_plot_data( data_gpu )
colors = cm.iceburn( np.linspace( 0.15, 0.85, uK.size ) )
# colors = cm.wildfire( np.linspace( 0.2, 0.8, uK.size ) )

fig, axs = plt.subplots( 1, 2, figsize=(5.90552*0.66,2.5), layout="constrained", sharey=True )
axs[0].set_title('CPU: AMD EPYC 7742', fontsize=9)
axs[1].set_title('GPU: NVIDIA A100', fontsize=9)

def label( k ):
    # return '$N_k = %s$' % sciform(k, 1)
    return f'$N_k = ({np.sqrt(k):.0f})^2$'

for i,k in enumerate(uK):
    sel_k = k == K
    sT = T[sel_k]
    sC = C[sel_k]
    st = t[sel_k] / 2

    uT = np.unique( sT )
    t_mean = [np.mean(st[u == sT]) for u in uT]
    t_std = [np.std(st[u == sT]) for u in uT]

    axs[0].plot( uT, t_mean, c=colors[i], label=label(k), marker='x', lw=0.8, markersize=4, mew=0.5 )
    axs[0].plot( uT, uT[0]/uT * t_mean[0], c=colors[i], lw=1.8, alpha=0.2 )

axs[0].set_xlabel( '# MPI ranks (×16 cores)' )
axs[0].set_ylabel( 'flow step time (s)' )
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].legend(frameon=False, fontsize=8, loc=(-0.01,-0.025))
axs[0].set_xlim(1,100)

for i,k in enumerate(guK):
    sel_k = k == gK
    sT = gT[sel_k]
    sC = gC[sel_k]
    st = gt[sel_k] / 2

    uT = np.unique( sT )
    t_mean = [np.mean(st[u == sT]) for u in uT]

    axs[1].plot( uT, t_mean, c=colors[i], label=label(k), marker='x', lw=0.8, markersize=4, mew=0.5 )
    axs[1].plot( uT, uT[0]/uT * t_mean[0], c=colors[i], lw=1.8, alpha=0.2 )
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xticks( [1,2,3,4] )
axs[1].set_xticklabels( ['1', '2', '3', '4'] )
axs[1].set_xlabel( 'number of GPUs', labelpad=5.3 )
axs[0].set_ylim( axs[1].get_ylim() )
axs[0].text( 0.03, 0.90, '$\\bf{a}$', transform=axs[0].transAxes, va='bottom', ha='left' )
axs[1].text( 0.97, 0.90, '$\\bf{b}$', transform=axs[1].transAxes, va='bottom', ha='right' )

fig.savefig( 'scaling.pdf' )
