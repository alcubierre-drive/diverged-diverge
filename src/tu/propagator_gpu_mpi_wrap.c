#include "propagator_details.h"
#include "propagator_gpu_mpi.h"

#ifndef USE_CUDA

void S_loop_static_red_gpu_mpi( tu_prop_t* t, complex128_t* L, bool pp ) { (void)t; (void)L; (void)pp; }

#else // USE_CUDA

void S_loop_static_red_gpu_mpi( tu_prop_t* t, complex128_t* L, bool pp ) {
    const index_t* vnk = t->model->nk;
    const index_t* vnkf = t->model->nkf;
    index_t dims[] = {(vnk[0]*vnkf[0]), (vnk[1]*vnkf[1]), (vnk[2]*vnkf[2])};

    index_t L_orbff_spin = POW2(t->n_orbff*POW2(t->n_spin));
    memset( (void*)L, 0, sizeof(complex128_t) * t->nkibz*L_orbff_spin );

    if (!t->has_tasks_off_saved) {
        t->has_tasks_off_saved = true;
        memcpy( t->tasks_prop, t->cuda_descr._tasks_prop, sizeof(t->tasks_prop) );
        memcpy( t->off_prop, t->cuda_descr._off_prop, sizeof(t->off_prop) );
        for (int i=0; i<tu_gpu_info().tu_num_of_devices; ++i) {
            index_t* tasks_count = diverge_mpi_distribute( t->tasks_prop[i] );
            index_t* tasks_displ = tasks_count + diverge_mpi_comm_size();

            t->off_prop_mpi[i] = t->off_prop[i] + tasks_displ[diverge_mpi_comm_rank()];
            t->tasks_prop_mpi[i] = tasks_count[diverge_mpi_comm_rank()];
            free( tasks_count );
        }
    }

    if (!t->c2fmap_full) {
        t->c2fmap_full_sz = t->nkibz;
        t->c2fmap_full = calloc( t->c2fmap_full_sz, sizeof*t->c2fmap_full );
        if (t->tu_data->symm->use_symmetries) {
            for (index_t i=0; i<t->nkibz; ++i)
                t->c2fmap_full[i] = kidxc2f(t->tu_data->symm->idx_ibz_in_fullmesh[i], vnk, vnkf);
        } else {
            for (index_t i=0; i<t->nk; ++i)
                t->c2fmap_full[i] = kidxc2f(i, vnk, vnkf);
        }
    }

    // set the MPI offsets
    memcpy( t->cuda_descr._tasks_prop, t->tasks_prop_mpi, sizeof(t->tasks_prop_mpi) );
    memcpy( t->cuda_descr._off_prop, t->off_prop_mpi, sizeof(t->off_prop_mpi) );

    // do the calc
    if (pp) {
        if (!t->gpu_mpi_dist_pp) {
            t->gpu_mpi_dist_pp = tu_prop_gpu_mpi_dist_init( t->my_nk, t->my_nk_off, t->nkibz,
                    L_orbff_spin, &t->gpu_mpi_dist_recv_pack, &t->gpu_mpi_dist_recv_pack_sz );
        }
        pp_bubble_momenta_handler_fft_red( &t->cuda_descr, L,
                t->GF_fft_plus, t->GF_fft_conj_plus, t->n_orb, t->n_orbff, t->n_spin, t->nktot,
                t->nkibz, t->c2fmap_full, dims, t->mi_to_ofrom, t->mi_to_oto, t->mi_to_R,
                t->tu_data->unique_mi_l, t->tu_data->unique_mi_r,
                t->tu_data->unique_mi_l_sz, tu_prop_gpu_mpi_dist_cuda_kernel_bitary(t->gpu_mpi_dist_pp) );
    } else {
        if (!t->gpu_mpi_dist_ph) {
            t->gpu_mpi_dist_ph = tu_prop_gpu_mpi_dist_init( t->my_nk, t->my_nk_off, t->nkibz,
                    L_orbff_spin, &t->gpu_mpi_dist_recv_pack, &t->gpu_mpi_dist_recv_pack_sz );
        }
        ph_bubble_momenta_handler_fft_red( &t->cuda_descr, L, t->GF_fft_plus,
                t->GF_fft_conj_plus, t->n_orb, t->n_orbff, t->n_spin, t->nktot,
                t->nkibz, t->c2fmap_full, dims, t->mi_to_ofrom, t->mi_to_oto, t->mi_to_R,
                t->tu_data->unique_mi_l, t->tu_data->unique_mi_r,
                t->tu_data->unique_mi_l_sz, tu_prop_gpu_mpi_dist_cuda_kernel_bitary(t->gpu_mpi_dist_ph) );
    }

    // reset the offsets
    memcpy( t->cuda_descr._tasks_prop, t->tasks_prop, sizeof(t->tasks_prop) );
    memcpy( t->cuda_descr._off_prop, t->off_prop, sizeof(t->off_prop) );

    tu_prop_gpu_mpi_dist_comm( pp ? t->gpu_mpi_dist_pp : t->gpu_mpi_dist_ph, L );
}

#endif // USE_CUDA
