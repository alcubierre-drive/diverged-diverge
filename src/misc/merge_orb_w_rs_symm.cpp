#include "merge_orb_w_rs_symm.h"
#include "../diverge_Eigen3.hpp"

typedef Eigen::Matrix<double,3,3,Eigen::RowMajor> CMat3d;
typedef Eigen::Matrix<complex128_t,-1,-1,Eigen::RowMajor> CMatXcd;


void merge_rs_orb(diverge_model_t* model) {
    if (!model->orb_symmetries || model->n_sym == 0) return;
    complex128_t* orb_symmetries = (complex128_t*)model->orb_symmetries;
    double* rs_symmetries = (double*)model->rs_symmetries;
    index_t n_orb = model->n_orb;
    index_t n_spin = model->n_spin;
    index_t n_sym = model->n_sym;
    index_t n_mi = n_orb * n_spin;

    complex128_t* orb_cpy = (complex128_t*)malloc(n_mi*n_mi*n_sym*sizeof(complex128_t));
    memcpy(orb_cpy,orb_symmetries,n_mi*n_mi*n_sym*sizeof(complex128_t));
    memset((void*)orb_symmetries, 0, n_mi*n_mi*n_sym*sizeof(complex128_t));

    int periodic_iter[3];
    periodic_iter[0] = model->internals->periodic_direction[0] ? 3:0;
    periodic_iter[1] = model->internals->periodic_direction[1] ? 3:0;
    periodic_iter[2] = model->internals->periodic_direction[2] ? 3:0;

    Map<Vec3d> UC1(model->lattice[0]);
    Map<Vec3d> UC2(model->lattice[1]);
    Map<Vec3d> UC3(model->lattice[2]);

    for(index_t s = 0; s < n_sym; ++s)
    for(index_t o1 = 0; o1 < n_orb; ++o1) {
        Map<CMat3d> sym_rs(rs_symmetries+s*9);
        Map<Vec3d> from(model->positions[o1]);
        Vec3d to = sym_rs*from;
        for(index_t o2 = 0; o2 < n_orb; ++o2) {
            bool found = false;
            Map<Vec3d> ro2(model->positions[o2]);
            for(int in = -periodic_iter[0];in<=periodic_iter[0];in++)
            for(int im = -periodic_iter[1];im<=periodic_iter[1];im++)
            for(int iq = -periodic_iter[2];iq<=periodic_iter[2];iq++) {
                Vec3d trial = ro2+in*UC1+im*UC2+iq*UC3;
                if((trial-to).squaredNorm() < 1e-6) found = true;
            }
            if(found)
            for(index_t s1 = 0; s1 < n_spin; ++s1)
            for(index_t s2 = 0; s2 < n_spin; ++s2) {
                orb_symmetries[IDX5(s,s2,o2,s1,o1,n_spin,n_orb,n_spin,n_orb)] =
                    orb_cpy[IDX5(s,s2,o2,s1,o1,n_spin,n_orb,n_spin,n_orb)];
            }
        }
    }
    free(orb_cpy);
}

