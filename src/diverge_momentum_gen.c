/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_internals_struct.h"
#include "diverge_momentum_gen.h"

static void cross(double A[3], const double B[3], const double C[3], double factor) {
    for (int i=0; i<3; ++i)
        A[i] = factor * (B[(i+1)%3] * C[(i+2)%3] - B[(i+2)%3] * C[(i+1)%3]);
}

static double dot(const double A[3], const double B[3]) {
    return A[0]*B[0] + A[1]*B[1] + A[2]*B[2];
}

void diverge_model_generate_mom_basis(double lattice[3][3], double kbasis[3][3]) {
    double helper[3];
    cross(helper,lattice[0],lattice[1],1.0);
    double VBZsign = dot(helper,lattice[2]);
    if (VBZsign < 0) {
        mpi_wrn_printf("left handed coordinate system detected and fixed\n");
        lattice[2][0] *= -1.,lattice[2][1] *= -1.,lattice[2][2] *= -1.;
        VBZsign *= -1;
    }
    double prefactor = 2.0 * M_PI / VBZsign;
    for (int i=0; i<3; ++i)
        cross( kbasis[i], lattice[(i+1)%3], lattice[(i+2)%3], prefactor );
}

static void fill_mesh(double* mesh, const index_t nk[3], double kbasis[3][3]) {
    const double dnk[3] = { nk[0], nk[1], nk[2] };
    for(index_t kx = 0; kx < nk[0]; ++kx)
    for(index_t ky = 0; ky < nk[1]; ++ky)
    for(index_t kz = 0; kz < nk[2]; ++kz) {
        index_t idx = 3 * IDX3(kx,ky,kz,nk[1],nk[2]);
        mesh[idx + 0] = kbasis[0][0] * (double)kx / dnk[0]
                      + kbasis[1][0] * (double)ky / dnk[1]
                      + kbasis[2][0] * (double)kz / dnk[2];
        mesh[idx + 1] = kbasis[0][1] * (double)kx / dnk[0]
                      + kbasis[1][1] * (double)ky / dnk[1]
                      + kbasis[2][1] * (double)kz / dnk[2];
        mesh[idx + 2] = kbasis[0][2] * (double)kx / dnk[0]
                      + kbasis[1][2] * (double)ky / dnk[1]
                      + kbasis[2][2] * (double)kz / dnk[2];
    }
}


void diverge_model_generate_meshes(double* kmesh, double* kfmesh,
        const index_t nk[3], const index_t nkf[3], double lattice[3][3]) {
    double kbasis[3][3];
    diverge_model_generate_mom_basis(lattice,kbasis);
    index_t nkff[3];
    for (int i=0; i<3; ++i) nkff[i] = nk[i] * nkf[i];

    fill_mesh(kmesh,nk,kbasis);
    fill_mesh(kfmesh,nkff,kbasis);
}


