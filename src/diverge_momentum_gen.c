/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_internals_struct.h"
#include "diverge_momentum_gen.h"
#define LINALG_NO_LAPACKE
#include "misc/linalg.h"
LINALG_IMPLEMENT

void diverge_model_generate_mom_basis( double* lattice, double* kbasis ) {
    claV3d_t lat[3] = {claV3d_map(lattice), claV3d_map(lattice+3), claV3d_map(lattice+6)};
    double VBZsign = claV3d_dot( claV3d_cross(lat[0],lat[1]), lat[2] );
    if (VBZsign < 0) {
        mpi_wrn_printf("left handed coordinate system detected and fixed\n");
        lat[2] = claV3d_scale(lat[2], -1.0);
        VBZsign *= -1;
    }
    double prefactor = 2.0 * M_PI / VBZsign;
    for (int i=0; i<3; ++i) {
        claV3d_rmap(claV3d_scale(claV3d_cross(lat[(i+1)%3],lat[(i+2)%3]), prefactor), kbasis+3*i);
        claV3d_rmap(lat[i], lattice + 3*i);
    }
}

static void fill_mesh(double* mesh, const index_t nk[3], double kbasis[3][3]) {
    const double dnk[3] = { nk[0], nk[1], nk[2] };
    const claM3d_t G = claM3d_map(kbasis[0]);
    for (index_t kx=0; kx<nk[0]; ++kx)
    for (index_t ky=0; ky<nk[1]; ++ky)
    for (index_t kz=0; kz<nk[2]; ++kz) {
        const claV3d_t kiv = {{(double)kx/dnk[0], (double)ky/dnk[1], (double)kz/dnk[2]}};
        claV3d_rmap(claM3d_vecmat(kiv, G), mesh+3*IDX3(kx,ky,kz,nk[1],nk[2]));
    }
}


diverge_lattice_t diverge_model_generate_meshes(double* kmesh, double* kfmesh,
        const index_t nk[3], const index_t nkf[3], double lattice[3][3]) {
    diverge_lattice_t kbasis = {0};
    diverge_model_generate_mom_basis(lattice[0],kbasis.v[0]);
    index_t nkff[3];
    for (int i=0; i<3; ++i) nkff[i] = nk[i] * nkf[i];

    fill_mesh(kmesh,nk,kbasis.v);
    fill_mesh(kfmesh,nkff,kbasis.v);
    return kbasis;
}


