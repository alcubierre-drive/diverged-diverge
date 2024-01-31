#include "hoppings_to_supercell.h"
#include "../diverge_Eigen3.hpp"
#include "init_internal_libs.h"
#include <vector>
#include <array>

using std::vector;
using std::array;

#include <iostream>

void rs_hopping_to_fractcell( diverge_model_t* m, const double* to_super_ ) {
    Map<const Mat3d> to_super_T(to_super_);
    Mat3d to_super = to_super_T.transpose();
    Map<Mat3d> latt_T(m->lattice[0]);

    Mat3d new_latt = to_super * latt_T.transpose();
    Mat3d inew_latt = new_latt.inverse().transpose();

    index_t limit_xyz = 2 * lround(to_super.array().abs().rowwise().sum().maxCoeff()) + 1;

    vector<array<index_t,3>> pmap;
    pmap.reserve( POW3(limit_xyz) );

    auto inbounds = []( double val )->bool{
        return (val > (0.0-DIVERGE_EPS_MESH)) && (val < (1.0-DIVERGE_EPS_MESH));
    };
    for (index_t i=-limit_xyz; i<=limit_xyz; ++i)
    for (index_t j=-limit_xyz; j<=limit_xyz; ++j)
    for (index_t k=-limit_xyz; k<=limit_xyz; ++k) {
        Vec3d sp = Map<Vec3d>(m->lattice[0]) * i + Map<Vec3d>(m->lattice[1]) * j + Map<Vec3d>(m->lattice[2]) * k;
        Vec3d isp = inew_latt * sp;
        if (inbounds(isp(0)) && inbounds(isp(1)) && inbounds(isp(2)))
            pmap.push_back( {i,j,k} );
    }

    vector<rs_hopping_t> newhop;
    newhop.reserve( pmap.size() * m->n_hop );

    #pragma omp parallel for num_threads(diverge_omp_num_threads()) schedule(dynamic)
    for (index_t h=0; h<m->n_hop; ++h) {
        rs_hopping_t* hop = m->hop + h;
        Vec3d ldist = hop->R[0] * latt_T.col(0) + hop->R[1] * latt_T.col(1) + hop->R[2] * latt_T.col(2);
        index_t Rmax = MAX( MAX(abs(hop->R[0]), abs(hop->R[1])), abs(hop->R[2]) );

        index_t search = MAX(Rmax, limit_xyz);

        for (uindex_t oo=0; oo<pmap.size(); ++oo) {
            const array<index_t,3>& o = pmap[oo];
            Vec3d ostart = o[0] * latt_T.col(0) + o[1] * latt_T.col(1) + o[2] * latt_T.col(2);
            rs_hopping_t new_hop;
            int found = 0;
            for (uindex_t pp=0; pp<pmap.size(); ++pp) {
                const array<index_t,3>& p = pmap[pp];
                Vec3d ostop = p[0] * latt_T.col(0) + p[1] * latt_T.col(1) + p[2] * latt_T.col(2);
                for (index_t i=-search; i<=search; ++i)
                for (index_t j=-search; j<=search; ++j)
                for (index_t k=-search; k<=search; ++k) {
                    Vec3d Ldist = (i * new_latt.row(0) + j * new_latt.row(1) + k * new_latt.row(2)).transpose() + ostop - ostart;
                    if ((Ldist - ldist).norm() < DIVERGE_EPS_MESH/10) {
                        new_hop.R[0] = j;
                        new_hop.R[1] = j;
                        new_hop.R[2] = k;
                        new_hop.o1 = hop->o1 + m->n_orb * oo;
                        new_hop.o2 = hop->o2 + m->n_orb * pp;
                        new_hop.s1 = hop->s1;
                        new_hop.s2 = hop->s2;
                        new_hop.t = hop->t;
                        found = 1;
                        goto found_elem;
                    }
                }
            }
            found_elem:
            if (found) {
                #pragma omp critical
                {
                    newhop.push_back( new_hop );
                    /* mpi_log_printf( "%li,%li,%li -> (%li,%li) : %li,%li,%li\n", */
                    /*         hop->R[0], hop->R[1], hop->R[2], */
                    /*         new_hop.o1, new_hop.o2, */
                    /*         new_hop.R[0], new_hop.R[1], new_hop.R[2] ); */
                }
            }
        }
    }

    // update positions
    vector<Vec3d> newpos;
    for (const array<index_t,3>& p: pmap)
    for (index_t i=0; i<m->n_orb; ++i)
        newpos.push_back( Map<Vec3d>( m->positions[i] ) + p[0] * latt_T.col(0) + p[1] * latt_T.col(1) + p[2] * latt_T.col(2) );
    for (uindex_t i=0; i<newpos.size(); ++i)
        Map<Vec3d>(m->positions[i]) = newpos[i];

    // update hoppings
    m->hop = (rs_hopping_t*)realloc(m->hop, sizeof(rs_hopping_t) * newhop.size());
    memcpy( m->hop, newhop.data(), sizeof(rs_hopping_t) * newhop.size() );
    m->n_hop = newhop.size();

    // set new #orbitals
    m->n_orb *= pmap.size();

    // set new lattice
    latt_T = new_latt.transpose();
}
