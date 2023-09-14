#include <diverge.h>
#include <diverge_Eigen3.hpp>

#include <iostream>

const index_t nk = 120;
const double t = 1.0;
const double tp = 0.2;
const double U = 3.0;
const double V = 0.0;
const double V2 = 0.0;
const index_t np = 2;

int main(int argc, char** argv) {
    diverge_init( &argc, &argv );
    mpi_loglevel_set( 5 );
    diverge_compilation_status();
    diverge_model_t* m = diverge_model_init();

    m->lattice[0][0] = m->lattice[1][0] = sin(M_PI/3.);
    m->lattice[0][1] = m->lattice[1][1] = cos(M_PI/3.);
    m->lattice[0][1] *= -1.0;
    m->lattice[2][2] = 1.0;
    Map<Mat3d> pos(m->positions[0]);
    Map<Mat3d> lat(m->lattice[0]);
    pos.col(0) = -1./3. * (lat.col(0) + lat.col(1));
    pos.col(1) =  1./3. * (lat.col(0) + lat.col(1));
    m->n_orb = 2;
    m->n_spin = 1;
    m->SU2 = 1;
    m->nk[0] = m->nk[1] = nk;
    m->nkf[0] = m->nkf[1] = 1;
    m->n_ibz_path = 4;
    m->ibz_path[1][0] = 0.5;
    m->ibz_path[2][0] = 2./3.;
    m->ibz_path[2][1] = 1./3.;
    strcpy( m->name, __FILE__ );

    double t_norm = pos.col(0).norm();
    double tp_norm = 1.0;

    m->hop = (rs_hopping_t*)calloc(sizeof(rs_hopping_t), 1024);
    m->vert = (rs_vertex_t*)calloc(sizeof(rs_vertex_t), 1024);
    m->vert[m->n_vert++] = rs_vertex_t{ 'D', {0,0,0}, 0,0, -1,0,0,0, U };
    m->vert[m->n_vert++] = rs_vertex_t{ 'D', {0,0,0}, 1,1, -1,0,0,0, U };
    for (int Rx=-5; Rx<=5; ++Rx)
    for (int Ry=-5; Ry<=5; ++Ry)
    for (int o=0; o<2; ++o)
    for (int p=0; p<2; ++p) {
        double dist = (lat.col(0) * Rx + lat.col(1) * Ry + pos.col(p) - pos.col(o)).norm();
        if (std::abs(dist - t_norm) < 1e-5) {
            m->hop[m->n_hop++] = rs_hopping_t{ {Rx,Ry,0}, o,p, 0,0, t };
            m->vert[m->n_vert++] = rs_vertex_t{ 'D', {Rx,Ry,0}, o,p, -1,0,0,0, V };
        }
        if (std::abs(dist - tp_norm) < 1e-5) {
            m->hop[m->n_hop++] = rs_hopping_t{ {Rx,Ry,0}, o,p, 0,0, tp };
            m->vert[m->n_vert++] = rs_vertex_t{ 'D', {Rx,Ry,0}, o,p, -1,0,0,0, V2 };
        }
    }

    m->orb_symmetries = (complex128_t*)malloc(sizeof(complex128_t)*12*4);
    site_descr_t sites[2];
    for (int s=0; s<2; ++s) {
        sites[s].amplitude[0] = 1.0;
        sites[s].function[0] = orb_s;
        sites[s].n_functions = 1;
    }
    sym_op_t sym;
    Map<Vec3d> sym_nvec(sym.normal_vector);
    sym_nvec = Vec3d(0,0,1);
    sym.type = 'R';
    for (int i=0; i<6; ++i) {
        sym.angle = 60*i;
        diverge_generate_symm_trafo( 1, sites, 2, &sym, 1, m->rs_symmetries[m->n_sym][0],
                m->orb_symmetries+m->n_sym*4 );
        m->n_sym++;
    }
    sym.type = 'M';
    sym_nvec *= 0.0;
    for (int i=0; i<6; ++i) {
        sym_nvec.head(2) = Rot2d(M_PI/6.0 * i).toRotationMatrix() * Vec2d(1,0);
        diverge_generate_symm_trafo( 1, sites, 2, &sym, 1, m->rs_symmetries[m->n_sym][0],
                m->orb_symmetries+m->n_sym*4 );
        m->n_sym++;
    }

    if (diverge_model_validate(m))
        return EXIT_FAILURE;

    diverge_model_internals_common(m);
    // in order to initialize symmetries correctly
    diverge_symmetrize_2pt_fine( m, diverge_model_internals_get_H(m), NULL );
    diverge_model_set_filling( m, NULL, -1, 0.7 );
    diverge_model_internals_patch( m, np );
    char* md5 = diverge_model_to_file(m, __FILE__ ".mod.dvg", 1);
    mpi_usr_printf( "md5sum: %s\n", md5 );

    diverge_flow_step_t* s = diverge_flow_step_init( m, "patch", "PCD" );
    diverge_euler_t eu = diverge_euler_defaults_CPP();
    eu.Lambda = 10.0;
    eu.dLambda = -1.0;
    eu.maxvert = 10.0;
    /* eu.maxiter = 10; */
    eu.dLambda_fac = 1.0;
    eu.dLambda_fac_scale = 0.1;
    double maxvert = 0;
    do {
        diverge_flow_step_euler( s, eu.Lambda, eu.dLambda );
        diverge_flow_step_vertmax( s, &maxvert );
        printf( "%.5e %.5e\n", eu.Lambda, maxvert );
    } while (diverge_euler_next( &eu, maxvert ));

    diverge_postprocess_and_write( s, NULL, __FILE__ ".out.dvg" );

    diverge_flow_step_free( s );
    diverge_model_free( m );
    diverge_finalize();
}
