#include <diverge.h>
#include <diverge_Eigen3.hpp>

int main(int argc, char** argv) {
    diverge_init( &argc, &argv );
    diverge_compilation_status();
    diverge_model_t* model = diverge_model_init();
    // lattice & sites
    model->lattice[0][0] = model->lattice[1][0] = sin(M_PI/3.);
    model->lattice[0][1] = model->lattice[1][1] = cos(M_PI/3.);
    model->lattice[0][1] *= -1.0;
    model->lattice[2][2] = 1.0;
    Map<Mat3d> positions(model->positions[0]);
    Map<Mat3d> latt_vecs(model->lattice[0]);
    positions.col(0) = -1./3. * (latt_vecs.col(0) + latt_vecs.col(1));
    positions.col(1) =  1./3. * (latt_vecs.col(0) + latt_vecs.col(1));
    // other model parameters
    index_t nk = 24, nkf = 15;
    model->n_orb = 2;
    model->n_spin = 1;
    model->SU2 = true;
    model->nk[0] = model->nk[1] = nk;
    model->nkf[0] = model->nkf[1] = nkf;
    model->n_ibz_path = 4;
    model->ibz_path[1][0] = 0.5;
    model->ibz_path[2][0] = 2./3.;
    model->ibz_path[2][1] = 1./3.;
    strcpy( model->name, "graphene" );
    // hopping parameters
    double t1_dist = positions.col(0).norm(),
           t2_dist = latt_vecs.col(0).norm();
    double t1 = 1.0, t2 = 0.1;
    model->hop = diverge_mem_alloc_rs_hopping_t( 1024 );
    for (int Rx=-5; Rx<=5; ++Rx) for (int Ry=-5; Ry<=5; ++Ry)
    for (int o=0; o<2; ++o) for (int p=0; p<2; ++p) {
        assert( model->n_hop < 1024 );
        double dist = ( latt_vecs.col(0) * Rx + latt_vecs.col(1) * Ry +
                        positions.col(p) - positions.col(o) ).norm();
        if (fabs(dist - t1_dist) < 1e-5)
            model->hop[model->n_hop++] = rs_hopping_t{ {Rx,Ry,0}, o,p, 0,0, t1 };
        if (fabs(dist - t2_dist) < 1e-5)
            model->hop[model->n_hop++] = rs_hopping_t{ {Rx,Ry,0}, o,p, 0,0, t2 };
    }
    // interaction parameters
    double V1_dist = positions.col(0).norm(),
           V2_dist = latt_vecs.col(0).norm();
    double V0 = 3.6, V1 = 0.1, V2 = 0.05;
    model->vert = diverge_mem_alloc_rs_vertex_t( 1024 );
    model->vert[model->n_vert++] = rs_vertex_t{ 'D', {0,0,0}, 0,0, -1,0,0,0, V0 };
    model->vert[model->n_vert++] = rs_vertex_t{ 'D', {0,0,0}, 1,1, -1,0,0,0, V0 };
    for (int Rx=-5; Rx<=5; ++Rx) for (int Ry=-5; Ry<=5; ++Ry)
    for (int o=0; o<2; ++o) for (int p=0; p<2; ++p) {
        assert( model->n_vert < 1024 );
        double dist = ( latt_vecs.col(0) * Rx + latt_vecs.col(1) * Ry +
                        positions.col(p) - positions.col(o) ).norm();
        if (fabs(dist - V1_dist) < 1e-5)
            model->vert[model->n_vert++] = rs_vertex_t{ 'D', {Rx,Ry,0}, o,p, -1,0,0,0, V1 };
        if (fabs(dist - V2_dist) < 1e-5)
            model->vert[model->n_vert++] = rs_vertex_t{ 'D', {Rx,Ry,0}, o,p, -1,0,0,0, V2 };
    }
    // check!
    if (diverge_model_validate(model))
        diverge_mpi_exit(EXIT_FAILURE);
    // finalize model and save to disk
    double filling = 0.6,
           ffdist = 2.1;
    diverge_model_internals_common( model );
    diverge_model_set_filling( model, NULL, -1, filling );
    diverge_model_internals_tu( model, ffdist );
    diverge_model_output_conf_t cfg = diverge_model_output_conf_defaults_CPP();
    cfg.kc = true;
    cfg.npath = -1;
    cfg.kc_ibz_path = 1;
    diverge_model_to_file_finegrained( model, "graphene_model.dvg", &cfg );
    // flow
    double vmax = 0;
    double cmax[3] = {0};
    diverge_flow_step_t* flowstep = diverge_flow_step_init_any( model, "PCD" );
    diverge_euler_t eu = diverge_euler_defaults_CPP();
    eu.Lambda = 10.0;
    eu.dLambda = -1.0;
    eu.maxvert = 10.0;
    do {
        diverge_flow_step_euler( flowstep, eu.Lambda, eu.dLambda );
        diverge_flow_step_vertmax( flowstep, &vmax );
        diverge_flow_step_chanmax( flowstep, cmax );
        mpi_printf( "%.5e %.5e %.5e %.5e %.5e\n", eu.Lambda, cmax[0], cmax[1], cmax[2], vmax );
    } while (diverge_euler_next( &eu, vmax ));
    // post processing
    diverge_postprocess_conf_t out = diverge_postprocess_conf_defaults_CPP();
    out.tu_storing_relative = true;
    out.tu_storing_threshold = 0.8;
    out.tu_which_solver_mode = 's';
    diverge_postprocess_and_write_finegrained( flowstep, "graphene_post.dvg", &out );
    // cleanup
    diverge_flow_step_free( flowstep );
    diverge_model_free( model );
    diverge_finalize();
}
