#include <diverge.h>
#include <diverge_Eigen3.hpp>

int main(int argc, char** argv) {
    diverge_init( &argc, &argv );
    diverge_compilation_status();
    diverge_model_t* model = diverge_model_init();
    // lattice & sites
    model->lattice[0][0] = model->lattice[2][2] = 1.0;
    model->lattice[1][0] = cos(M_PI/3.);
    model->lattice[1][1] = sin(M_PI/3.);
    Map<Mat3d> positions(model->positions[0]);
    Map<Mat3d> latt_vecs(model->lattice[0]);
    positions.col(0) = -1./3. * (latt_vecs.col(0) + latt_vecs.col(1));
    positions.col(1) =  1./3. * (latt_vecs.col(0) + latt_vecs.col(1));
    // other model parameters
    index_t nk = 24, nkf = 15;
    model->n_orb = 3;
    model->nk[0] = model->nk[1] = nk;
    model->nkf[0] = model->nkf[1] = nkf;
    model->n_ibz_path = 4;
    model->ibz_path[1][0] = 0.5;
    model->ibz_path[2][0] = 2./3.;
    model->ibz_path[2][1] = 1./3.;
    strcpy( model->name, "hybrid-graphene" );
    // hopping parameters
    double t1_dist = positions.col(0).norm(),
           t2_dist = latt_vecs.col(0).norm();
    double t1 = 1.0, t2 = 0.0;
    model->hop = (rs_hopping_t*) calloc( 1024, sizeof(rs_hopping_t) );
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
    double V = 0.8;
    model->hop[model->n_hop++] = rs_hopping_t{ {0,0,0}, 2,0, 0,0, V };
    model->hop[model->n_hop++] = rs_hopping_t{ {0,0,0}, 0,2, 0,0, V };
    model->vert = (rs_vertex_t*) calloc( 1024, sizeof(rs_vertex_t) );
    // check!
    if (diverge_model_validate(model))
        diverge_mpi_exit(EXIT_FAILURE);
    diverge_model_internals_common( model );

    index_t nkkf[3] = {-nk*nkf, nk*nkf, 1};
    double* berry = diverge_fukui( model, NULL, -1, nkkf );
    index_t which_bands[3] = {1,0,2};
    double* qgt = diverge_qgt( model, NULL, -1, nkkf, which_bands, 1, 1 );
    double* qgt_duo = diverge_qgt( model, NULL, -1, nkkf, which_bands, 2, 1 );
    double* qgt_tri = diverge_qgt( model, NULL, -1, nkkf, which_bands, 3, 1 );

    double scale = kdimtot(model->nk, model->nkf);
    for (index_t k=0; k<kdimtot(model->nk, model->nkf); ++k)
        qgt[k] *= scale, qgt_duo[k] *= scale, qgt_tri[k] *= scale;

    index_t sz_berry = sizeof(double) * kdimtot(model->nk, model->nkf) * 3*model->n_orb,
            sz_qgt = sizeof(double) * kdimtot(model->nk, model->nkf);
    char* data = (char*)malloc( (model->nbytes_data = sz_berry + 3*sz_qgt) );
    memcpy( data, berry, sz_berry );
    memcpy( data + sz_berry + 0*sz_qgt, qgt, sz_qgt );
    memcpy( data + sz_berry + 1*sz_qgt, qgt_duo, sz_qgt );
    memcpy( data + sz_berry + 2*sz_qgt, qgt_tri, sz_qgt );
    model->data = data;
    free( berry );
    free( qgt );
    free( qgt_duo );
    free( qgt_tri );

    diverge_model_output_conf_t oc = diverge_model_output_conf_defaults_CPP();
    oc.kf = 1;
    diverge_model_to_file_finegrained( model, __FILE__ ".mod.dvg", &oc );
    diverge_model_free( model );
    diverge_finalize();
}
