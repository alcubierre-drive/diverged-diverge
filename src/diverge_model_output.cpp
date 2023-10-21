/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "diverge_model_output.h"
#include "diverge_internals_struct.h"
#include "diverge_model.h"
#include "misc/number_repr_checks.h"
#include "misc/mpi_functions.h"
#include "misc/kmesh_to_bands.h"

#include "misc/eigen.h"
#include "misc/md5.h"
#include "misc/version.h"

#include "diverge_momentum_gen.h"
#include "diverge_Eigen3.hpp"

static void generate_bandstructure( diverge_model_t* mod, double* buf, index_t npath );
static index_t h_diverge_model_nbandstructure = DIVERGE_MODEL_NBANDSTRUCTURE;
static void* kmesh_to_bands_serialize( diverge_model_t* mod, bool crs, index_t* psize );

void diverge_model_output_set_npath( int npath ) {
    h_diverge_model_nbandstructure = npath;
}

char* diverge_model_to_file( diverge_model_t* mod, const char* fname ) {
    return diverge_model_to_file_finegrained( mod, fname, NULL );
}

char* diverge_model_to_file_finegrained( diverge_model_t* mod, const char* fname,
        const diverge_model_output_conf_t* cfg ) {

    diverge_model_output_conf_t dcfg = diverge_model_output_conf_defaults_CPP();
    if (!cfg) cfg = &dcfg;

    void* k2b_serial = NULL;
    index_t k2b_serial_sz = 0;
    void* kf2b_serial = NULL;
    index_t kf2b_serial_sz = 0;
    if (cfg->kc_ibz_path)
        k2b_serial = kmesh_to_bands_serialize( mod, true, &k2b_serial_sz );
    if (cfg->kf_ibz_path)
        kf2b_serial = kmesh_to_bands_serialize( mod, false, &kf2b_serial_sz );

    mpi_log_printf( "output model to %s\n", fname );
    char* md5sum = NULL;
    uindex_t lmd5sum = 0;
    if (diverge_mpi_comm_rank() == 0) {
        // allocate the header
        char* buf = (char*)calloc(sizeof(index_t),128);

        // print the magic number
        index_t* ibuf = (index_t*)buf;
        index_t displ = sizeof(index_t)*128;
        int idx = 0;
        ibuf[idx++] = DIVERGE_MODEL_MAGIC_NUMBER;

        #define ADD_ARRAY_TO_IBUF( size ) { \
        ibuf[idx++] = displ; \
        ibuf[idx] = size; \
        displ += ibuf[idx++]; \
        }

        // name
        ADD_ARRAY_TO_IBUF( sizeof(char)*MAX_NAME_LENGTH );

        ibuf[idx++] = mod->internals->dim;
        ibuf[idx++] = mod->nk[0];
        ibuf[idx++] = mod->nk[1];
        ibuf[idx++] = mod->nk[2];
        ibuf[idx++] = mod->nkf[0];
        ibuf[idx++] = mod->nkf[1];
        ibuf[idx++] = mod->nkf[2];

        // patching
        if (mod->patching != NULL) {
            ibuf[idx++] = mod->patching->n_patches;

            ADD_ARRAY_TO_IBUF( sizeof(index_t)*ibuf[10] );
            ADD_ARRAY_TO_IBUF( sizeof(double)*ibuf[10] );
            ADD_ARRAY_TO_IBUF( sizeof(index_t)*ibuf[10] );
            ADD_ARRAY_TO_IBUF( sizeof(index_t)*ibuf[10] );

            ADD_ARRAY_TO_IBUF( sizeof(index_t) *
                (mod->patching->p_displ[mod->patching->n_patches-1] +
                 mod->patching->p_count[mod->patching->n_patches-1]) );
            ADD_ARRAY_TO_IBUF( sizeof(index_t) *
                (mod->patching->p_displ[mod->patching->n_patches-1] +
                 mod->patching->p_count[mod->patching->n_patches-1]) );
        } else {
            // set to displ if NULL
            ibuf[11] = ibuf[13] = ibuf[15] = ibuf[17] = ibuf[19] = ibuf[21] = displ;
            idx = 23;
        }

        // ibz path
        ADD_ARRAY_TO_IBUF(sizeof(double)*3*mod->n_ibz_path);

        ibuf[idx++] = mod->n_orb;

        // lattice as double[3][3]
        double* dbuf = (double*)buf;
        for (int ll=0; ll<9; ++ll)
            dbuf[idx++] = ((double*)mod->lattice)[ll];

        // positions
        ADD_ARRAY_TO_IBUF(sizeof(double)*3*mod->n_orb);

        ibuf[idx++] = mod->n_sym;

        // orbital symmetries
        ADD_ARRAY_TO_IBUF(sizeof(complex128_t)*mod->n_orb*mod->n_orb*mod->n_spin*mod->n_spin*mod->n_sym);
        // realspace symmetries
        ADD_ARRAY_TO_IBUF(sizeof(double)*9*mod->n_sym);

        ibuf[idx++] = mod->n_hop;
        // realspace hoppings
        ADD_ARRAY_TO_IBUF(sizeof(rs_hopping_t)*mod->n_hop);

        ibuf[idx++] = mod->SU2;
        ibuf[idx++] = mod->n_spin;

        ibuf[idx++] = mod->n_vert;
        // realspace vertex
        ADD_ARRAY_TO_IBUF(sizeof(rs_vertex_t)*mod->n_vert);

        ibuf[idx++] = mod->n_tu_ff;
        // formfactors
        ADD_ARRAY_TO_IBUF(sizeof(tu_formfactor_t)*mod->n_tu_ff);

        ibuf[idx++] = mod->n_vert_chan[0];
        ibuf[idx++] = mod->n_vert_chan[1];
        ibuf[idx++] = mod->n_vert_chan[2];

        // additional data
        ADD_ARRAY_TO_IBUF(mod->nbytes_data);

        // reciprocal lattice as double[3][3]
        double Gvec[3][3];
        diverge_model_generate_mom_basis(mod->lattice, Gvec);
        for (int ll=0; ll<9; ++ll)
            dbuf[idx++] = ((double*)Gvec)[ll];

        if (cfg->kc) {
            idx = 80;
            ADD_ARRAY_TO_IBUF(sizeof(double)*3*kdim(mod->nk));
        } else {
            ibuf[80] = displ;
        }
        if (cfg->kf) {
            idx = 82;
            ADD_ARRAY_TO_IBUF(sizeof(double)*3*kdimtot(mod->nk, mod->nkf));
        } else {
            ibuf[82] = displ;
        }
        if (cfg->H) {
            idx = 84;
            ADD_ARRAY_TO_IBUF(sizeof(complex128_t)*POW2(mod->n_orb*mod->n_spin)*kdimtot(mod->nk, mod->nkf));
        } else {
            ibuf[84] = displ;
        }
        if (cfg->U) {
            idx = 86;
            ADD_ARRAY_TO_IBUF(sizeof(complex128_t)*POW2(mod->n_orb*mod->n_spin)*kdimtot(mod->nk, mod->nkf));
        } else {
            ibuf[86] = displ;
        }
        if (cfg->E) {
            idx = 88;
            ADD_ARRAY_TO_IBUF(sizeof(double)*mod->n_orb*mod->n_spin*kdimtot(mod->nk, mod->nkf));
        } else {
            ibuf[88] = displ;
        }
        if (cfg->kc_ibz_path) {
            idx = 90;
            ADD_ARRAY_TO_IBUF(k2b_serial_sz);
        } else {
            ibuf[90] = displ;
        }
        if (cfg->kf_ibz_path) {
            idx = 92;
            ADD_ARRAY_TO_IBUF(kf2b_serial_sz);
        } else {
            ibuf[92] = displ;
        }
        idx = 94;

        index_t *band_indices = NULL, n_band_indices = 0, *n_band_segments = NULL;
        bool use_band_indices = false;
        bool include_bandstructure = false;
        // precedence behavior: using config value only if it is *not* zero:
        int npath = cfg->npath == 0 ? h_diverge_model_nbandstructure : cfg->npath;
        ibuf[idx++] = npath; // 94

        idx = 100;
        if (mod->n_ibz_path > 0) {
            // bandstructure, including the k coordinates *after* the bands
            if (npath == -1 && mod->internals->has_common_internals) {
                mpi_vrb_printf( "using fine kmesh data for band structure\n" );
                n_band_segments = diverge_kmesh_to_bands( mod, &band_indices, &n_band_indices );
                if (band_indices != NULL && n_band_indices > 0) {
                    use_band_indices = true;
                    include_bandstructure = true;
                    ADD_ARRAY_TO_IBUF( sizeof(double) * (mod->n_orb*mod->n_spin + 3) * n_band_indices );
                } else {
                    mpi_err_printf( "could not find band structure indices, skipping\n" );
                    ADD_ARRAY_TO_IBUF( 0 );
                }
            } else if (npath < -1) {
                mpi_wrn_printf( "invalid value %li for npath\n", npath );
                ADD_ARRAY_TO_IBUF( 0 );
            } else {
                include_bandstructure = true;
                ADD_ARRAY_TO_IBUF( sizeof(double) * (mod->n_orb*mod->n_spin + 3) *
                        ((mod->n_ibz_path-1)*npath + 1) );
            }
        } else {
            ADD_ARRAY_TO_IBUF( 0 );
        }

        // save struct offsets and sizes
        ibuf[102] = offsetof(rs_hopping_t, R);
        ibuf[103] = offsetof(rs_hopping_t, o1);
        ibuf[104] = offsetof(rs_hopping_t, o2);
        ibuf[105] = offsetof(rs_hopping_t, s1);
        ibuf[106] = offsetof(rs_hopping_t, s2);
        ibuf[107] = offsetof(rs_hopping_t, t);
        ibuf[108] = sizeof(rs_hopping_t);
        ibuf[109] = offsetof(rs_vertex_t, chan);
        ibuf[110] = offsetof(rs_vertex_t, R);
        ibuf[111] = offsetof(rs_vertex_t, o1);
        ibuf[112] = offsetof(rs_vertex_t, o2);
        ibuf[113] = offsetof(rs_vertex_t, s1);
        ibuf[114] = offsetof(rs_vertex_t, s2);
        ibuf[115] = offsetof(rs_vertex_t, s3);
        ibuf[116] = offsetof(rs_vertex_t, s4);
        ibuf[117] = offsetof(rs_vertex_t, V);
        ibuf[118] = sizeof(rs_vertex_t);
        ibuf[119] = offsetof(tu_formfactor_t, R);
        ibuf[120] = offsetof(tu_formfactor_t, ofrom);
        ibuf[121] = offsetof(tu_formfactor_t, oto);
        ibuf[122] = offsetof(tu_formfactor_t, d);
        ibuf[123] = offsetof(tu_formfactor_t, ffidx);
        ibuf[124] = sizeof(tu_formfactor_t);

        // file format version
        idx = 125;
        const char* version = tag_version();
        ADD_ARRAY_TO_IBUF( strlen(version) );

        // information on whether numerics is OK
        uindex_t checkmask = 0;
        uint8_t one = 1;
        uindex_t firstbit = checkmask | (one << 7);
        if (!check_iec_float())
            checkmask |= firstbit >> 0;
        if (!check_iec_double())
            checkmask |= firstbit >> 1;
        if (!check_little_endian())
            checkmask |= firstbit >> 2;
        if (!check_8bit_char())
            checkmask |= firstbit >> 3;
        uindex_t* ubuf = (uindex_t*)ibuf;
        ubuf[127] = checkmask;

        // realloc all the memory needed
        buf = (char*)realloc(buf, ibuf[125] + ibuf[126]);
        ibuf = (index_t*)buf;
        dbuf = (double*)buf;

        // copy over the arrays
        #define COPY_ARY_TO_BUF( ary ) { \
        memcpy( buf + ibuf[idx], ary, ibuf[idx+1] ); \
        idx+=2; \
        }
        idx = 1;
        COPY_ARY_TO_BUF( mod->name );
        if (mod->patching != NULL) {
            idx = 11;
            COPY_ARY_TO_BUF( mod->patching->patches );
            COPY_ARY_TO_BUF( mod->patching->weights );
            COPY_ARY_TO_BUF( mod->patching->p_count );
            COPY_ARY_TO_BUF( mod->patching->p_displ );
            COPY_ARY_TO_BUF( mod->patching->p_map );
            COPY_ARY_TO_BUF( mod->patching->p_weights );
        }

        idx = 23;
        COPY_ARY_TO_BUF( mod->ibz_path )

        idx = 35;
        COPY_ARY_TO_BUF( mod->positions );

        idx = 38;
        COPY_ARY_TO_BUF( mod->orb_symmetries );
        COPY_ARY_TO_BUF( mod->rs_symmetries );

        idx = 43;
        COPY_ARY_TO_BUF( mod->hop );

        idx = 48;
        COPY_ARY_TO_BUF( mod->vert );

        idx = 51;
        COPY_ARY_TO_BUF( mod->tu_ff );

        idx = 56;
        if (mod->data != NULL) {
            COPY_ARY_TO_BUF( mod->data );
        }

        if (cfg->kc) {
            idx = 80;
            COPY_ARY_TO_BUF( mod->internals->kmesh );
        }
        if (cfg->kf) {
            idx = 82;
            COPY_ARY_TO_BUF( mod->internals->kfmesh );
        }
        if (cfg->H) {
            idx = 84;
            COPY_ARY_TO_BUF( mod->internals->ham );
        }
        if (cfg->U) {
            idx = 86;
            COPY_ARY_TO_BUF( mod->internals->U );
        }
        if (cfg->E) {
            idx = 88;
            COPY_ARY_TO_BUF( mod->internals->E );
        }
        if (cfg->kc_ibz_path) {
            idx = 90;
            COPY_ARY_TO_BUF( k2b_serial );
        }
        if (cfg->kf_ibz_path) {
            idx = 92;
            COPY_ARY_TO_BUF( kf2b_serial );
        }

        // and generate the bandstructure array
        if (include_bandstructure) {
            if (use_band_indices) {
                mpi_vrb_printf("reading bandstructure from fine mesh, %li points\n", n_band_indices);
                double* E = mod->internals->E;
                double* kf = mod->internals->kfmesh;
                double* out = (double*)(buf+ibuf[100]);
                index_t nb = mod->n_orb * mod->n_spin;
                index_t cnt = 0;
                for (index_t i=0; i<n_band_indices; ++i) {
                    for (index_t b=0; b<nb; ++b)
                        out[cnt++] = E[band_indices[i]*nb + b];
                    for (int d=0; d<3; ++d)
                        out[cnt++] = kf[band_indices[i]*3+d];
                }
            } else {
                mpi_vrb_printf("generating bandstructure with %li points per segment\n", npath );
                generate_bandstructure( mod, (double*)(buf+ibuf[100]), npath );
            }
        }

        idx = 125;
        COPY_ARY_TO_BUF( version );

        // free band indices
        if (band_indices != NULL) free(band_indices);
        if (n_band_segments != NULL) free(n_band_segments);

        // calculate md5sum
        md5sum = md5sum_div(buf, ibuf[125]+ibuf[126]);
        lmd5sum = strlen(md5sum);

        // write to file
        FILE* f = fopen(fname, "wb");
        if (!f) {
            mpi_err_printf( "could not open file %s. no output.\n", fname );
        } else {
            fwrite(buf, sizeof(char), ibuf[125]+ibuf[126], f);
            fclose(f);
        }

        // and free the buffer
        free(buf);
    }

    free(k2b_serial);
    free(kf2b_serial);

    // broadcast the md5sum to all ranks
    diverge_mpi_bcast_bytes( &lmd5sum, sizeof(uindex_t), 0 );
    if (md5sum == NULL)
        // hack to not use malloc but static memory instead
        md5sum = md5sum_div( (char*)&lmd5sum, sizeof(uindex_t) );
    diverge_mpi_bcast_bytes( md5sum, lmd5sum+1, 0 );
    return md5sum;
}

static void generate_bandstructure( diverge_model_t* mod, double* buf, index_t npath ) {

    typedef Eigen::Vector<index_t,3> Vec3idx;

    if (mod->n_ibz_path <= 1) {
        mpi_wrn_printf("bandstructure is bogus. need n_ibz_path > 1\n");
        return;
    }

    Mat3d Gvec;
    Map<Mat3d> Lvec((double*)mod->lattice);
    diverge_model_generate_mom_basis(mod->lattice, (double(*)[3])Gvec.data());

    index_t no = mod->n_orb,
            ns = mod->n_spin,
            nb = mod->n_orb * mod->n_spin;

    complex128_t* H = (complex128_t*)malloc(sizeof(complex128_t)*POW2(nb));
    complex128_t* U = (complex128_t*)malloc(sizeof(complex128_t)*POW2(nb));
    #define zero_out(Hbuf) { \
        memset((void*)Hbuf, 0, sizeof(complex128_t)*POW2(nb)); \
    }

    index_t running_idx = 0;
    index_t nseg = mod->n_ibz_path-1;
    for (index_t seg=0; seg<nseg; ++seg) {
        index_t ni = (seg != (nseg-1)) ? (npath) : (npath+1);
        Map<Vec3d> s0vec( (double*)&mod->ibz_path[seg] );
        Map<Vec3d> s1vec( (double*)&mod->ibz_path[seg+1] );
        for (index_t i=0; i<ni; ++i) {
            double x = (double)i / (double)npath;
            Vec3d k0vec = Gvec * s0vec,
                  k1vec = Gvec * s1vec;
            Vec3d kvec = k0vec + (k1vec - k0vec) * x;
            zero_out(H);
            for (index_t h=0; h<mod->n_hop; ++h) {
                rs_hopping_t* hh = mod->hop+h;
                Vec3d Rvec = Lvec * Map<Vec3idx>( hh->R ).cast<double>();
                H[IDX4(hh->s2, hh->o2, hh->s1, hh->o1, no, ns, no)] +=
                    std::exp(-complex128_t(0,1) * Rvec.dot(kvec)) * hh->t;
            }
            double* here_buf = buf + (nb+3) * running_idx;
            eigen_solve(H, U, here_buf, nb);
            for (index_t b=0; b<nb; ++b) {
                here_buf[b] -= mod->internals->mu;
            }
            Map<Vec3d>(here_buf + nb) = kvec;
            running_idx++;
        }
    }

    free(H);
    free(U);
}

static void* kmesh_to_bands_serialize( diverge_model_t* mod, bool crs, index_t* psize ) {
    index_t *ibz_idx, nibz_idx;

    index_t* n_per_segment = crs ? diverge_kmesh_to_bands_crs( mod, &ibz_idx, &nibz_idx ) :
                                   diverge_kmesh_to_bands( mod, &ibz_idx, &nibz_idx );
    double* kmesh = crs ? mod->internals->kmesh : mod->internals->kfmesh;

    if (n_per_segment == NULL || ibz_idx == NULL || nibz_idx == -1) {
        free(n_per_segment);
        free(ibz_idx);
        *psize = 0;
        return NULL;
    }

    index_t sz = sizeof(index_t) * (1 + (mod->n_ibz_path-1)) +
                 sizeof(index_t) * (1 + nibz_idx) +
                 sizeof(double) * 3 * (nibz_idx);
    void* buf = malloc( sz );

    index_t* header = (index_t*)buf;
    index_t* path = header + (1 + (mod->n_ibz_path-1));
    double* vecs = (double*)(path + (1 + nibz_idx));

    header[0] = mod->n_ibz_path-1;
    for (index_t i=0; i<mod->n_ibz_path-1; ++i)
        header[i+1] = n_per_segment[i];

    path[0] = nibz_idx;
    for (index_t i=0; i<nibz_idx; ++i)
        path[i+1] = ibz_idx[i];

    for (index_t i=0; i<nibz_idx; ++i)
        for (short d=0; d<3; ++d)
            vecs[i*3+d] = kmesh[ibz_idx[i]*3+d];

    free(n_per_segment);
    free(ibz_idx);

    *psize = sz;
    return buf;
}
