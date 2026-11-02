#include "wigner.h"

#include "../diverge_model.h"
#include "../diverge_phonons.h"
#include "mpi_functions.h"
#include "bitary.h"
#include "text_io.h"
#include "gemm.h"

#include <ctype.h>

#define LINALG_NO_LAPACKE
#include "linalg.h"
LINALG_IMPLEMENT

static const double Rydberg = 13.605693122994,
                    Bohr    =  0.529177000000;

// this is needed for the interpretation of the ifc file
typedef struct {
    VECTOR_DECL( claV3ll_t, irvec )
    VECTOR_DECL( index_t, ndegen )
    VECTOR_DECL( double, wslen )

    int sgn; // needed for further operation
    index_t n_orb; // needed for iteration
} wsdata_t;
static wsdata_t wsdata_construct( index_t nr1, index_t nr2, index_t nr3,
        const double* lattice, const double* positions, index_t n_orb,
        const diverge_wigner_config_t* wc );
static void wsdata_free( wsdata_t w );

// this struct reads in the ifc file
typedef struct {
    int32_t ntyp, n_at, ibrav;
    double celldm[6];
    double lattice[3][3]; // if ibrav == 0

    struct {char n[64];}* names; // (ntyp,)
    double* masses; // (ntyp,)
    struct {int32_t typ, at; double pos[3];}* atoms; // (n_at,)

    int64_t nq[3];
    double* data; // (nq[0],nq[1],nq[2], n_at,3, n_at,3)
} ifc_t;
static ifc_t* read_ifc_file( const char* fname );
static void ifc_free( ifc_t* ifc );

diverge_wigner_config_t diverge_wigner_config_defaults_CPP( void ) { return diverge_wigner_config_defaults; }

static void wsdata_free( wsdata_t w ) {
    VECTOR_FREE( w.irvec ); VECTOR_FREE( w.ndegen ); VECTOR_FREE( w.wslen );
}

static inline claV3d_t claV3ll_to_double( claV3ll_t v ) {
    return (claV3d_t){{v.v[0], v.v[1], v.v[2]}};
}

#define vtree_key_t claV3ll_t
#define vtree_val_t index_t*
#define vtree_padding_bytes 4
#include "vtree.h"

static inline void wsdata_iter( vtree_node_t* node, void* ctx ) {
    wsdata_t* w = ctx;

    claV3ll_t R = node->key;
    index_t* ndeg_ary = node->val;
    double* wslen_ary = (double*)(ndeg_ary + POW2(w->n_orb));

    VECTOR_PUSH_BACK( w->irvec, R );
    for (index_t ij=0; ij<POW2(w->n_orb); ++ij)
        VECTOR_PUSH_BACK( w->ndegen, ndeg_ary[ij] );
    for (index_t ij=0; ij<POW2(w->n_orb); ++ij)
        VECTOR_PUSH_BACK( w->wslen, wslen_ary[ij] );
    free( ndeg_ary ); node->val = 0;
}

static wsdata_t wsdata_construct( index_t nr1, index_t nr2, index_t nr3,
        const double* lattice, const double* positions, index_t n_orb,
        const diverge_wigner_config_t* wc ) {

    wc = wc ? wc : &diverge_wigner_config_defaults;
    wsdata_t wsdata_global = {.sgn = wc->sgn, .n_orb = n_orb};
    claM3d_t A = claM3d_map(lattice);

    int dim = nr3 == 0 ? (nr2 == 0 ? (nr1 == 0 ? 0 : 1) : 2) : 3;
    nr1 = MAX(nr1,1); nr2 = MAX(nr2,1); nr3 = MAX(nr3,1);
    if (wc->dim != -1) dim = wc->dim;
    mpi_vrb_printf( "doing wsdata construction of dim %i\n", dim );

    if (dim == 0) return wsdata_global;

    index_t n_supercells = POW3(2*wc->nsc);
    claV3ll_t* supercells = calloc(n_supercells, sizeof*supercells);
    n_supercells = 0;
    for (index_t s=-wc->nsc; s<wc->nsc; ++s)
    for (index_t t=-wc->nsc; t<wc->nsc; ++t)
    for (index_t u=-wc->nsc; u<wc->nsc; ++u)
        supercells[n_supercells++] = (claV3ll_t){{s,t,u}};

    const int nthr = diverge_omp_num_threads();
    #pragma omp parallel num_threads(nthr)
    {
        xvtree_t* data = xvtree_init( nr1*nr2*nr3 );
        claV3ll_t* copies = calloc(n_supercells, sizeof*copies);
        claV3d_t* shifts = calloc(n_supercells, sizeof*shifts);
        claV3d_t* bonds = calloc(n_supercells, sizeof*bonds);
        double* lengths = calloc(n_supercells, sizeof*lengths);

        #pragma omp for collapse(3)
        for (index_t m1=0; m1<nr1; ++m1)
        for (index_t m2=0; m2<nr2; ++m2)
        for (index_t m3=0; m3<nr3; ++m3) {

            for (index_t s=0; s<n_supercells; ++s) {
                copies[s] = claV3ll_scale( (claV3ll_t){{ supercells[s].v[0] * nr1 + m1,
                          supercells[s].v[1] * nr2 + m2, supercells[s].v[2] * nr3 + m3, }}, wc->sgn );
                shifts[s] = claM3d_vecmat( claV3ll_to_double(copies[s]), A );
            }

            for (index_t i=0; i<n_orb; ++i)
            for (index_t j=0; j<n_orb; ++j) {
                double length = -2;
                for (index_t s=0; s<n_supercells; ++s) {
                    bonds[s] = claV3d_add(shifts[s], claV3d_sub(claV3d_map(positions+3*j),
                                claV3d_map(positions+3*i)));
                    lengths[s] = claV3d_norm(bonds[s]);
                    length = length < -1 ? lengths[s] : MIN(length, lengths[s]);
                }

                index_t nsel = 0;
                for (index_t s=0; s<n_supercells; ++s)
                    if (fabs(lengths[s] - length) < wc->eps) nsel++;

                for (index_t s=0; s<n_supercells; ++s) {
                    if (fabs(lengths[s] - length) < wc->eps) {
                        claV3ll_t R = copies[s];
                        vtree_node_t* R_data = xvtree_find_or_insert( data, R );
                        index_t* R_data_ndegen = R_data->val;
                        if (!R_data_ndegen) {
                            R_data->val = R_data_ndegen = calloc( POW2(n_orb), sizeof(index_t)+sizeof(double) );
                        }
                        double* R_data_length = (double*)(R_data_ndegen + POW2(n_orb));

                        R_data_ndegen[IDX2(i,j,n_orb)] = nsel;
                        R_data_length[IDX2(i,j,n_orb)] = length;
                    }
                }
            }
        }

        free( copies );
        free( shifts );
        free( bonds );
        free( lengths );

        wsdata_t wsdata = {.n_orb = n_orb};
        xvtree_iter( data, &wsdata_iter, &wsdata );
        xvtree_free( data );

        #pragma omp for ordered
        for (int t=0; t<nthr; ++t) {
            (void)t;
            #pragma omp ordered
            {
                cbytes_t* sel = bitary_alloc(wsdata.irvec_sz);
                for (index_t w=0; w<wsdata.irvec_sz; ++w) {
                    int dim_invalid = 0;
                    for (int d=dim; d<3; ++d)
                        if (wsdata.irvec[w].v[d] != 0) dim_invalid++;
                    if (!dim_invalid) {
                        bitary_set( sel, w );
                        VECTOR_PUSH_BACK( wsdata_global.irvec, wsdata.irvec[w] );
                    }
                }

                for (index_t wo=0; wo<wsdata.irvec_sz * POW2(n_orb); ++wo) {
                    if (bitary_get(sel, wo/POW2(n_orb)))
                        VECTOR_PUSH_BACK( wsdata_global.ndegen, wsdata.ndegen[wo] );
                }
                for (index_t wo=0; wo<wsdata.irvec_sz * POW2(n_orb); ++wo) {
                    if (bitary_get(sel, wo/POW2(n_orb)))
                        VECTOR_PUSH_BACK( wsdata_global.wslen, wsdata.wslen[wo] );
                }

                free( sel );
            }
        }

        wsdata_free( wsdata );
    }
    free( supercells );
    return wsdata_global;
}

static void lattice_from_ibrav( int ibrav, const double celldm[6], double* lat );

diverge_model_t* diverge_phonons_read_ifc( const char* fname, const diverge_wigner_config_t* wc ) {
    if (!wc) wc = &diverge_wigner_config_defaults;

    const double Rydberg_2 = (wc->apply_rydberg & DIVERGE_WIGNER_CONFIG_APPLY_RYDBERG_2_IFC) ?
        POW2(Rydberg) : 1.0;

    diverge_model_t* m = diverge_model_init();
    ifc_t* ifc = read_ifc_file( fname );
    if (!ifc) return NULL;

    if (ifc->ibrav) lattice_from_ibrav( ifc->ibrav, ifc->celldm, m->lattice[0] );
    else            claM3d_rmap( claM3d_scale(claM3d_map(ifc->lattice[0]),ifc->celldm[0]), m->lattice[0] );

    m->n_orb = ifc->n_at * 3;
    m->n_spin = m->SU2 = 1;
    double* pos = calloc( ifc->n_at, sizeof*pos * 3 );
    for (int a=0; a<ifc->n_at; ++a)
    for (int x=0; x<3; ++x) {
        pos[IDX2(a,x,3)] = ifc->atoms[a].pos[x] * ifc->celldm[0];
        memcpy( m->positions[IDX2(a,x,3)], ifc->atoms[a].pos, sizeof(double)*3 );
        for (int d=0; d<3; ++d) m->positions[IDX2(a,x,3)][d] *= ifc->celldm[0];
    }

    wsdata_t ws = wsdata_construct( ifc->nq[0], ifc->nq[1], ifc->nq[2],
            m->lattice[0], pos, ifc->n_at, wc );
    free( pos );

    double* masses = diverge_phonons_model_get_masses( m );
    for (index_t orb=0; orb<m->n_orb; ++orb)
        masses[orb] = ifc->masses[ifc->atoms[orb/3].typ-1];

    VECTOR_INIT( rs_hopping_t, hop );
    for (index_t n=0; n<ws.irvec_sz; ++n) {
        claV3ll_t R = claV3ll_scale( ws.irvec[n], ws.sgn );
        claV3ll_t pR = R; for (int d=0; d<3; ++d) {
            while (pR.v[d] < 0) pR.v[d] += ifc->nq[d];
            pR.v[d] %= ifc->nq[d];
        }
        index_t qq = IDX3(pR.v[0],pR.v[1],pR.v[2], ifc->nq[1],ifc->nq[2]);
        for (index_t o1=0; o1<m->n_orb; ++o1)
        for (index_t o2=0; o2<m->n_orb; ++o2) {
            double tmp = ifc->data[IDX3(qq, o1, o2, m->n_orb, m->n_orb)];
            index_t ndg = ws.ndegen[IDX3(n, o1/3, o2/3, ifc->n_at, ifc->n_at)];
            if (!wc->apply_ndegen) ndg = 1;
            if (ndg) {
                tmp /= (double)ndg;
                VECTOR_PUSH_BACK( hop, (rs_hopping_t){ .R={R.v[0],R.v[1],R.v[2]},
                        .o1=o2, .o2=o1, .t=tmp * Rydberg_2 /
                        sqrt(wc->apply_massdiv ? masses[o1]*masses[o2] : 1.0) } );
            }
        }
    }
    wsdata_free( ws );
    ifc_free( ifc );

    if (wc->apply_asr & DIVERGE_WIGNER_CONFIG_APPLY_ASR_HOPPINGS) {
        const double* asr_masses = (wc->apply_massdiv &&
                (wc->apply_asr & DIVERGE_WIGNER_CONFIG_APPLY_ASR_MASSDIV_H)) ? masses : NULL;
        hop = diverge_phonons_asr_n_orb_massdiv( hop, &hop_sz, m->n_orb, asr_masses );
    }
    m->hop = hop;
    m->n_hop = hop_cap = hop_sz;

    return m;
}

#define VRB_ASSERT( cond ) { \
    if (!(cond)) { mpi_err_printf( "%s\n", #cond ); errors++; } \
}

static ifc_t* read_ifc_file( const char* fname ) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        mpi_err_printf( "could not open file %s for reading\n", fname );
        return NULL;
    }
    ifc_t* ifc = calloc(1, sizeof*ifc);

    int nread = -1;
    nread = fscanf( f, "%i %i %i %lf %lf %lf %lf %lf %lf\n", &ifc->ntyp, &ifc->n_at,
            &ifc->ibrav, &ifc->celldm[0], &ifc->celldm[1], &ifc->celldm[2],
            &ifc->celldm[3], &ifc->celldm[4], &ifc->celldm[5] );
    int errors = 0;
    VRB_ASSERT( nread == 9 );

    if (ifc->ibrav == 0) {
        mpi_log_printf( "found ibrav = %i, read lattice by hand\n", ifc->ibrav );
        for (int i=0; i<9; ++i) {
            nread = fscanf( f, "%lf", ((double*)ifc->lattice[0])+i );
            VRB_ASSERT( nread == 1 );
        }
    }

    ifc->names = calloc(ifc->ntyp, sizeof*ifc->names);
    ifc->masses = calloc(ifc->ntyp, sizeof*ifc->masses);
    for (int t=0; t<ifc->ntyp; ++t) {
        int discard = -1;
        nread = fscanf( f, "%i '%[^\n^']' %lf\n", &discard, ifc->names[t].n, &ifc->masses[t] );
        VRB_ASSERT( nread == 3 );
    }

    ifc->atoms = calloc(ifc->n_at, sizeof*ifc->atoms);
    for (int a=0; a<ifc->n_at; ++a) {
        nread = fscanf( f, "%i %i %lf %lf %lf\n", &ifc->atoms[a].at, &ifc->atoms[a].typ,
                &ifc->atoms[a].pos[0], &ifc->atoms[a].pos[1], &ifc->atoms[a].pos[2] );
        VRB_ASSERT( nread == 5 );
    }

    char line[DIVERGE_GETLINE_MAX] = {0}; diverge_getline( line, f );
    char line_cmp[DIVERGE_GETLINE_MAX] = {0};
    char line_cpy[DIVERGE_GETLINE_MAX] = {0};
    strcpy(line_cpy, line); diverge_chomp_inplace(line_cpy);
    if (line_cpy[0] == 'T') { // we have some distance to cover to reach the qpts
        mpi_log_printf( "ifc: discarding lines\n" );
        mpi_log_printf( "=====================\n%s", line );
        for (int i=0; i<3; ++i) { // that's the lattice
            diverge_getline(line, f);
            if (mpi_loglevel_get()>=mpi_loglevel_log) mpi_eprintf( "%s", line );
        }
        for (int a=0; a<ifc->n_at; ++a) { // and now the Born charges
            diverge_getline(line, f);
            if (mpi_loglevel_get()>=mpi_loglevel_log) mpi_eprintf( "%s", line );
            for (int i=0; i<3; ++i) {
                diverge_getline(line, f);
                if (mpi_loglevel_get()>=mpi_loglevel_log) mpi_eprintf( "%s", line );
            }
        }

rescan_next_line:
        diverge_getline(line, f);
        strcpy(line_cpy, line); diverge_chomp_inplace(line_cpy);
        nread = sscanf(line_cpy, "%li %li %li", &ifc->nq[0], &ifc->nq[1], &ifc->nq[2]);
        sprintf(line_cmp, "%li %li %li", ifc->nq[0], ifc->nq[1], ifc->nq[2]);
        if (!strcmp(line_cmp, line_cpy)) { // actually found three integers
            mpi_log_printf( "=====================\n" );
            goto rescan_stop;
        } else {
            if (mpi_loglevel_get()>=mpi_loglevel_log) mpi_eprintf( "%s", line );
            goto rescan_next_line;
        }
    } else {
        nread = fscanf(f, "%li %li %li", &ifc->nq[0], &ifc->nq[1], &ifc->nq[2]);
    }

rescan_stop:
    VRB_ASSERT( nread == 3 );
    int64_t nq = ifc->nq[0]*ifc->nq[1]*ifc->nq[2];
    ifc->data = calloc(nq*POW2(ifc->n_at*3), sizeof*ifc->data);
    for (int64_t x=0; x<3; ++x)
    for (int64_t y=0; y<3; ++y)
    for (int64_t a=0; a<ifc->n_at; ++a)
    for (int64_t b=0; b<ifc->n_at; ++b)
    for (int64_t qz=0; qz<ifc->nq[2]; ++qz)
    for (int64_t qy=0; qy<ifc->nq[1]; ++qy)
    for (int64_t qx=0; qx<ifc->nq[0]; ++qx) {
        if (qx == 0 && qy == 0 && qz == 0) {
            int dc[4] = {0};
            nread = fscanf( f, "%i %i %i %i\n", &dc[0], &dc[1], &dc[2], &dc[3] );
            VRB_ASSERT( nread == 4 );
            int correct = (dc[0] == (x+1)) && (dc[1] == (y+1)) &&
                          (dc[2] == (a+1)) && (dc[3] == (b+1));
            if (!correct) {
                mpi_err_printf( "wrong orb/dir (file:%i,%i,%i,%i, loop:%li,%li,%li,%li)\n",
                        dc[0]-1, dc[1]-1, dc[2]-1, dc[3]-1, x, y, a, b );
            }
        }
        int qq[3] = {0};
        double d = 0;
        nread = fscanf( f, "%i %i %i %lf\n", &qq[0], &qq[1], &qq[2], &d );
        VRB_ASSERT( nread == 4 );
        int correct = (qq[0] == (qx+1)) && (qq[1] == (qy+1)) && (qq[2] == (qz+1));
        if (!correct) {
            mpi_err_printf( "wrong q-points (file:%i,%i,%i, loop:%li,%li,%li)\n",
                    qq[0]-1, qq[1]-1, qq[2]-1, qx, qy, qz );
        }
        int64_t qxyz = IDX3(qx,qy,qz, ifc->nq[1],ifc->nq[2]);
        ifc->data[IDX5(qxyz, a,x, b,y, ifc->n_at,3, ifc->n_at,3)] = d;
    }
    fclose( f );

    // rescale lengths to Angstrom
    ifc->celldm[0] *= Bohr;

    if (errors)
        mpi_err_printf( "read_ifc_file with %i errors... trying to return anyways\n", errors );

    return ifc;
}

static void ifc_free( ifc_t* ifc ) {
    if (ifc) {
        free( ifc->names );
        free( ifc->masses );
        free( ifc->atoms );
        free( ifc->data );
        free( ifc );
    }
}

static void lattice_from_ibrav( int ibrav, const double celldm[6], double* lat ) {
    const double a = celldm[0], b_a = celldm[1], c_a = celldm[2],
                 cos_gamma = celldm[3], cos_beta = celldm[4], cos_alpha = celldm[5];
    double tx = 0, ty = 0, tz = 0, u = 0, v = 0;
    claV3d_t v1 = {0}, v2 = {0}, v3 = {0};

    switch (ibrav) {
        /* cubic P (sc) */
        cubic:
        case 1: v1=(claV3d_t){{a,0,0}};
                v2=(claV3d_t){{0,a,0}};
                v3=(claV3d_t){{0,0,a}}; break;

        /* cubic F (fcc) */
        case 2: v1=(claV3d_t){{-a/2.,0,a/2.}};
                v2=(claV3d_t){{0, a/2.,a/2.}};
                v3=(claV3d_t){{-a/2.,a/2.,0}}; break;

        /* cubic I (bcc) */
        case 3: v1=(claV3d_t){{a/2.,a/2.,a/2.}};
                v2=(claV3d_t){{-a/2.,a/2.,a/2.}};
                v3=(claV3d_t){{-a/2.,-a/2.,a/2.}}; break;

        /* cubic I (bcc), more symmetric axis: */
        case -3: v1=claV3d_scale((claV3d_t){{-1,1,1}},  a/2.);
                 v2=claV3d_scale((claV3d_t){{1,-1,1}},  a/2.);
                 v3=claV3d_scale((claV3d_t){{1,1,-1}},  a/2.); break;

        /* Hexagonal and Trigonal P */
        case 4: v1=(claV3d_t){{a,0,0}};
                v2=(claV3d_t){{-a/2.,sqrt(3.)/2.*a,0}};
                v3=(claV3d_t){{0,0,a*c_a}}; break;

        /* Trigonal R, 3fold axis c */
        case 5: tx=sqrt((1.-cos_gamma)/2.), ty=sqrt((1.-cos_gamma)/6.), tz=sqrt((1.+2.*cos_gamma)/3.);
                v1=claV3d_scale((claV3d_t){{tx,-ty,tz}},   a);
                v2=claV3d_scale((claV3d_t){{0,2.*ty,tz}},  a);
                v3=claV3d_scale((claV3d_t){{-tx,-ty,tz}},  a); break;

        /* Trigonal R, 3fold axis <111>    celldm(4)=cos(gamma) */
        case -5: tx=sqrt((1.-cos_gamma)/2.), ty=sqrt((1.-cos_gamma)/6.), tz=sqrt((1.+2.*cos_gamma)/3.);
                 u = tz - 2.*sqrt(2.)*ty,  v = tz + sqrt(2.)*ty;
                 v1=claV3d_scale((claV3d_t){{u,v,v}}, a/sqrt(3.) );
                 v2=claV3d_scale((claV3d_t){{v,u,v}}, a/sqrt(3.) );
                 v3=claV3d_scale((claV3d_t){{v,v,u}}, a/sqrt(3.) ); break;

        /* Tetragonal P (st)               celldm(3)=c/a */
        case 6: v1=claV3d_scale((claV3d_t){{1,0,0}},   a);
                v2=claV3d_scale((claV3d_t){{0,1,0}},   a);
                v3=claV3d_scale((claV3d_t){{0,0,c_a}}, a); break;

        /* Tetragonal I (bct)              celldm(3)=c/a */
        case 7: v1=claV3d_scale((claV3d_t){{1,-1,c_a}},  a/2.);
                v2=claV3d_scale((claV3d_t){{1,1,c_a}},   a/2.);
                v3=claV3d_scale((claV3d_t){{-1,-1,c_a}}, a/2.); break;

        /* Orthorhombic P                  celldm(2)=b/a
                                           celldm(3)=c/a */
        case 8: v1=(claV3d_t){{a,0,0}};
                v2=(claV3d_t){{0,a*b_a,0}};
                v3=(claV3d_t){{0,0,a*c_a}}; break;

        /* Orthorhombic base-centered(bco) celldm(2)=b/a
                                           celldm(3)=c/a */
        case 9: v1=(claV3d_t){{a/2.,b_a*a/2.,0}};
                v2=(claV3d_t){{-a/2.,b_a*a/2.,0}};
                v3=(claV3d_t){{0,0,c_a*a}}; break;

        /* as 9, alternate description */
        case -9: v1=(claV3d_t){{a/2.,-b_a*a/2.,0}};
                 v2=(claV3d_t){{a/2., b_a*a/2.,0}};
                 v3=(claV3d_t){{0,0,c_a*a}}; break;
        /* Orthorhombic one-face base-centered A-type celldm(2)=b/a
                                                      celldm(3)=c/a */
        case 91: v1=(claV3d_t){{a, 0, 0}};
                 v2=(claV3d_t){{0,b_a*a/2.,-c_a*a/2.}};
                 v3=(claV3d_t){{0,b_a*a/2., c_a*a/2.}}; break;


        /* Orthorhombic face-centered      celldm(2)=b/a
                                           celldm(3)=c/a */
        case 10: v1=(claV3d_t){{a/2.,0,c_a*a/2.}};
                 v2=(claV3d_t){{a/2.,b_a*a/2.,0}};
                 v3=(claV3d_t){{0,b_a*a/2.,c_a*a/2.}}; break;

        /* Orthorhombic body-centered      celldm(2)=b/a
                                           celldm(3)=c/a */
        case 11: v1=(claV3d_t){{a/2.,b_a*a/2.,c_a*a/2.}};
                 v2=(claV3d_t){{-a/2.,b_a*a/2.,c_a*a/2.}};
                 v3=(claV3d_t){{-a/2.,-b_a*a/2.,c_a*a/2.}}; break;

        /* Monoclinic P, unique axis c     celldm(2)=b/a
                                           celldm(3)=c/a,
                                           celldm(4)=cos(ab) */
        /* where gamma is the angle between axis a and b. */
        case 12: v1=(claV3d_t){{a,0,0}};
                 v2=claV3d_scale((claV3d_t){{cos_gamma,sin(acos(cos_gamma)),0}}, b_a*a);
                 v3=(claV3d_t){{0,0,c_a*a}}; break;

        /* Monoclinic P, unique axis b     celldm(2)=b/a
                                           celldm(3)=c/a,
                                           celldm(5)=cos(ac) */
        /* where beta is the angle between axis a and c */
        case -12: v1=(claV3d_t){{a,0,0}};
                  v2=(claV3d_t){{0,b_a*a,0}};
                  v3=claV3d_scale((claV3d_t){{cos_beta,0,sin(acos(cos_beta))}}, c_a*a); break;

        /* Monoclinic base-centered        celldm(2)=b/a
           (unique axis c)                 celldm(3)=c/a,
                                           celldm(4)=cos(gamma) */
        /* where gamma=angle between axis a and b projected on xy plane */
        case 13: v1=(claV3d_t){{  a/2.,         0,          -c_a*a/2.}};
                 v2=(claV3d_t){{b_a*a*cos_gamma, b_a*a*sin(acos(cos_gamma)), 0}};
                 v3=(claV3d_t){{  a/2.,         0,           c_a*a/2.}}; break;

        /* Monoclinic base-centered        celldm(2)=b/a
           (unique axis b)                 celldm(3)=c/a,
                                           celldm(5)=cos(beta) */
        /* where beta=angle between axis a and c projected on xz plane
            IMPORTANT NOTICE: until QE v.6.4.1, axis for ibrav=-13 had a
            different definition: v1(old) =-v2(now), v2(old) = v1(now) */
        case -13: v1=(claV3d_t){{  a/2.,        b_a*a/2.,             0}};
                  v2=(claV3d_t){{ -a/2.,        b_a*a/2.,             0}};
                  v3=(claV3d_t){{c_a*a*cos_beta,       0, c_a*a*sin(acos(cos_beta))}}; break;

        /* Triclinic                       celldm(2)= b/a,
                                           celldm(3)= c/a,
                                           celldm(4)= cos(bc),
                                           celldm(5)= cos(ac),
                                           celldm(6)= cos(ab) */
        /* where alpha is the angle between axis b and c
                  beta is the angle between axis a and c
                ´gamma is the angle between axis a and b */
        case 14: v1=(claV3d_t){{a, 0, 0}};
                 v2=claV3d_scale((claV3d_t){{cos_gamma, sin(acos(cos_gamma)), 0}}, b_a*a);
                 v3=claV3d_scale((claV3d_t){{cos_beta,
                        (cos_alpha-cos_beta*cos_gamma)/sin(acos(cos_gamma)),
                        sqrt(1. + 2.*cos_alpha*cos_beta*cos_gamma -
                                POW2(cos_alpha)-POW2(cos_beta)-POW2(cos_gamma))/sin(acos(cos_gamma))
                  }}, c_a*a); break;

        default: mpi_err_printf( "ibrav = %i not implemented or wrong, reset to cubic\n", ibrav );
                 goto cubic; break;
    }
    memcpy( lat + 0*3, v1.v, sizeof(v1) );
    memcpy( lat + 1*3, v2.v, sizeof(v2) );
    memcpy( lat + 2*3, v3.v, sizeof(v3) );
}

typedef struct {
    int nrr_k, nrr_q, nrr_g;
    int dims, dims2; // dims2 is the ph n_at, dims is the el n_orb

    claV3i_t *irvec_k, *irvec_q, *irvec_g;
    int *ndegen_k /*(nrr_k,dims,dims)*/,
        *ndegen_q /*(nrr_q,dims2,dims2)*/,
        *ndegen_g /*(nrr_g,dims,dims2)*/;
} qe_wigner_t;

static void qe_wigner_free( qe_wigner_t* q ) {
    if (q) {
        free(q->irvec_k);
        free(q->irvec_q);
        free(q->irvec_g);
        free(q->ndegen_k);
        free(q->ndegen_q);
        free(q->ndegen_g);
        free(q);
    }
}

static int ndegen_irvec_read( FILE* f, claV3i_t* irvec, int nrr, int* ndegen, int dim ) {
    int errors = 0;
    int ndegen_count = 0;
    for (int ir=0; ir<nrr; ++ir) {
        // read irvec
        float discard = 0;
        int nread = fscanf( f, "%i %i %i %f\n", &irvec[ir].v[0],
                &irvec[ir].v[1], &irvec[ir].v[2], &discard );
        VRB_ASSERT( nread == 4 );
        // and then the ndegen (which are just dim integers in subsequent lines)
        for (int d=0; d<dim; ++d) {
            nread = fscanf( f, "%i", &ndegen[ndegen_count++] );
            VRB_ASSERT( nread == 1 );
        }
    }
    return errors;
}

#define READ_INT( nam ) \
    nam = 0; \
    { \
        int tmp = 0; \
        if (fread( &tmp, sizeof(int), 1, f ) != 1) { \
            mpi_err_printf( "cannot read int " #nam " from file\n" ); \
        } \
        mpi_vrb_printf( "read " #nam " = %i\n", tmp ); \
        nam = tmp; \
    }
#define READ_INT_ARY( nam, sz ) \
    nam = calloc( sz, sizeof(int) ); \
    if (fread( nam, sizeof(int), sz, f ) != (size_t)sz) { \
        mpi_err_printf( "cannot read int ary " #nam " from file\n" ); \
    } \
    mpi_vrb_printf( "read " #nam " (%li)\n", sz );
static qe_wigner_t* epw_wigner_read( const char* fname ) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        mpi_err_printf( "could not open file %s for reading\n", fname );
        return NULL;
    }
    qe_wigner_t* w = calloc(1, sizeof*w);
    READ_INT( w->dims );
    READ_INT( w->dims2 );
    READ_INT( w->nrr_k );
    READ_INT_ARY( w->irvec_k, w->nrr_k*3 );

    // must be transposed from (dims,dims, nrr_k) to (nrr_k, dims,dims)
    READ_INT_ARY( w->ndegen_k, POW2(w->dims)*w->nrr_k );
    transpose_inplace_rect_any( w->ndegen_k, sizeof*w->ndegen_k, POW2(w->dims), w->nrr_k );

    READ_INT( w->nrr_g );
    READ_INT_ARY( w->irvec_g, w->nrr_g*3 );

    // must be transposed from (dims2, nrr_g,dims) to (nrr_g,dims, dims2)
    READ_INT_ARY( w->ndegen_g, w->dims2*w->dims*w->nrr_g );
    transpose_inplace_rect_any( w->ndegen_g, sizeof*w->ndegen_g, w->dims2, w->nrr_g*w->dims );

    fclose( f );
    return w;
}

static qe_wigner_t* qe_wigner_read( const char* fname ) {
    int errors = 0;
    FILE* f = fopen(fname, "r");
    if (!f) {
        mpi_err_printf( "could not open file %s for reading\n", fname );
        return NULL;
    }

    qe_wigner_t* q = calloc(1, sizeof*q);
    int nread = fscanf( f, "%i %i %i %i %i\n", &q->nrr_k, &q->nrr_q,
            &q->nrr_g, &q->dims, &q->dims2 );
    VRB_ASSERT( nread == 5 );

    if (!errors) {
        q->irvec_k = calloc(q->nrr_k, sizeof*q->irvec_k);
        q->irvec_q = calloc(q->nrr_q, sizeof*q->irvec_q);
        q->irvec_g = calloc(q->nrr_g, sizeof*q->irvec_g);
        q->ndegen_k = calloc(POW2(q->dims) * q->nrr_k, sizeof*q->ndegen_k);
        q->ndegen_q = calloc(POW2(q->dims2) * q->nrr_q, sizeof*q->ndegen_q);
        q->ndegen_g = calloc(q->dims*q->dims2 * q->nrr_g, sizeof*q->ndegen_g);

        errors += ndegen_irvec_read( f, q->irvec_k, q->nrr_k, q->ndegen_k, POW2(q->dims) );
        errors += ndegen_irvec_read( f, q->irvec_q, q->nrr_q, q->ndegen_q, POW2(q->dims2) );
        errors += ndegen_irvec_read( f, q->irvec_g, q->nrr_g, q->ndegen_g, q->dims*q->dims2 );
    }

    fclose(f);

    if (errors) {
        qe_wigner_free(q);
        mpi_err_printf( "fallback to binary file (as in EPW)\n" );
        return epw_wigner_read(fname);
    } else {
        return q;
    }
}

static complex128_t* epmatwp_read( const char* fname, const qe_wigner_t* w ) {
    const index_t n_ph = w->dims2 * 3,
                  n_el = w->dims;
    FILE* f = fopen( fname, "rb" );
    if (!f) {
        mpi_err_printf( "could not open file %s for reading\n", fname );
        return NULL;
    }
    complex128_t* epw = calloc(POW2(n_el)*n_ph*w->nrr_k*w->nrr_g, sizeof*epw);
    index_t nread = fread( epw, sizeof*epw, POW2(n_el)*n_ph*w->nrr_k*w->nrr_g, f );
    if (nread != POW2(n_el)*n_ph*w->nrr_k*w->nrr_g)
        mpi_err_printf( "cannot read epw\n" );
    fclose( f );

    for (index_t irg=0; irg<w->nrr_g; ++irg)
    for (index_t p=0; p<n_ph; ++p)
    for (index_t irk=0; irk<w->nrr_k; ++irk)
    for (index_t m=0; m<n_el; ++m)
    for (index_t n=0; n<n_el; ++n) {
        index_t ndg_el = w->ndegen_k[IDX3(irk,m,n, n_el,n_el)];
        index_t ndg_ph = w->ndegen_g[IDX3(irg,n,p/3, n_el,n_ph/3)];
        epw[IDX5(irg,p, irk,m,n, n_ph,w->nrr_k,n_el,n_el)] *= (ndg_el?1./ndg_el:0.0) * (ndg_ph?1./ndg_ph:0.0);
    }

    return epw;
}

static ph_vertex_t* epmatwp_to_ph_vertex_t( const complex128_t* data,
        claV3i_t* Rk, index_t nRk, claV3i_t* Rg, index_t nRg,
        index_t n_orb_ph, index_t n_orb_el, double maxrel, index_t* p_vtx_sz ) {

    double vtx_max = 0.0;
    index_t ntot = nRg * n_orb_ph * nRk * POW2(n_orb_el);
    #pragma omp parallel for num_threads(diverge_omp_num_threads()) reduction(max:vtx_max)
    for (index_t i=0; i<ntot; ++i) {
        double cv = cabs(data[i]);
        vtx_max = MAX(vtx_max, cv);
    }

    index_t n_vtx = 0;
    const double l_vtx = vtx_max*maxrel;
    #pragma omp parallel for num_threads(diverge_omp_num_threads()) reduction(+:n_vtx)
    for (index_t i=0; i<ntot; ++i) {
        if (cabs(data[i]) >= l_vtx)
            n_vtx++;
    }

    *p_vtx_sz = n_vtx;
    VECTOR_INIT_CAP( ph_vertex_t, vtx, n_vtx );
    if (!vtx)
        mpi_err_printf( "cant allocate %.0fGB ph_vertex_t(%li,)\n",
                (double)(sizeof(ph_vertex_t)*n_vtx) / (double)(1024*1024*1024), n_vtx );
    for (index_t g=0; g<nRg; ++g)
    for (index_t OA=0; OA<n_orb_ph; ++OA)
    for (index_t k=0; k<nRk; ++k)
    for (index_t o1=0; o1<n_orb_el; ++o1)
    for (index_t o2=0; o2<n_orb_el; ++o2) {
        index_t i = IDX5(g,OA,k,o2,o1, n_orb_ph,nRk,n_orb_el,n_orb_el);
        if (cabs(data[i]) >= l_vtx) {
            VECTOR_PUSH_BACK( vtx, (ph_vertex_t){ .R12={Rk[k].v[0], Rk[k].v[1], Rk[k].v[2]},
                    .o1=o1, .o2=o2, .OA=OA, .RA={Rg[g].v[0], Rg[g].v[1], Rg[g].v[2]},
                    .g=data[i]} );
        }
    }
    return vtx;
}

ph_vertex_t* diverge_phonons_read_elph( const char* wigner, const char* epmatwp,
        const double* masses, double maxrel, index_t* p_vtx_sz, int apply_rydberg ) {
    qe_wigner_t* q = qe_wigner_read( wigner );
    complex128_t* g = epmatwp_read( epmatwp, q );
    ph_vertex_t* pv = NULL;

    const double Rydberg_3_2 = apply_rydberg ? pow(Rydberg, 1.5) : 1.0;
    if (g) {
        pv = epmatwp_to_ph_vertex_t( g, q->irvec_k, q->nrr_k, q->irvec_g,
                q->nrr_g, q->dims2*3, q->dims, maxrel, p_vtx_sz );
        for (index_t i=0; i<*p_vtx_sz; ++i) pv[i].g *= Rydberg_3_2/sqrt(masses[pv[i].OA]);
        free( g );
    }
    qe_wigner_free(q);
    return pv;
}

double* diverge_phonons_model_get_masses( diverge_model_t* m ) {
    return m->positions[MAX_N_ORBS-m->n_orb/3];
}

diverge_model_t* diverge_phonons_read_ifc_and_elph( const char* ifc,
        const char* wigner, const char* epmatwp, double maxrel,
        const diverge_wigner_config_t* wc ) {
    if (!wc) wc = &diverge_wigner_config_defaults;
    diverge_model_t* m = diverge_phonons_read_ifc( ifc, wc );

    // logic to enable/disable orbital detection
    const double* masses = diverge_phonons_model_get_masses(m);

    m->vert = (rs_vertex_t*)diverge_phonons_read_elph( wigner, epmatwp,
            masses, maxrel, &m->n_vert, wc->apply_rydberg & DIVERGE_WIGNER_CONFIG_APPLY_RYDBERG_3_2_G);
    if ((wc->apply_asr&DIVERGE_WIGNER_CONFIG_APPLY_ASR_VERTICES) && m->vert)
        m->vert = (rs_vertex_t*)diverge_phonons_asr_vertex_n_orb_massdiv( (ph_vertex_t*)m->vert,
                &m->n_vert, m->n_orb, (wc->apply_asr&DIVERGE_WIGNER_CONFIG_APPLY_ASR_MASSDIV_V)?masses:NULL );

    return m;
}
