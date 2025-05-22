#include "wigner.h"

#include "../diverge_model.h"
#include "../diverge_phonons.h"
#include "mpi_functions.h"
#include "btree.h"
#include "bitary.h"

#define LINALG_NO_LAPACKE
#include "linalg.h"
LINALG_IMPLEMENT

// this is needed for the interpretation of the ifc file
typedef struct {
    VECTOR_DECL( claV3ll_t, irvec )
    VECTOR_DECL( index_t, ndegen )
    VECTOR_DECL( double, wslen )

    int sgn; // needed for further operation
    index_t n_orb; // needed for iteration
} wsdata_t;
wsdata_t wsdata_construct( index_t nr1, index_t nr2, index_t nr3,
        const double* lattice, const double* positions, index_t n_orb,
        const diverge_wigner_config_t* wc );
void wsdata_free( wsdata_t w );

// this struct reads in the ifc file
typedef struct {
    int32_t ntyp, n_at, ibrav;
    double celldm[6];

    struct {char n[64];}* names; // (ntyp,)
    double* masses; // (ntyp,)
    struct {int32_t typ, at; double pos[3];}* atoms; // (n_at,)

    int64_t nq[3];
    double* data; // (nq[0],nq[1],nq[2], n_at,3, n_at,3)
} ifc_t;
ifc_t* read_ifc_file( const char* fname );
void ifc_free( ifc_t* ifc );


static inline index_t py_mod( index_t i, index_t n ) { return (n + (i % n)) % n; }
diverge_wigner_config_t diverge_wigner_config_defaults_CPP( void ) { return diverge_wigner_config_defaults; }

void wsdata_free( wsdata_t w ) {
    VECTOR_FREE( w.irvec ); VECTOR_FREE( w.ndegen ); VECTOR_FREE( w.wslen );
}

static inline claV3d_t claV3ll_to_double( claV3ll_t v ) {
    return (claV3d_t){{v.v[0], v.v[1], v.v[2]}};
}

typedef struct { int64_t v0:21, v1:21, v2:21, v3:1; } compressed_t;
static inline index_t claV3ll_to_index( claV3ll_t v ) {
    compressed_t c = {.v0=v.v[0], .v1=v.v[1], .v2=v.v[2]};
    index_t cc = 0; memcpy( &cc, &c, sizeof cc );
    return cc;
}
static inline claV3ll_t index_to_claV3ll( index_t i ) {
    compressed_t c = {0}; memcpy( &c, &i, sizeof c );
    return (claV3ll_t){{ c.v0, c.v1, c.v2 }};
}

static inline void wsdata_iter( btree_t* node, void* ctx ) {
    wsdata_t* w = ctx;

    claV3ll_t R = index_to_claV3ll( node->key );
    index_t* ndeg_ary = (index_t*)node->val;
    double* wslen_ary = (double*)(ndeg_ary + POW2(w->n_orb));

    VECTOR_PUSH_BACK( w->irvec, R );
    for (index_t ij=0; ij<POW2(w->n_orb); ++ij)
        VECTOR_PUSH_BACK( w->ndegen, ndeg_ary[ij] );
    for (index_t ij=0; ij<POW2(w->n_orb); ++ij)
        VECTOR_PUSH_BACK( w->wslen, wslen_ary[ij] );
    free( ndeg_ary ); node->val = 0;
}

wsdata_t wsdata_construct( index_t nr1, index_t nr2, index_t nr3,
        const double* lattice, const double* positions, index_t n_orb,
        const diverge_wigner_config_t* wc ) {

    wc = wc ? wc : &diverge_wigner_config_defaults;
    wsdata_t wsdata_global = {.sgn = wc->sgn, .n_orb = n_orb};
    claM3d_t A = claM3d_scale(claM3d_map(lattice), wc->sgn);

    int dim = nr3 == 0 ? (nr2 == 0 ? (nr1 == 0 ? 0 : 1) : 2) : 3;
    nr1 = MAX(nr1,1); nr2 = MAX(nr2,1); nr3 = MAX(nr3,1);
    if (wc->dim != -1) dim = wc->dim;
    mpi_vrb_printf( "doing wsdata construction of dim %i\n", dim );

    if (dim == 0) return wsdata_global;

    index_t n_supercells = POW3(2*wc->nsc);
    claV3ll_t supercells[n_supercells];
    n_supercells = 0;
    for (index_t s=-wc->nsc; s<wc->nsc; ++s)
    for (index_t t=-wc->nsc; t<wc->nsc; ++t)
    for (index_t u=-wc->nsc; u<wc->nsc; ++u)
        supercells[n_supercells++] = (claV3ll_t){{s,t,u}};

    const int nthr = diverge_omp_num_threads();
    #pragma omp parallel num_threads(nthr)
    {
        xbtree_t* data = xbtree_init( 1024 );

        #pragma omp for collapse(3)
        for (index_t m1=0; m1<nr1; ++m1)
        for (index_t m2=0; m2<nr2; ++m2)
        for (index_t m3=0; m3<nr3; ++m3) {

            claV3ll_t copies[n_supercells];
            claV3d_t shifts[n_supercells];
            for (index_t s=0; s<n_supercells; ++s) {
                copies[s] = (claV3ll_t){{ supercells[s].v[0] * nr1 + m1,
                                          supercells[s].v[1] * nr2 + m2,
                                          supercells[s].v[2] * nr3 + m3, }};
                shifts[s] = claM3d_vecmat( claV3ll_to_double(copies[s]), A );
            }

            for (index_t i=0; i<n_orb; ++i)
            for (index_t j=0; j<n_orb; ++j) {
                claV3d_t bonds[n_supercells];
                double lengths[n_supercells];
                for (index_t s=0; s<n_supercells; ++s) {
                    bonds[s] = claV3d_add(shifts[s], claV3d_sub(claV3d_map(positions+3*j),
                                claV3d_map(positions+3*i)));
                    lengths[s] = claV3d_norm(bonds[s]);
                }
                double length = lengths[0];
                for (index_t s=1; s<n_supercells; ++s) length = MIN(length,lengths[s]);

                index_t nsel = 0;
                for (index_t s=0; s<n_supercells; ++s)
                    if (fabs(lengths[s] - length) < wc->eps)
                        nsel++;

                for (index_t s=0; s<n_supercells; ++s) {
                    if (fabs(lengths[s] - length) < wc->eps) {
                        claV3ll_t R = copies[s];
                        index_t R_key = claV3ll_to_index( R );
                        btree_t* R_data = xbtree_find_or_insert( data, R_key );
                        index_t* R_data_ndegen = (index_t*)R_data->val;
                        if (!R_data_ndegen) {
                            R_data_ndegen = calloc( POW2(n_orb), sizeof(index_t)+sizeof(double) );
                            R_data->val = (index_t)R_data_ndegen;
                        }
                        double* R_data_length = (double*)(R_data_ndegen + POW2(n_orb));

                        R_data_ndegen[IDX2(i,j,n_orb)] = nsel;
                        R_data_length[IDX2(i,j,n_orb)] = length;
                    }
                }
            }
        }
        wsdata_t wsdata = {.n_orb = n_orb};
        xbtree_iter( data, &wsdata_iter, &wsdata );
        xbtree_free( data );

        #pragma omp for ordered
        for (int t=0; t<nthr; ++t) {
            (void)t;
            #pragma omp ordered
            {
                cbyte_t* sel = bitary_alloc(wsdata.irvec_sz);
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
    return wsdata_global;
}

static void lattice_from_ibrav( int ibrav, const double celldm[6], double* lat );

diverge_model_t* diverge_phonons_read_ifc( const char* fname, const diverge_wigner_config_t* wc ) {
    diverge_model_t* m = diverge_model_init();
    ifc_t* ifc = read_ifc_file( fname );
    if (!ifc) return NULL;

    lattice_from_ibrav( ifc->ibrav, ifc->celldm, m->lattice[0] );
    m->n_orb = ifc->n_at * 3;
    m->n_spin = m->SU2 = 1;
    double* pos = calloc( ifc->n_at, sizeof*pos * 3 );
    for (int a=0; a<ifc->n_at; ++a)
    for (int x=0; x<3; ++x) {
        pos[IDX2(a,x,3)] = ifc->atoms[a].pos[x];
        memcpy( m->positions[IDX2(a,x,3)], ifc->atoms[a].pos, sizeof(double)*3 );
        for (int d=0; d<3; ++d) m->positions[IDX2(a,x,3)][d] *= ifc->celldm[0];
    }

    wsdata_t ws = wsdata_construct( ifc->nq[0], ifc->nq[1], ifc->nq[2],
            m->lattice[0], pos, ifc->n_at, wc );
    free( pos );

    VECTOR_INIT( rs_hopping_t, hop );
    for (index_t n=0; n<ws.irvec_sz; ++n) {
        claV3ll_t R = claV3ll_scale( ws.irvec[n], ws.sgn );
        claV3ll_t mR = {0}; for (int d=0; d<3; ++d) mR.v[d] = py_mod(R.v[d], ifc->nq[d]);
        claV3ll_t pR = mR; for (int d=0; d<3; ++d) { while (mR.v[d] < 0) mR.v[d] += ifc->nq[d]; }
        index_t qq = IDX3( pR.v[0], pR.v[1], pR.v[2], ifc->nq[1], ifc->nq[2] );
        for (index_t o1=0; o1<m->n_orb; ++o1)
        for (index_t o2=0; o2<m->n_orb; ++o2) {
            double tmp = ifc->data[IDX3(qq, o1, o2, m->n_orb, m->n_orb)];
            index_t ndg = ws.ndegen[IDX3(n, o1/3, o2/3, ifc->n_at, ifc->n_at)];
            if (ndg) {
                tmp /= (double)ndg;
                VECTOR_PUSH_BACK( hop, (rs_hopping_t){ .R={R.v[0],R.v[1],R.v[2]}, .o1=o1, .o2=o2, .t=tmp } );
            }
        }
    }
    wsdata_free( ws );
    ifc_free( ifc );

    m->hop = hop = diverge_phonons_asr( hop, &hop_sz );
    m->n_hop = hop_cap = hop_sz;

    return m;
}

#define VRB_ASSERT( cond ) { \
    if (!(cond)) { mpi_err_printf( "%s\n", #cond ); } \
}

ifc_t* read_ifc_file( const char* fname ) {
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
    VRB_ASSERT( nread == 9 );

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

    nread = fscanf( f, "F\n" );
    nread = fscanf( f, "%li %li %li\n", &ifc->nq[0], &ifc->nq[1], &ifc->nq[2] );
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
    return ifc;
}

void ifc_free( ifc_t* ifc ) {
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
    (void)a;
    (void)b_a;
    (void)c_a;
    (void)cos_gamma;
    (void)cos_beta;
    (void)cos_alpha;

    claV3d_t v1 = {0}, v2 = {0}, v3 = {0};
    switch (ibrav) {
        /* cubic P (sc) */
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

        /* Hexagonal and Trigonal P */
        case 4: v1=(claV3d_t){{a,0,0}};
                v2=(claV3d_t){{-a/2.,sqrt(3.)/2.*a,0}};
                v3=(claV3d_t){{0,0,a*c_a}}; break;
        default:
                mpi_err_printf( "ibrav = %i not implemented/wrong\n", ibrav );
                break;
        /* XXX from the QE manual
         -3          cubic I (bcc), more symmetric axis:
              v1 = (a/2)(-1,1,1), v2 = (a/2)(1,-1,1),  v3 = (a/2)(1,1,-1)

          5          Trigonal R, 3fold axis c        celldm(4)=cos(gamma)
              The crystallographic vectors form a three-fold star around
              the z-axis, the primitive cell is a simple rhombohedron:
              v1 = a(tx,-ty,tz),   v2 = a(0,2ty,tz),   v3 = a(-tx,-ty,tz)
              where c=cos(gamma) is the cosine of the angle gamma between
              any pair of crystallographic vectors, tx, ty, tz are:
                tx=sqrt((1-c)/2), ty=sqrt((1-c)/6), tz=sqrt((1+2c)/3)
         -5          Trigonal R, 3fold axis <111>    celldm(4)=cos(gamma)
              The crystallographic vectors form a three-fold star around
              <111>. Defining a' = a/sqrt(3) :
              v1 = a' (u,v,v),   v2 = a' (v,u,v),   v3 = a' (v,v,u)
              where u and v are defined as
                u = tz - 2*sqrt(2)*ty,  v = tz + sqrt(2)*ty
              and tx, ty, tz as for case ibrav=5
              Note: if you prefer x,y,z as axis in the cubic limit,
                    set  u = tz + 2*sqrt(2)*ty,  v = tz - sqrt(2)*ty
                    See also the note in Modules/latgen.f90

          6          Tetragonal P (st)               celldm(3)=c/a
              v1 = a(1,0,0),  v2 = a(0,1,0),  v3 = a(0,0,c/a)

          7          Tetragonal I (bct)              celldm(3)=c/a
              v1=(a/2)(1,-1,c/a),  v2=(a/2)(1,1,c/a),  v3=(a/2)(-1,-1,c/a)

          8          Orthorhombic P                  celldm(2)=b/a
                                                     celldm(3)=c/a
              v1 = (a,0,0),  v2 = (0,b,0), v3 = (0,0,c)

          9          Orthorhombic base-centered(bco) celldm(2)=b/a
                                                     celldm(3)=c/a
              v1 = (a/2, b/2,0),  v2 = (-a/2,b/2,0),  v3 = (0,0,c)
         -9          as 9, alternate description
              v1 = (a/2,-b/2,0),  v2 = (a/2, b/2,0),  v3 = (0,0,c)
         91          Orthorhombic one-face base-centered A-type
                                                     celldm(2)=b/a
                                                     celldm(3)=c/a
              v1 = (a, 0, 0),  v2 = (0,b/2,-c/2),  v3 = (0,b/2,c/2)

         10          Orthorhombic face-centered      celldm(2)=b/a
                                                     celldm(3)=c/a
              v1 = (a/2,0,c/2),  v2 = (a/2,b/2,0),  v3 = (0,b/2,c/2)

         11          Orthorhombic body-centered      celldm(2)=b/a
                                                     celldm(3)=c/a
              v1=(a/2,b/2,c/2),  v2=(-a/2,b/2,c/2),  v3=(-a/2,-b/2,c/2)

         12          Monoclinic P, unique axis c     celldm(2)=b/a
                                                     celldm(3)=c/a,
                                                     celldm(4)=cos(ab)
              v1=(a,0,0), v2=(b*cos(gamma),b*sin(gamma),0),  v3 = (0,0,c)
              where gamma is the angle between axis a and b.
        -12          Monoclinic P, unique axis b     celldm(2)=b/a
                                                     celldm(3)=c/a,
                                                     celldm(5)=cos(ac)
              v1 = (a,0,0), v2 = (0,b,0), v3 = (c*cos(beta),0,c*sin(beta))
              where beta is the angle between axis a and c

         13          Monoclinic base-centered        celldm(2)=b/a
                     (unique axis c)                 celldm(3)=c/a,
                                                     celldm(4)=cos(gamma)
              v1 = (  a/2,         0,          -c/2),
              v2 = (b*cos(gamma), b*sin(gamma), 0  ),
              v3 = (  a/2,         0,           c/2),
              where gamma=angle between axis a and b projected on xy plane

        -13          Monoclinic base-centered        celldm(2)=b/a
                     (unique axis b)                 celldm(3)=c/a,
                                                     celldm(5)=cos(beta)
              v1 = (  a/2,       b/2,             0),
              v2 = ( -a/2,       b/2,             0),
              v3 = (c*cos(beta),   0,   c*sin(beta)),
              where beta=angle between axis a and c projected on xz plane
         IMPORTANT NOTICE: until QE v.6.4.1, axis for ibrav=-13 had a
         different definition: v1(old) =-v2(now), v2(old) = v1(now)

         14          Triclinic                       celldm(2)= b/a,
                                                     celldm(3)= c/a,
                                                     celldm(4)= cos(bc),
                                                     celldm(5)= cos(ac),
                                                     celldm(6)= cos(ab)
              v1 = (a, 0, 0),
              v2 = (b*cos(gamma), b*sin(gamma), 0)
              v3 = (c*cos(beta),  c*(cos(alpha)-cos(beta)cos(gamma))/sin(gamma),
                   c*sqrt( 1 + 2*cos(alpha)cos(beta)cos(gamma)
                             - cos(alpha)^2-cos(beta)^2-cos(gamma)^2 )/sin(gamma) )
              where alpha is the angle between axis b and c
                     beta is the angle between axis a and c
                    gamma is the angle between axis a and b
        */
    }
    memcpy( lat + 0*3, v1.v, sizeof(v1) );
    memcpy( lat + 1*3, v2.v, sizeof(v2) );
    memcpy( lat + 2*3, v3.v, sizeof(v3) );
}

