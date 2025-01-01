/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "generate_symmetries.h"
/* #include "cmath_helpers.h" */
#include "assert.h"

#define LINALG_NO_LAPACKE
#define LINALG_EXPORT static inline
#include "linalg.h"
LINALG_IMPLEMENT
LINALG_IMPL( LINALG_PREFIX, 25 )

#define SKIP_CLAV25CD_T
#include "real_harmonics.h"

static const double d2rad = M_PI/180.0;

static const claM3d_t Mxz = { {{1.0,  0.0, 0.0},
                               {0.0, -1.0, 0.0},
                               {0.0,  0.0, 1.0}} };
static const claV2cd_t up = {{ 1.0, 0.0 }},
                       dn = {{ 0.0, 1.0 }};

static claM3d_t t_spatial(char TYPE, claV3d_t a, double theta);
static claM2cd_t t_spin(char TYPE, claV3d_t a, double theta, index_t n_spin);
static claM25cd_t t_orbital(char TYPE, claV3d_t a, claV3d_t euler);

static inline claM25cd_t Wigner_dMatrix(claV3d_t euler_angles);
static inline claM25cd_t Mirror_xz( void );
static inline claM25cd_t Parity( void );

static claV3d_t gen_euler_angle(claV3d_t a, double theta);
static spherical_harmonic_t* gen_spherical_harmonics( const site_descr_t* orbs, index_t n_orb );

void diverge_generate_symm_trafo( index_t n_spin, const site_descr_t* orbs, index_t n_orb,
    const sym_op_t* syms, index_t n_syms, double* rs_trafo, complex128_t* orb_trafo ) {

    claM3d_t rs_sym = claM3d_id();
    claM2cd_t spin_sym = claM2cd_id();
    claM25cd_t orb_sym = claM25cd_id();

    VECTOR_INIT( sym_op_t, symmetry );
    for (index_t i=0; i<n_syms; ++i) {
        // for improper rotation we push back the rotation as well as the mirror
        if (syms[i].type == 'r') {
            sym_op_t xop = syms[i];
            xop.type = 'R'; VECTOR_PUSH_BACK( symmetry, xop );
            xop.type = 'I'; VECTOR_PUSH_BACK( symmetry, xop );
        } else {
            VECTOR_PUSH_BACK( symmetry, syms[i] );
        }
    }

    // apply symmetries to the three spaces
    for (index_t s=0; s<symmetry_sz; ++s) {
        sym_op_t elem = symmetry[s];
        claV3d_t ax = claV3d_map( elem.normal_vector );
        if (claV3d_norm(ax) > 1e-5) ax = claV3d_scale(ax, 1./claV3d_norm(ax));
        double angle = elem.angle;
        if (elem.type == 'R') angle *= d2rad;
        rs_sym = claM3d_matmul(rs_sym, t_spatial(elem.type, ax, angle));
        spin_sym = claM2cd_matmul(spin_sym, t_spin(elem.type, ax, angle, n_spin));
        orb_sym = claM25cd_matmul(orb_sym, t_orbital(elem.type, ax, gen_euler_angle(ax,angle)));
    }
    free( symmetry );

    // output RS
    claM3d_rmap( rs_sym, rs_trafo );

    // output orb
    spherical_harmonic_t* harmonics = gen_spherical_harmonics( orbs, n_orb );
    for (index_t o1=0; o1<n_orb; ++o1)
    for (index_t s1=0; s1<n_spin; ++s1) {
        claV2cd_t sp1 = claM2cd_matvec(spin_sym, ((s1 == 0) ? up : dn));
        spherical_harmonic_t op1 = claM25cd_matvec(orb_sym, harmonics[o1]);
        for (index_t o2=0; o2<n_orb; ++o2)
        for (index_t s2=0; s2<n_spin; ++s2) {
            spherical_harmonic_t op2 = harmonics[o2];
            claV2cd_t sp2 = ((s2 == 0) ? up : dn);
            complex128_t val = claV25cd_dot( op2, op1 ) * claV2cd_dot( sp2, sp1 );
            orb_trafo[o1+n_orb*(s1+n_spin*(o2+n_orb*s2))] = val;
            // printf( "%+.2f%+.2fj%c", creal(val), cimag(val), ((o2==n_orb-1) && (s2==n_spin-1)) ? '\n' : ' ' );
        }
    }
    free( harmonics );
}

// all subtrafos generation
static claM3d_t t_spatial(char TYPE, claV3d_t a_, double theta) {
    if (TYPE=='R') {
        double ct = cos(theta);
        double st = sin(theta);
        double omct = 1.-cos(theta);
        double* a = a_.v;
        return (claM3d_t){{{ct + POW2(a[0])*omct, a[0]*a[1]*omct-a[2]*st, a[0]*a[2]*omct+a[1]*st},
                           {a[0]*a[1]*omct+a[2]*st, ct + POW2(a[1])*omct, a[1]*a[2]*omct-a[0]*st},
                           {a[0]*a[2]*omct-a[1]*st, a[1]*a[2]*omct+a[0]*st, ct + a[2]*a[2]*omct}}};
    } else if (TYPE=='M') {
        claV3d_t normal_vector = {{0.0,1.0,0.0}};
        claV3d_t rot_axis = claV3d_cross(a_, normal_vector);
        double angle = acos(claV3d_dot(a_, normal_vector));
        if (fabs(angle) < 1e-5) return Mxz;
        rot_axis = claV3d_scale(rot_axis, 1./claV3d_norm(rot_axis));
        return claM3d_matmul(claM3d_matmul( t_spatial('R',rot_axis,-angle), Mxz), t_spatial('R',rot_axis,angle) );
    } else if (TYPE=='I') {
        return claM3d_scale(claM3d_id(), -1.0);
    } else if (TYPE=='E' || TYPE=='F' || TYPE=='S') {
        return claM3d_id();
    } else {
        mpi_err_printf("unknown spatial symop (%c), default to 'E'\n", TYPE);
        return claM3d_id();
    }
}

static claM2cd_t t_spin(char TYPE, claV3d_t a_, double theta, index_t nspin) {
    if (nspin == 1)
        return claM2cd_id();
    if (TYPE=='S' || TYPE=='R') {
        double* a = a_.v;
        double p = atan2(a[1],a[0]);
        double t = acos(a[2]);
        claM2cd_t S = {{{cos(t),  cexp(-I128*p)*sin(t)},
                        {cexp(I128*p)*sin(t), -cos(t)}}};
        return claM2cd_sub(claM2cd_scale(claM2cd_id(),cos(theta/2.)), claM2cd_scale(S,I128*sin(theta/2.)));
    } else if  (TYPE=='M') {
        claV3d_t normal_vector = {{0,0,1}};
        claV3d_t rot_axis = claV3d_cross(a_,normal_vector);
        rot_axis = claV3d_scale( rot_axis, 1./claV3d_norm(rot_axis) );
        double angle = acos(claV3d_dot(a_, normal_vector));
        if (fabs(angle) < 1e-8)
            return t_spin('S', normal_vector, M_PI, nspin);
        return claM2cd_matmul( claM2cd_matmul( t_spin('S',rot_axis,-angle, nspin),
                                               t_spin('S',normal_vector, M_PI, nspin) ),
                                               t_spin('S',rot_axis,angle, nspin) );
    } else if  (TYPE=='F') {
        return (claM2cd_t){{{0.0,1.0},
                            {1.0,0.0}}};
    } else if (TYPE=='E' || TYPE=='I') {
        return claM2cd_id();
    } else {
        mpi_err_printf("unknown spin symop (%c), default to 'E'\n", TYPE);
        return claM2cd_id();
    }
}

static claM25cd_t t_orbital(char TYPE, claV3d_t a, claV3d_t euler) {
    if (TYPE=='R') {
        return Wigner_dMatrix(euler);
    } else if (TYPE=='M') {
        claV3d_t normal_vector = {{0,1,0}};
        claV3d_t rot_axis = claV3d_cross(a, normal_vector);
        double angle = acos(claV3d_dot(a, normal_vector));
        if (fabs(angle) < 1e-8)
            return Mirror_xz();
        rot_axis = claV3d_scale( rot_axis, 1./claV3d_norm(rot_axis) );
        claV3d_t e2 = gen_euler_angle(rot_axis,-angle);
        claV3d_t e1 = gen_euler_angle(rot_axis, angle);
        return claM25cd_matmul(claM25cd_matmul( Wigner_dMatrix(e2), Mirror_xz()), Wigner_dMatrix(e1));
    } else if (TYPE=='I') {
        return Parity();
    } else if (TYPE=='E' || TYPE=='F' || TYPE=='S') {
        return claM25cd_id();
    } else {
        mpi_err_printf("unknown orbital symop (%c), default to 'E'\n", TYPE);
        return claM25cd_id();
    }
}

#define fac( arg ) tgamma( (arg)+1 )

static inline claM25cd_t Wigner_dMatrix(claV3d_t euler_angles_) {
    claM25cd_t D = {0};
    double* euler_angles = euler_angles_.v;
    const double sb2 = sin(euler_angles[1]/2.0);
    const double cb2 = cos(euler_angles[1]/2.0);

    index_t off = 0;
    for (int l=0; l<5; ++l) {
        for (int m=-l;m<=l;++m)
        for (int mp=-l;mp<=l;++mp) {
            const double prefac = sqrt(fac(l+mp)*fac(l-mp)*fac(l+m)*fac(l-m));
            const complex128_t p2 = cexp(-I128*((double)mp*euler_angles[2]+(double)m*euler_angles[0]));
            const int s_min = (m - mp < 0) ? 0 : (m - mp);
            const int s_max = (l + m < l - mp) ? (l + m) : (l - mp);
            double sum = 0.0;
            for (int s=s_min; s<=s_max; ++s) {
                double pref = ((mp-m+s)%2 == 0 ) ? 1 : -1;
                double denom = fac(l+m-s)*fac(s)*fac(mp-m+s)*fac(l-mp-s);
                double exp_cb = pow(cb2, 2*l+m-mp-2*s);
                double exp_sb = pow(sb2, mp-m+2*s);
                sum += pref*exp_sb*exp_cb/denom;
            }
            D.m[m+l+off][mp+l+off] = sum*p2*prefac;
        }
        off += 2*l+1;
    }
    return D;
}

static inline claM25cd_t Mirror_xz( void ) {
    claM25cd_t Ml = {0};
    index_t off = 0;
    for (int l=0; l<5; ++l) {
        for (int m=-l; m<=l; ++m) {
            int mp = -m;
            Ml.m[m+l+off][mp+l+off] = pow(-1.,m);
        }
        off += 2*l+1;
    }
    return Ml;
}

static inline claM25cd_t Parity( void ) {
    claM25cd_t Pmat = claM25cd_id();
    index_t off = 0;
    for (int l=0; l<5; ++l) {
        index_t blocksize = 2*l+1;
        for (index_t elem=0; elem<blocksize; ++elem) {
            Pmat.m[elem + off][elem + off] = pow(-1.,l);
        }
        off += blocksize;
    }
    return Pmat;
}

static claV3d_t gen_euler_angle(claV3d_t a_, double theta) {
    double ct = cos(theta);
    double st = sin(theta);
    double omct = 1.-cos(theta);
    double* a = a_.v;
    claM3d_t R = {{{ct + POW2(a[0])*omct, a[0]*a[1]*omct-a[2]*st, a[0]*a[2]*omct+a[1]*st},
                   {a[0]*a[1]*omct+a[2]*st, ct + POW2(a[1])*omct, a[1]*a[2]*omct-a[0]*st},
                   {a[0]*a[2]*omct-a[1]*st, a[1]*a[2]*omct+a[0]*st, ct + a[2]*a[2]*omct}}};
    claV3d_t help = {0};

    if (R.m[2][2] < (1.-1e-8)) {
        if (R.m[2][2] > (-1.+1e-8)) {
            help.v[2] = atan2(R.m[2][1],-R.m[2][0]);
            help.v[0] = atan2(R.m[1][2], R.m[0][2]);
            help.v[1] = acos(R.m[2][2]);
        } else { // R_22 = -1
            help.v[2] = 0.;
            help.v[0] = -atan2(R.m[1][0], R.m[1][1]);
            help.v[1] = -M_PI;
        }
    } else {
        help.v[2] = 0.;
        help.v[0] = atan2(R.m[1][0], R.m[1][1]);
        help.v[1] = 0.;
    }
    return help;
}

spherical_harmonic_t extract_spherical_harmonic(real_harmonics_t function) {
    spherical_harmonic_t result = real_harmonics_s;
    switch (function) {
        case 0:  result = real_harmonics_s; break;
        case 1:  result = real_harmonics_py; break;
        case 2:  result = real_harmonics_pz; break;
        case 3:  result = real_harmonics_px; break;
        case 4:  result = real_harmonics_dxy; break;
        case 5:  result = real_harmonics_dyz; break;
        case 6:  result = real_harmonics_dz2; break;
        case 7:  result = real_harmonics_dxz; break;
        case 8:  result = real_harmonics_dx2my2; break;
        case 9:  result = real_harmonics_f_3_m3; break;
        case 10: result = real_harmonics_f_3_m2; break;
        case 11: result = real_harmonics_f_3_m1; break;
        case 12: result = real_harmonics_f_3_m0; break;
        case 13: result = real_harmonics_f_3_1; break;
        case 14: result = real_harmonics_f_3_2; break;
        case 15: result = real_harmonics_f_3_3; break;
        case 16: result = real_harmonics_g_4_m4; break;
        case 17: result = real_harmonics_g_4_m3; break;
        case 18: result = real_harmonics_g_4_m2; break;
        case 19: result = real_harmonics_g_4_m1; break;
        case 20: result = real_harmonics_g_4_0; break;
        case 21: result = real_harmonics_g_4_1; break;
        case 22: result = real_harmonics_g_4_2; break;
        case 23: result = real_harmonics_g_4_3; break;
        case 24: result = real_harmonics_g_4_4; break;
        default: result = real_harmonics_s; mpi_err_printf("invalid real harmonic (%i)\n", function); break;
    }
    return result;
}

static spherical_harmonic_t rotatexz(spherical_harmonic_t inp, claV3d_t exi, claV3d_t exf, claV3d_t ezi, claV3d_t ezf) {
    claV3d_t rotv_x = claV3d_cross(exi,exf);
    double angle_x = atan2(claV3d_norm(rotv_x), claV3d_dot(exi,exf));
    if (claV3d_norm(rotv_x) > 1e-10) {
        rotv_x = claV3d_scale( rotv_x, 1./claV3d_norm(rotv_x) );
        inp = claM25cd_matvec(t_orbital('R',rotv_x,gen_euler_angle(rotv_x,angle_x)), inp);
        ezi = claM3d_matvec( t_spatial('R',rotv_x,angle_x), ezi );
        /* assert((exf/exf.norm()-t_spatial('R',rotv_x,angle_x) * exi).norm() < 1e-10); */
    }

    claV3d_t rotv_z = claV3d_cross(ezi,ezf);
    double angle_z = atan2(claV3d_norm(rotv_z), claV3d_dot(ezi,ezf));
    if (claV3d_norm(rotv_z) > 1e-10) {
        rotv_z = claV3d_scale( rotv_z, 1./claV3d_norm(rotv_z) );
        inp = claM25cd_matvec(t_orbital('R',rotv_z,gen_euler_angle(rotv_z,angle_z)), inp);
        /* assert((ezf/ezf.norm()-t_spatial('R',rotv_z,angle_z) * ezi).norm() < 1e-10); */
    }
    return inp;
}

static spherical_harmonic_t* gen_spherical_harmonics( const site_descr_t* orbs, index_t n_orb ) {
    spherical_harmonic_t* sphv = calloc( n_orb, sizeof*sphv );

    claV3d_t ez = {{0.,0.,1.}};
    claV3d_t ex = {{1.,0.,0.}};
    for (int o=0; o<n_orb; ++o) {
        if (orbs[o].n_functions < 1)
            mpi_err_printf("orbital %d has no functions specified \n",o);
        assert(orbs[o].n_functions > 0);
        for (int id=0; id<orbs[o].n_functions; ++id) {
            spherical_harmonic_t orbital = extract_spherical_harmonic(orbs[o].function[id]);
            claV3d_t exdef = claV3d_map(orbs[o].xaxis[id]),
                     ezdef = claV3d_map(orbs[o].zaxis[id]);
            orbital = rotatexz(orbital, ex,exdef, ez,ezdef);
            sphv[o] = claV25cd_add(sphv[o], claV25cd_scale(orbital, orbs[o].amplitude[id]));
        }
        if (fabs(claV25cd_norm(sphv[o]) - 1.0) > 1e-10) {
            mpi_log_printf("orbital at %d not normalized, fix now \n",o);
            sphv[o] = claV25cd_scale(sphv[o], 1./claV25cd_norm(sphv[o]));
        }
    }
    return sphv;
}

