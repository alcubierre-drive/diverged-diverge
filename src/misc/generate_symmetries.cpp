/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "generate_symmetries.h"
#include "real_harmonics.h"
#include "cmath_helpers.h"
#include <vector>
#include <cassert>
using std::vector;

typedef Eigen::Matrix<complex128_t,2,2,Eigen::RowMajor> CMat2cd;
typedef Eigen::Matrix<double,3,3,Eigen::RowMajor> CMat3d;
typedef Eigen::Matrix<complex128_t,-1,-1,Eigen::RowMajor> CMatXcd;

static const double d2rad = M_PI/180.0;
static const index_t n_spherical = 1+3+5+7+9;

static const Mat3d Mxz = (Eigen::Matrix3d() <<
                            1.0,  0.0, 0.0,
                            0.0, -1.0, 0.0,
                            0.0,  0.0, 1.0).finished();
const Vec2d up = (Eigen::Vector2d() << 1.0    ,  0.0 ).finished();
const Vec2d dn = (Eigen::Vector2d() << 0.0    ,  1.0 ).finished();

static Mat3d t_spatial(char TYPE, Vec3d a, double theta);
static Mat2cd t_spin(char TYPE, Vec3d a, double theta, index_t n_spin);
static MatXcd t_orbital(char TYPE, Vec3d a, Vec3d euler);

static inline MatXcd Wigner_dMatrix(Vec3d euler_anles);
static inline MatXcd Mirror_xz();
static inline MatXcd Parity();

static Vec3d gen_euler_angle(const Vec3d a,const double theta);
static vector<spherical_harmonic_t> construct_spherical_harmonic_vector( vector<site_descr_t>& orbital_content );

void diverge_generate_symm_trafo( index_t n_spin, const site_descr_t* orbs, index_t n_orb,
    const sym_op_t* syms, index_t n_syms, double* rs_trafo, complex128_t* orb_trafo ) {

    Map<CMat3d> rs_sym( rs_trafo );
    Map<CMatXcd> comb_sym( orb_trafo, n_orb*n_spin, n_orb*n_spin );

    rs_sym = Mat3d::Identity();

    Mat2cd spin_sym = Mat2d::Identity();
    MatXcd orb_sym = MatXcd::Identity(n_spherical,n_spherical);

    vector<sym_op_t> symmetry(n_syms);
    memcpy( symmetry.data(), syms, sizeof(sym_op_t)*n_syms );

    vector<site_descr_t> orbital_content(n_orb);
    memcpy( orbital_content.data(), orbs, sizeof(site_descr_t)*n_orb );

    for (const sym_op_t& elem : symmetry) {
        Vec3d ax( elem.normal_vector[0], elem.normal_vector[1], elem.normal_vector[2] );
        if (ax.norm() > 1e-5) ax /= ax.norm();
        double angle = elem.angle;
        if (elem.type == 'R') angle *= d2rad;
        rs_sym *= t_spatial(elem.type, ax, angle);
        spin_sym *= t_spin(elem.type, ax, angle, n_spin);
        orb_sym *= t_orbital(elem.type, ax, gen_euler_angle(ax,angle));
    }
    vector<spherical_harmonic_t> harmonics = construct_spherical_harmonic_vector(orbital_content);

    for( index_t o1 = 0; o1 < n_orb; ++o1 )
    for( index_t s1 = 0; s1 < n_spin; ++s1 ) {
        Vec2cd sp1 = spin_sym*((s1 == 0) ? up : dn);
        spherical_harmonic_t op1 = orb_sym * harmonics[o1];
        for( index_t o2 = 0; o2 < n_orb; ++o2 )
        for( index_t s2 = 0; s2 < n_spin; ++s2 ) {
            spherical_harmonic_t op2 = harmonics[o2];
            Vec2cd sp2 = ((s2 == 0) ? up : dn);
            orb_trafo[o1+n_orb*(s1+n_spin*(o2+n_orb*s2))] = op2.dot(op1) * sp2.dot(sp1);
        }
    }
}

// all subtrafos generation
static Mat3d t_spatial(char TYPE, Vec3d a, double theta) {
    if (TYPE=='R') {
        double ct = cos(theta);
        double st = sin(theta);
        double omct = 1.-cos(theta);
        return (Mat3d() <<
            ct + POW2(a(0))*omct, a(0)*a(1)*omct-a(2)*st, a(0)*a(2)*omct+a(1)*st,
            a(0)*a(1)*omct+a(2)*st, ct + POW2(a(1))*omct, a(1)*a(2)*omct-a(0)*st,
            a(0)*a(2)*omct-a(1)*st, a(1)*a(2)*omct+a(0)*st, ct + a(2)*a(2)*omct).finished();
    } else if (TYPE=='M') {
        Vec3d normal_vector;
        normal_vector << 0.0,1.0,0.0;
        Vec3d rot_axis = a.cross(normal_vector);
        double angle = acos(a.dot(normal_vector));
        if(std::abs(angle) < 1e-5)
            return Mxz;
        rot_axis /= rot_axis.norm();
        return t_spatial('R',rot_axis,-angle)* Mxz * t_spatial('R',rot_axis,angle);
    } else if (TYPE=='I') {
        return (Mat3d() << -1.0,.0    ,.0 ,
                            .0   , -1.0, .0 ,
                            .0    ,.0    , -1.0).finished();
    } else if (TYPE=='E' || TYPE=='F' || TYPE=='S') {
        return (Mat3d() << 1.0,.0    ,.0 ,
                           .0   , 1.0, .0 ,
                           .0    ,.0    , 1.0).finished();
    }else{
        mpi_err_printf("Unknown operation!! default to trivial case\n");
        return (Mat3d() << 1.0,.0    ,.0 ,
                            .0   , 1.0, .0 ,
                            .0   ,.0  , 1.0).finished();
    }
}

static Mat2cd t_spin(char TYPE, Vec3d a, double theta, index_t nspin) {
    if(nspin == 1)
        return (Mat2cd() <<   1.0, 0.0,
                              0.0, 1.0).finished();
    if (TYPE=='S' || TYPE=='R') {
        double p = atan2(a(1),a(0));
        double t = acos(a(2));
        const Mat2cd S = (Mat2cd() << cos(t)         , exp(-I128*p)*sin(t),
                                      exp(I128*p)*sin(t), -cos(t)).finished();

        return cos(theta/2.)*Mat2cd::Identity() - I128*sin(theta/2.)*S;
    } else if  (TYPE=='M') {
        Vec3d normal_vector;
        normal_vector << 0.0,0.0,1.0;
        Vec3d rot_axis = a.cross(normal_vector);
        rot_axis /= rot_axis.norm();
        double angle = acos(a.dot(normal_vector));
        if(std::abs(angle) < 1e-8)
        {return t_spin('S',normal_vector, M_PI, nspin);}
        return t_spin('S',rot_axis,-angle, nspin) *
                t_spin('S',normal_vector, M_PI, nspin) *
                 t_spin('S',rot_axis,angle, nspin);
    } else if  (TYPE=='F') {
        return (Mat2cd() <<   0.0, 1.0,
                              1.0, 0.0).finished();
    }else if (TYPE=='E' || TYPE=='I'){
        return (Mat2cd() <<   1.0, 0.0,
                              0.0, 1.0).finished();
    }else{
        mpi_err_printf("Unknown operation!! default to trivial case\n");
        return (Mat2cd() <<   1.0, 0.0,
                              0.0, 1.0).finished();
    }
}

static MatXcd t_orbital(char TYPE, Vec3d a, Vec3d euler) {
    if (TYPE=='R') {
        return Wigner_dMatrix(euler);
    } else if (TYPE=='M') {
        Vec3d normal_vector;
        normal_vector << 0.0,1.0,0.0;
        Vec3d rot_axis = a.cross(normal_vector);
        double angle = acos(a.dot(normal_vector));
        if(std::abs(angle) < 1e-8)
        {return Mirror_xz();}
        rot_axis /= rot_axis.norm();
        Vec3d e2 = gen_euler_angle(rot_axis,-angle);
        Vec3d e1 = gen_euler_angle(rot_axis, angle);
        return Wigner_dMatrix(e2) * Mirror_xz() *
                        Wigner_dMatrix(e1);
    } else if (TYPE=='I') {
        return Parity();
    } else if (TYPE=='E' || TYPE=='F' || TYPE=='S') {
        return MatXcd::Identity(n_spherical,n_spherical);
    } else {
        mpi_err_printf("Unknown operation!! default to trivial case\n");
        return MatXcd::Identity(n_spherical,n_spherical);
    }
}


#define fac( arg ) tgamma( (arg)+1 )

static inline MatXcd Wigner_dMatrix(Vec3d euler_anles) {
    MatXcd D = MatXcd::Zero(n_spherical,n_spherical);
    const double sb2 = sin(euler_anles(1)/2.0);
    const double cb2 = cos(euler_anles(1)/2.0);

    index_t off = 0;
    for(int l = 0; l < 5; ++l) {
        for(int m = -l;m<=l;++m)
        for(int mp = -l;mp<=l;++mp) {
            const double prefac = sqrt((double)(fac(l+mp)*fac(l-mp)*fac(l+m)*fac(l-m)));
            const complex128_t p2 = exp(-I128*((double)mp*euler_anles(2)+(double)m*euler_anles(0)));
            const int s_min = (m - mp < 0) ? 0 : (m - mp);
            const int s_max = (l + m < l - mp) ? (l + m) : (l - mp);
            double sum = 0.0;
            for(int s = s_min;s<=s_max;++s) {
                double pref = ((mp-m+s)%2 == 0 ) ? 1 : -1;
                double denom = fac(l+m-s)*fac(s)*fac(mp-m+s)*fac(l-mp-s);
                double exp_cb = cmath_pow(cb2, 2*l+m-mp-2*s);
                double exp_sb = cmath_pow(sb2, mp-m+2*s);
                sum += pref*exp_sb*exp_cb/denom;
            }
            D(m+l+off,mp+l+off) = sum*p2*prefac;
        }
        off += 2*l+1;
    }
    return D;
}

static inline MatXcd Mirror_xz() {
    MatXcd Ml = MatXd::Zero(n_spherical,n_spherical);
    index_t off = 0;
    for(int l = 0; l < 5; ++l) {
        for(int m = -l;m<=l;++m) {
            int mp = -m;
            Ml(m+l+off,mp+l+off) = cmath_pow(-1., m);
        }
        off += 2*l+1;
    }
    return Ml;
}

static inline MatXcd Parity() {
    MatXcd Pmat = MatXd::Identity(n_spherical,n_spherical);
    index_t off = 0;
    for(int l = 0; l < 5; ++l) {
        index_t blocksize = 2*l+1;
        for(index_t elem = 0; elem < blocksize; ++elem) {
            Pmat(elem + off,elem + off) = cmath_pow(-1., l);
        }
        off += blocksize;
    }
    return Pmat;
}

static Vec3d gen_euler_angle(const Vec3d a,const double theta) {
    double ct = cos(theta);
    double st = sin(theta);
    double omct = 1.-cos(theta);
    Vec3d help;
    const Mat3d R =  (Mat3d() <<
        ct + POW2(a(0))*omct, a(0)*a(1)*omct-a(2)*st, a(0)*a(2)*omct+a(1)*st,
        a(0)*a(1)*omct+a(2)*st, ct + POW2(a(1))*omct, a(1)*a(2)*omct-a(0)*st,
        a(0)*a(2)*omct-a(1)*st, a(1)*a(2)*omct+a(0)*st, ct + a(2)*a(2)*omct).finished();

    if(R(2,2) < 1.-1e-8) {
        if(R(2,2) > -1.+1e-8) {
            help(2) = atan2(R(2,1),-R(2,0));
            help(0) = atan2(R(1,2), R(0,2));
            help(1) = acos(R(2,2));
        }else{ // R_22 = -1
            help(2) = 0.;
            help(0) = -atan2(R(1,0), R(1,1));
            help(1) = -M_PI;
        }
    } else {
        help(2) = 0.;
        help(0) = atan2(R(1,0), R(1,1));
        help(1) = 0.;
    }
    return help;
}

spherical_harmonic_t extract_spherical_harmonic(real_harmonics_t function) {
    spherical_harmonic_t result = real_harmonics_s;
    switch(function) {
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
        default: result = real_harmonics_s; mpi_err_printf("invalid input!\n"); break;
    }
    return result;
}

static spherical_harmonic_t rotate(spherical_harmonic_t inp, Vec3d v_in, Vec3d v_fin) {
    Vec3d rot_vec = v_in.cross(v_fin);
    if(rot_vec.norm() > 1e-10) {
        double angle = atan2(rot_vec.norm(), v_in.dot(v_fin));
        rot_vec /= rot_vec.norm();
        spherical_harmonic_t helper = t_orbital('R', rot_vec, gen_euler_angle(rot_vec,angle))*inp;
        assert((v_fin/v_fin.norm()-t_spatial('R',rot_vec,angle) * v_in).norm() < 1e-10);
        inp = helper;
    }
    return inp;
}

static vector<spherical_harmonic_t> construct_spherical_harmonic_vector( vector<site_descr_t>& orbital_content ) {
    index_t n_orb = orbital_content.size();
    vector<spherical_harmonic_t> orbs;
    orbs.resize(n_orb);

    Vec3d ez = {0.,0.,1.};
    Vec3d ex = {1.,0.,0.};
    for(int o = 0; o < n_orb; ++o) {
        orbs[o] = spherical_harmonic_t::Zero();
        if(orbital_content[o].n_functions < 1)
            mpi_err_printf("orbital %d has no functions specified \n",o);
        assert(orbital_content[o].n_functions > 0);
        for(int id = 0; id < orbital_content[o].n_functions; ++id) {
            spherical_harmonic_t orbital = extract_spherical_harmonic(orbital_content[o].function[id]);
            Map<Vec3d> exdef (&(orbital_content[o].xaxis[id][0]));
            Map<Vec3d> ezdef (&(orbital_content[o].zaxis[id][0]));
            orbital = rotate(orbital,ex,exdef);
            orbital = rotate(orbital,ez,ezdef);
            orbs[o] += orbital_content[o].amplitude[id] * orbital;
        }
        if(std::abs(orbs[o].norm() - 1.0) > 1e-10 ) {
            mpi_log_printf("orbital at %d not normalized, fix now \n",o);
            orbs[o] /= orbs[o].norm();
        }
    }
    return orbs;
}
