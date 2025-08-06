#ifndef DIVERGE_SKIP_TESTS
#ifndef DIVERGE_SKIP_TESTS_LINALG

#define CT_SUITE_NAME linalg_cla
#include "../ctest.h"
// #include "../catch.hpp"
#include "../../src/diverge_common.h"
#include "../../src/diverge_Eigen3.hpp"
#include "t_linalg_cla_solvers.hpp"

#include <string>
#include <sstream>

// we include linalg with the additional test_ prefix. it is compiled in the .c
// file with the same name as this file except for the ending.
#define LINALG_EXPORT extern "C"
#include "../../src/misc/linalg.h"
LINALG_TYPES(test_cla, 2)
LINALG_TYPES(test_cla, 3)
LINALG_TYPES(test_cla, 4)

#ifndef CLA_ZERO
#define CLA_ZERO 1e-9
#endif

#include <iostream>

typedef Eigen::Matrix<int64_t,2,1> Vec2ll;
typedef Eigen::Matrix<int64_t,3,1> Vec3ll;
typedef Eigen::Matrix<int64_t,4,1> Vec4ll;

#define RANDOM_DATA_GEN_V2d Vec2d::Random()
#define RANDOM_DATA_GEN_V3d Vec3d::Random()
#define RANDOM_DATA_GEN_V4d Vec4d::Random()
#define RANDOM_DATA_GEN_V2cd Vec2cd::Random()
#define RANDOM_DATA_GEN_V3cd Vec3cd::Random()
#define RANDOM_DATA_GEN_V4cd Vec4cd::Random()
#define RANDOM_DATA_GEN_V2i Vec2i::Random().array().unaryExpr([](int x){return x%100;}).matrix()
#define RANDOM_DATA_GEN_V3i Vec3i::Random().array().unaryExpr([](int x){return x%100;}).matrix()
#define RANDOM_DATA_GEN_V4i Vec4i::Random().array().unaryExpr([](int x){return x%100;}).matrix()
#define RANDOM_DATA_GEN_V2ll Vec2ll::Random().array().unaryExpr([](int64_t x)->int64_t{return x%100;}).matrix()
#define RANDOM_DATA_GEN_V3ll Vec3ll::Random().array().unaryExpr([](int64_t x)->int64_t{return x%100;}).matrix()
#define RANDOM_DATA_GEN_V4ll Vec4ll::Random().array().unaryExpr([](int64_t x)->int64_t{return x%100;}).matrix()

#ifndef SKIP_FANCY_PARENS
#define PAREN_L ((row>1) ? (i==0) ? "⎛" : (i==row-1) ? "⎝" : "⎜" : "(")
#define PAREN_R ((row>1) ? (i==0) ? "⎞" : (i==row-1) ? "⎠" : "⎟" : ")")
#else // SKIP_FANCY_PARENS
#define PAREN_L "["
#define PAREN_R "]"
#endif // SKIP_FANCY_PARENS

static const char print_matrix_padding[] = "    ";
static void print_matrix_d( char* buf_, double* mat, int row, int col ) {
    char* buf = buf_;
    for (int i=0; i<row; ++i) {
        sprintf( buf, "%s%s", print_matrix_padding, PAREN_L ); buf += strlen(buf);
        for (int j=0; j<col; ++j) {
            sprintf( buf, "%+4.1f  ", mat[IDX2(i,j,col)] );
            buf += strlen(buf);
        }
        sprintf( buf -= 2, "%s\n", PAREN_R ); buf += strlen(buf);
    }
    for (size_t i=0; i<strlen(buf_); ++i)
        if (buf_[i] == '+')  buf_[i] = ' ';
}

static void print_matrix_cd( char* buf_, complex128_t* mat, int row, int col ) {
    char* buf = buf_;
    for (int i=0; i<row; ++i) {
        sprintf( buf, "%s%s", print_matrix_padding, PAREN_L ); buf += strlen(buf);
        for (int j=0; j<col; ++j) {
            sprintf( buf, "%+4.1f%+4.1fj  ", mat[IDX2(i,j,col)].real(), mat[IDX2(i,j,col)].imag() );
            for (size_t i=0; i<strlen(buf); ++i) {
                if (buf[i] == '+')  buf[i] = ' ';
                break;
            }
            buf += strlen(buf);
        }
        sprintf( buf -= 2, "%s\n", PAREN_R ); buf += strlen(buf);
    }
}

static void print_matrix_i( char* buf_, int* mat, int row, int col ) {
    char* buf = buf_;
    for (int i=0; i<row; ++i) {
        sprintf( buf, "%s%s", print_matrix_padding, PAREN_L ); buf += strlen(buf);
        for (int j=0; j<col; ++j) {
            sprintf( buf, "%3i  ", mat[IDX2(i,j,col)] );
            buf += strlen(buf);
        }
        sprintf( buf -= 2, "%s\n", PAREN_R ); buf += strlen(buf);
    }
    for (size_t i=0; i<strlen(buf_); ++i)
        if (buf_[i] == '+')  buf_[i] = ' ';
}

static void print_matrix_ll( char* buf_, int64_t* mat, int row, int col ) {
    char* buf = buf_;
    for (int i=0; i<row; ++i) {
        sprintf( buf, "%s%s", print_matrix_padding, PAREN_L ); buf += strlen(buf);
        for (int j=0; j<col; ++j) {
            sprintf( buf, "%3li  ", mat[IDX2(i,j,col)] );
            buf += strlen(buf);
        }
        sprintf( buf -= 2, "%s\n", PAREN_R ); buf += strlen(buf);
    }
    for (size_t i=0; i<strlen(buf_); ++i)
        if (buf_[i] == '+')  buf_[i] = ' ';
}

#define VECTOR_TEST_CASE( dim, dtype_long, dtype_short ) \
\
static inline Vec##dim##dtype_short c##dim##dtype_short##Xeigen( test_claV##dim##dtype_short##_t vec ) { \
    return Map<Vec##dim##dtype_short>( vec.v ); \
} \
\
CT_TEST_NAME( claV##dim##dtype_short##_t, "cla,V" #dim #dtype_short ) { \
    Vec##dim##dtype_short RANDOM_DATA = RANDOM_DATA_GEN_V##dim##dtype_short ; \
    test_claV##dim##dtype_short##_t C_RANDOM_DATA; \
    for (int i=0; i<dim; ++i) C_RANDOM_DATA.v[i] = RANDOM_DATA(i); \
 \
    Vec##dim##dtype_short RANDOM_DATA2 = RANDOM_DATA_GEN_V##dim##dtype_short ; \
    test_claV##dim##dtype_short##_t C_RANDOM_DATA2; \
    for (int i=0; i<dim; ++i) C_RANDOM_DATA2.v[i] = RANDOM_DATA2(i); \
 \
    char matbuf1[2048] = {0}; \
    print_matrix_##dtype_short( matbuf1, C_RANDOM_DATA.v, 1, dim ); \
    char matbuf2[2048] = {0}; \
    print_matrix_##dtype_short( matbuf2, C_RANDOM_DATA2.v, 1, dim ); \
    mpi_vrb_printf( "C linalg vector %s@%s, #1:\n%s", #dtype_long, #dim, matbuf1 ); \
    mpi_vrb_printf( "C linalg vector %s@%s, #2:\n%s", #dtype_long, #dim, matbuf2 ); \
 \
    test_claV##dim##dtype_short##_t c_zero = test_claV##dim##dtype_short##_zero(); \
    Vec##dim##dtype_short zero = Vec##dim##dtype_short::Zero(); \
    CT_CHECK( (zero - c##dim##dtype_short##Xeigen(c_zero)).norm() < CLA_ZERO ); \
 \
    Vec##dim##dtype_short map = RANDOM_DATA; \
    test_claV##dim##dtype_short##_t c_map = test_claV##dim##dtype_short##_map( map.data() ); \
    CT_CHECK( (map - c##dim##dtype_short##Xeigen(c_map)).norm() < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_rmap = C_RANDOM_DATA; \
    Vec##dim##dtype_short rmap; \
    test_claV##dim##dtype_short##_rmap( c_rmap, rmap.data() ); \
    CT_CHECK( (rmap - c##dim##dtype_short##Xeigen(c_rmap)).norm() < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_scale = C_RANDOM_DATA; \
    CT_CHECK( (c##dim##dtype_short##Xeigen(test_claV##dim##dtype_short##_scale(c_scale,2 )) - c##dim##dtype_short##Xeigen(c_scale)*2).norm() < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_add_1 = C_RANDOM_DATA, c_add_2 = C_RANDOM_DATA2; \
    CT_CHECK( (c##dim##dtype_short##Xeigen(test_claV##dim##dtype_short##_add(c_add_1,c_add_2)) - (c##dim##dtype_short##Xeigen(c_add_1)+c##dim##dtype_short##Xeigen(c_add_2))).norm() < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_sub_1 = C_RANDOM_DATA, c_sub_2 = C_RANDOM_DATA2; \
    CT_CHECK( (c##dim##dtype_short##Xeigen(test_claV##dim##dtype_short##_sub(c_sub_1,c_sub_2)) - (c##dim##dtype_short##Xeigen(c_sub_1)-c##dim##dtype_short##Xeigen(c_sub_2))).norm() < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_odot_1 = C_RANDOM_DATA, c_odot_2 = C_RANDOM_DATA2; \
    CT_CHECK( (c##dim##dtype_short##Xeigen(test_claV##dim##dtype_short##_odot(c_odot_1,c_odot_2)) - (c##dim##dtype_short##Xeigen(c_odot_1).array() * c##dim##dtype_short##Xeigen(c_odot_2).array()).matrix()).norm() < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_dot_1 = C_RANDOM_DATA, c_dot_2 = C_RANDOM_DATA2; \
    CT_CHECK( std::abs(test_claV##dim##dtype_short##_dot(c_dot_1, c_dot_2) - c##dim##dtype_short##Xeigen(c_dot_1).dot(c##dim##dtype_short##Xeigen(c_dot_2))) < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_norm = C_RANDOM_DATA; \
    CT_CHECK( std::abs(test_claV##dim##dtype_short##_norm(c_norm) - sqrt((double)(c##dim##dtype_short##Xeigen(c_norm).squaredNorm()))) < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_norm2 = C_RANDOM_DATA; \
    CT_CHECK( std::abs(test_claV##dim##dtype_short##_norm2(c_norm2) - c##dim##dtype_short##Xeigen(c_norm2).squaredNorm()) < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_cross_1 = C_RANDOM_DATA, c_cross_2 = C_RANDOM_DATA2; \
    CT_CHECK( check_cross<dim>( \
        c##dim##dtype_short##Xeigen(c_cross_1), \
        c##dim##dtype_short##Xeigen(c_cross_2), \
        c##dim##dtype_short##Xeigen(test_claV##dim##dtype_short##_cross(c_cross_1,c_cross_2)) \
        ) ); \
}

// welcome to C++ and its template mess; all of this just to check whether the
// vector type is complex
template <class T, class U>
struct are_same_template : std::is_same<T, U> {};
template <template<class...> class T, class T1, class T2>
struct are_same_template<T<T1>, T<T2>> : std::true_type {};
template <class T, class U> constexpr bool CheckTypes(T, U) {
    return are_same_template<T, U>::value;
}
template<int dim, typename V_eigen> bool check_cross( const V_eigen& a, const V_eigen& b, const V_eigen& c ) {
    if constexpr(dim != 3) {
        return true;
    } else {
        if (CheckTypes<typename V_eigen::Scalar, complex128_t>({},{})) return true;
        return (a.cross(b) - c).norm() < CLA_ZERO;
    }
}

VECTOR_TEST_CASE( 2, double, d )
VECTOR_TEST_CASE( 3, double, d )
VECTOR_TEST_CASE( 4, double, d )
VECTOR_TEST_CASE( 2, complex128_t, cd )
VECTOR_TEST_CASE( 3, complex128_t, cd )
VECTOR_TEST_CASE( 4, complex128_t, cd )
VECTOR_TEST_CASE( 2, int, i )
VECTOR_TEST_CASE( 3, int, i )
VECTOR_TEST_CASE( 4, int, i )
VECTOR_TEST_CASE( 2, int64_t, ll )
VECTOR_TEST_CASE( 3, int64_t, ll )
VECTOR_TEST_CASE( 4, int64_t, ll )

typedef Eigen::Matrix<double,2,2,Eigen::RowMajor> CMat2d;
typedef Eigen::Matrix<double,3,3,Eigen::RowMajor> CMat3d;
typedef Eigen::Matrix<double,4,4,Eigen::RowMajor> CMat4d;
typedef Eigen::Matrix<complex128_t,2,2,Eigen::RowMajor> CMat2cd;
typedef Eigen::Matrix<complex128_t,3,3,Eigen::RowMajor> CMat3cd;
typedef Eigen::Matrix<complex128_t,4,4,Eigen::RowMajor> CMat4cd;
typedef Eigen::Matrix<int,2,2,Eigen::RowMajor> CMat2i;
typedef Eigen::Matrix<int,3,3,Eigen::RowMajor> CMat3i;
typedef Eigen::Matrix<int,4,4,Eigen::RowMajor> CMat4i;
typedef Eigen::Matrix<int64_t,2,2,Eigen::RowMajor> CMat2ll;
typedef Eigen::Matrix<int64_t,3,3,Eigen::RowMajor> CMat3ll;
typedef Eigen::Matrix<int64_t,4,4,Eigen::RowMajor> CMat4ll;

#define RANDOM_DATA_GEN_M2d CMat2d::Random()
#define RANDOM_DATA_GEN_M3d CMat3d::Random()
#define RANDOM_DATA_GEN_M4d CMat4d::Random()
#define RANDOM_DATA_GEN_M2cd CMat2cd::Random()
#define RANDOM_DATA_GEN_M3cd CMat3cd::Random()
#define RANDOM_DATA_GEN_M4cd CMat4cd::Random()
#define RANDOM_DATA_GEN_M2i CMat2i::Random().array().unaryExpr([](int x){return x%100;}).matrix()
#define RANDOM_DATA_GEN_M3i CMat3i::Random().array().unaryExpr([](int x){return x%100;}).matrix()
#define RANDOM_DATA_GEN_M4i CMat4i::Random().array().unaryExpr([](int x){return x%100;}).matrix()
#define RANDOM_DATA_GEN_M2ll CMat2ll::Random().array().unaryExpr([](int64_t x)->int64_t{return x%100;}).matrix()
#define RANDOM_DATA_GEN_M3ll CMat3ll::Random().array().unaryExpr([](int64_t x)->int64_t{return x%100;}).matrix()
#define RANDOM_DATA_GEN_M4ll CMat4ll::Random().array().unaryExpr([](int64_t x)->int64_t{return x%100;}).matrix()

#define MATRIX_TEST_CASE( dim, dtype_long, dtype_short ) \
\
static inline CMat##dim##dtype_short c##dim##dtype_short##Meigen( test_claM##dim##dtype_short##_t vec ) { \
    return Map<CMat##dim##dtype_short>( vec.m[0] ); \
} \
\
CT_TEST_NAME( claM##dim##dtype_short##_t, "cla,M" #dim #dtype_short ) { \
    CMat##dim##dtype_short RANDOM_DATA = RANDOM_DATA_GEN_M##dim##dtype_short ; \
    test_claM##dim##dtype_short##_t C_RANDOM_DATA; \
    for (int i=0; i<dim*dim; ++i) ((dtype_long*)C_RANDOM_DATA.m[0])[i] = RANDOM_DATA.data()[i]; \
 \
    char matbuf1[2048] = {0}; \
    print_matrix_##dtype_short( matbuf1, C_RANDOM_DATA.m[0], dim, dim ); \
    mpi_vrb_printf( "C linalg matrix %s@%s, #1:\n%s", #dtype_long, #dim, matbuf1 ); \
 \
    CMat##dim##dtype_short RANDOM_DATA2 = RANDOM_DATA_GEN_M##dim##dtype_short ; \
    test_claM##dim##dtype_short##_t C_RANDOM_DATA2; \
    for (int i=0; i<dim*dim; ++i) ((dtype_long*)C_RANDOM_DATA2.m[0])[i] = RANDOM_DATA2.data()[i]; \
 \
    char matbuf2[2048] = {0}; \
    print_matrix_##dtype_short( matbuf2, C_RANDOM_DATA2.m[0], dim, dim ); \
    mpi_vrb_printf( "C linalg matrix %s@%s, #2:\n%s", #dtype_long, #dim, matbuf2 ); \
 \
    Vec##dim##dtype_short RANDOM_DATA3 = RANDOM_DATA_GEN_V##dim##dtype_short ; \
    test_claV##dim##dtype_short##_t C_RANDOM_DATA3; \
    for (int i=0; i<dim; ++i) C_RANDOM_DATA3.v[i] = RANDOM_DATA3(i); \
 \
    char matbuf3[2048] = {0}; \
    print_matrix_##dtype_short( matbuf3, C_RANDOM_DATA3.v, 1, dim ); \
    mpi_vrb_printf( "C linalg matvec %s@%s, #3:\n%s", #dtype_long, #dim, matbuf3 ); \
 \
    test_claM##dim##dtype_short##_t c_zero = test_claM##dim##dtype_short##_zero(); \
    CMat##dim##dtype_short zero = CMat##dim##dtype_short::Zero(); \
    CT_CHECK( (zero - c##dim##dtype_short##Meigen(c_zero)).norm() < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_id = test_claM##dim##dtype_short##_id(); \
    CMat##dim##dtype_short id = CMat##dim##dtype_short::Identity(); \
    CT_CHECK( (id - c##dim##dtype_short##Meigen(c_id)).norm() < CLA_ZERO ); \
 \
    CMat##dim##dtype_short map = RANDOM_DATA; \
    test_claM##dim##dtype_short##_t c_map = test_claM##dim##dtype_short##_map( map.data() ); \
    CT_CHECK( (map - c##dim##dtype_short##Meigen(c_map)).norm() < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_rmap = C_RANDOM_DATA; \
    CMat##dim##dtype_short rmap; \
    test_claM##dim##dtype_short##_rmap( c_rmap, rmap.data() ); \
    CT_CHECK( (rmap - c##dim##dtype_short##Meigen(c_rmap)).norm() < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_scale = C_RANDOM_DATA; \
    CT_CHECK( (c##dim##dtype_short##Meigen(test_claM##dim##dtype_short##_scale(c_scale,2 )) - c##dim##dtype_short##Meigen(c_scale)*2).norm() < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_add_1 = C_RANDOM_DATA, c_add_2 = C_RANDOM_DATA2; \
    CT_CHECK( (c##dim##dtype_short##Meigen(test_claM##dim##dtype_short##_add(c_add_1,c_add_2)) - (c##dim##dtype_short##Meigen(c_add_1)+c##dim##dtype_short##Meigen(c_add_2))).norm() < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_sub_1 = C_RANDOM_DATA, c_sub_2 = C_RANDOM_DATA2; \
    CT_CHECK( (c##dim##dtype_short##Meigen(test_claM##dim##dtype_short##_sub(c_sub_1,c_sub_2)) - (c##dim##dtype_short##Meigen(c_sub_1)-c##dim##dtype_short##Meigen(c_sub_2))).norm() < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_odot_1 = C_RANDOM_DATA, c_odot_2 = C_RANDOM_DATA2; \
    CT_CHECK( (c##dim##dtype_short##Meigen(test_claM##dim##dtype_short##_odot(c_odot_1,c_odot_2)) - (c##dim##dtype_short##Meigen(c_odot_1).array() * c##dim##dtype_short##Meigen(c_odot_2).array()).matrix()).norm() < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_dot_1 = C_RANDOM_DATA, c_dot_2 = C_RANDOM_DATA2; \
    CT_CHECK( std::abs(test_claM##dim##dtype_short##_dot(c_dot_1, c_dot_2) - (c##dim##dtype_short##Meigen(c_dot_1).array().conjugate() * c##dim##dtype_short##Meigen(c_dot_2).array()).sum()) < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_norm = C_RANDOM_DATA; \
    CT_CHECK( std::abs(test_claM##dim##dtype_short##_norm(c_norm) - sqrt((double)(c##dim##dtype_short##Meigen(c_norm).squaredNorm()))) < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_norm2 = C_RANDOM_DATA; \
    CT_CHECK( std::abs(test_claM##dim##dtype_short##_norm2(c_norm2) - c##dim##dtype_short##Meigen(c_norm2).squaredNorm()) < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_matmul_1 = C_RANDOM_DATA, c_matmul_2 = C_RANDOM_DATA2; \
    CT_CHECK( (c##dim##dtype_short##Meigen(test_claM##dim##dtype_short##_matmul(c_matmul_1,c_matmul_2)) - \
                (c##dim##dtype_short##Meigen(c_matmul_1) * c##dim##dtype_short##Meigen(c_matmul_2))).norm() < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_trace = C_RANDOM_DATA; \
    CT_CHECK( std::abs(test_claM##dim##dtype_short##_trace(c_trace) - c##dim##dtype_short##Meigen(c_trace).trace()) < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_matvec_v = C_RANDOM_DATA3; \
    test_claM##dim##dtype_short##_t c_matvec_m = C_RANDOM_DATA; \
    Vec##dim##dtype_short matvec_cla = \
        c##dim##dtype_short##Xeigen(test_claM##dim##dtype_short##_matvec( c_matvec_m, c_matvec_v )); \
    Vec##dim##dtype_short matvec_eig = \
        c##dim##dtype_short##Meigen(c_matvec_m) * c##dim##dtype_short##Xeigen(c_matvec_v); \
    /* std::cout << std::endl << matvec_cla << std::endl << std::endl << matvec_eig << std::endl; */ \
    CT_CHECK( (matvec_eig-matvec_cla).norm() < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_vecmat_v = C_RANDOM_DATA3; \
    test_claM##dim##dtype_short##_t c_vecmat_m = C_RANDOM_DATA; \
    Vec##dim##dtype_short vecmat_cla = \
        c##dim##dtype_short##Xeigen(test_claM##dim##dtype_short##_vecmat( c_vecmat_v, c_vecmat_m )); \
    Vec##dim##dtype_short vecmat_eig = \
        c##dim##dtype_short##Meigen(c_vecmat_m).transpose() * c##dim##dtype_short##Xeigen(c_vecmat_v); \
    /* std::cout << std::endl << vecmat_cla << std::endl << std::endl << vecmat_eig << std::endl; */ \
    CT_CHECK( (vecmat_eig-vecmat_cla).norm() < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_transpose = C_RANDOM_DATA; \
    CT_CHECK( (c##dim##dtype_short##Meigen(test_claM##dim##dtype_short##_transpose(c_transpose)) - \
            c##dim##dtype_short##Meigen(c_transpose).transpose()).norm() < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_adjoint = C_RANDOM_DATA; \
    CT_CHECK( (c##dim##dtype_short##Meigen(test_claM##dim##dtype_short##_adjoint(c_adjoint)) - \
            c##dim##dtype_short##Meigen(c_adjoint).adjoint()).norm() < CLA_ZERO ); \
}

MATRIX_TEST_CASE( 2, double, d )
MATRIX_TEST_CASE( 3, double, d )
MATRIX_TEST_CASE( 4, double, d )
MATRIX_TEST_CASE( 2, complex128_t, cd )
MATRIX_TEST_CASE( 3, complex128_t, cd )
MATRIX_TEST_CASE( 4, complex128_t, cd )
MATRIX_TEST_CASE( 2, int, i )
MATRIX_TEST_CASE( 3, int, i )
MATRIX_TEST_CASE( 4, int, i )
MATRIX_TEST_CASE( 2, int64_t, ll )
MATRIX_TEST_CASE( 3, int64_t, ll )
MATRIX_TEST_CASE( 4, int64_t, ll )

#define MATRIX_DECOMPOSITION_TEST_CASE( dim, dtype_long, dtype_short ) \
\
CT_TEST_NAME( claM##dim##dtype_short##_decomp, "cla,decompose,M" #dim #dtype_short ) { \
    CMat##dim##dtype_short RANDOM_DATA0 = RANDOM_DATA_GEN_M##dim##dtype_short ; \
    CMat##dim##dtype_short RANDOM_DATA = RANDOM_DATA0 + RANDOM_DATA0.adjoint(); \
    test_claM##dim##dtype_short##_t C_RANDOM_DATA; \
    for (int i=0; i<dim*dim; ++i) ((dtype_long*)C_RANDOM_DATA.m[0])[i] = RANDOM_DATA.data()[i]; \
 \
    char matbuf1[2048] = {0}; \
    print_matrix_##dtype_short( matbuf1, C_RANDOM_DATA.m[0], dim, dim ); \
    mpi_vrb_printf( "C decomp matrix %s@%s, #1:\n%s", #dtype_long, #dim, matbuf1 ); \
 \
    CMat##dim##dtype_short RANDOM_DATA2 = RANDOM_DATA_GEN_M##dim##dtype_short ; \
    test_claM##dim##dtype_short##_t C_RANDOM_DATA2; \
    for (int i=0; i<dim*dim; ++i) ((dtype_long*)C_RANDOM_DATA2.m[0])[i] = RANDOM_DATA2.data()[i]; \
 \
    char matbuf2[2048] = {0}; \
    print_matrix_##dtype_short( matbuf2, C_RANDOM_DATA2.m[0], dim, dim ); \
    mpi_vrb_printf( "C decomp matrix %s@%s, #2:\n%s", #dtype_long, #dim, matbuf2 ); \
 \
    test_claM##dim##dtype_short##_t c_eigvalsh_H = C_RANDOM_DATA; \
    test_claV##dim##d_t c_eigvalsh_E = test_claM##dim##dtype_short##_eigvalsh( c_eigvalsh_H ); \
    auto S = eigensolve_##dim##dtype_short( RANDOM_DATA ); \
    CT_CHECK( (c##dim##dXeigen(c_eigvalsh_E) - S.E).norm() < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_eigh_H = C_RANDOM_DATA; \
    test_claM##dim##dtype_short##_t c_eigh_U, c_eigh_UE; \
    test_claV##dim##d_t c_eigh_E = test_claM##dim##dtype_short##_eigh( c_eigh_H, &c_eigh_U ); \
    CT_CHECK( (c##dim##dXeigen(c_eigh_E) - S.E).norm() < CLA_ZERO ); \
    c_eigh_UE = c_eigh_U; \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        c_eigh_UE.m[i][j] *= c_eigh_E.v[j]; \
    CT_CHECK( test_claM##dim##dtype_short##_norm( test_claM##dim##dtype_short##_sub( \
            test_claM##dim##dtype_short##_matmul( c_eigh_UE, test_claM##dim##dtype_short##_adjoint(c_eigh_U) ), \
            c_eigh_H ) ) < CLA_ZERO ); \
 \
    test_claM##dim##dtype_short##_t c_svd_A = C_RANDOM_DATA2; \
    test_claM##dim##dtype_short##_t c_svd_U, c_svd_V; \
    test_claV##dim##d_t c_svd_E = test_claM##dim##dtype_short##_svd( c_svd_A, &c_svd_U, &c_svd_V ); \
    Vec##dim##d singularValues = svd_##dim##dtype_short( RANDOM_DATA2 ); \
    CT_CHECK( (c##dim##dXeigen(c_svd_E) - singularValues).norm() < CLA_ZERO ); \
    for (int i=0; i<dim; ++i) \
    for (int j=0; j<dim; ++j) \
        c_svd_U.m[i][j] *= c_svd_E.v[j]; \
    CT_CHECK( test_claM##dim##dtype_short##_norm( test_claM##dim##dtype_short##_sub( \
            test_claM##dim##dtype_short##_matmul( c_svd_U, c_svd_V ), \
            c_svd_A ) ) < CLA_ZERO ); \
}

MATRIX_DECOMPOSITION_TEST_CASE( 2, double, d )
MATRIX_DECOMPOSITION_TEST_CASE( 3, double, d )
MATRIX_DECOMPOSITION_TEST_CASE( 4, double, d )
MATRIX_DECOMPOSITION_TEST_CASE( 2, complex128_t, cd )
MATRIX_DECOMPOSITION_TEST_CASE( 3, complex128_t, cd )
MATRIX_DECOMPOSITION_TEST_CASE( 4, complex128_t, cd )

#endif // DIVERGE_SKIP_TESTS_LINALG
#endif // DIVERGE_SKIP_TESTS
