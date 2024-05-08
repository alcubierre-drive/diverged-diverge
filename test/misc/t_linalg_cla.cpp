#ifndef DIVERGE_SKIP_TESTS

#include "../catch.hpp"
#include "../../src/diverge_common.h"
#include "../../src/diverge_Eigen3.hpp"

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

#define VECTOR_TEST_CASE( dim, dtype_long, dtype_short ) \
\
static inline Vec##dim##dtype_short c##dim##dtype_short##Xeigen( test_claV##dim##dtype_short##_t vec ) { \
    return Map<Vec##dim##dtype_short>( vec.v ); \
} \
\
TEST_CASE("C linalg vector " #dtype_long "@" #dim, "[cla][V" #dim #dtype_short "]") { \
    Vec##dim##dtype_short RANDOM_DATA = RANDOM_DATA_GEN_V##dim##dtype_short ; \
    test_claV##dim##dtype_short##_t C_RANDOM_DATA; \
    for (int i=0; i<dim; ++i) C_RANDOM_DATA.v[i] = RANDOM_DATA(i); \
 \
    Vec##dim##dtype_short RANDOM_DATA2 = RANDOM_DATA_GEN_V##dim##dtype_short ; \
    test_claV##dim##dtype_short##_t C_RANDOM_DATA2; \
    for (int i=0; i<dim; ++i) C_RANDOM_DATA2.v[i] = RANDOM_DATA2(i); \
 \
    std::stringstream ss; \
    ss.precision( 1 ); \
    ss << "[" << RANDOM_DATA.transpose() << "]; [" << RANDOM_DATA2.transpose() << "]"; \
    std::string s = ss.str(); \
    mpi_log_printf( "CLA %s\n", s.c_str() ); \
 \
    test_claV##dim##dtype_short##_t c_zero = test_claV##dim##dtype_short##_zero(); \
    Vec##dim##dtype_short zero = Vec##dim##dtype_short::Zero(); \
    CHECK( (zero - c##dim##dtype_short##Xeigen(c_zero)).norm() < CLA_ZERO ); \
 \
    Vec##dim##dtype_short map = RANDOM_DATA; \
    test_claV##dim##dtype_short##_t c_map = test_claV##dim##dtype_short##_map( map.data() ); \
    CHECK( (map - c##dim##dtype_short##Xeigen(c_map)).norm() < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_rmap = C_RANDOM_DATA; \
    Vec##dim##dtype_short rmap; \
    test_claV##dim##dtype_short##_rmap( c_rmap, rmap.data() ); \
    CHECK( (rmap - c##dim##dtype_short##Xeigen(c_rmap)).norm() < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_scale = C_RANDOM_DATA; \
    CHECK( (c##dim##dtype_short##Xeigen(test_claV##dim##dtype_short##_scale(c_scale,2 )) - c##dim##dtype_short##Xeigen(c_scale)*2).norm() < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_add_1 = C_RANDOM_DATA, c_add_2 = C_RANDOM_DATA2; \
    CHECK( (c##dim##dtype_short##Xeigen(test_claV##dim##dtype_short##_add(c_add_1,c_add_2)) - (c##dim##dtype_short##Xeigen(c_add_1)+c##dim##dtype_short##Xeigen(c_add_2))).norm() < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_sub_1 = C_RANDOM_DATA, c_sub_2 = C_RANDOM_DATA2; \
    CHECK( (c##dim##dtype_short##Xeigen(test_claV##dim##dtype_short##_sub(c_sub_1,c_sub_2)) - (c##dim##dtype_short##Xeigen(c_sub_1)-c##dim##dtype_short##Xeigen(c_sub_2))).norm() < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_odot_1 = C_RANDOM_DATA, c_odot_2 = C_RANDOM_DATA2; \
    CHECK( (c##dim##dtype_short##Xeigen(test_claV##dim##dtype_short##_odot(c_odot_1,c_odot_2)) - (c##dim##dtype_short##Xeigen(c_odot_1).array() * c##dim##dtype_short##Xeigen(c_odot_2).array()).matrix()).norm() < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_dot_1 = C_RANDOM_DATA, c_dot_2 = C_RANDOM_DATA2; \
    CHECK( std::abs(test_claV##dim##dtype_short##_dot(c_dot_1, c_dot_2) - c##dim##dtype_short##Xeigen(c_dot_1).dot(c##dim##dtype_short##Xeigen(c_dot_2))) < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_norm = C_RANDOM_DATA; \
    CHECK( std::abs(test_claV##dim##dtype_short##_norm(c_norm) - sqrt((double)(c##dim##dtype_short##Xeigen(c_norm).squaredNorm()))) < CLA_ZERO ); \
 \
    test_claV##dim##dtype_short##_t c_norm2 = C_RANDOM_DATA; \
    CHECK( std::abs(test_claV##dim##dtype_short##_norm2(c_norm2) - c##dim##dtype_short##Xeigen(c_norm).squaredNorm()) < CLA_ZERO ); \
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

#endif // DIVERGE_SKIP_TESTS
