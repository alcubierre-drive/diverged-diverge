#pragma once

#include "../../src/diverge_common.h"
#include "../../src/diverge_Eigen3.hpp"

typedef Eigen::Matrix<double,2,2,Eigen::RowMajor> CMat2d;
typedef Eigen::Matrix<double,3,3,Eigen::RowMajor> CMat3d;
typedef Eigen::Matrix<double,4,4,Eigen::RowMajor> CMat4d;
typedef Eigen::Matrix<complex128_t,2,2,Eigen::RowMajor> CMat2cd;
typedef Eigen::Matrix<complex128_t,3,3,Eigen::RowMajor> CMat3cd;
typedef Eigen::Matrix<complex128_t,4,4,Eigen::RowMajor> CMat4cd;

typedef struct { CMat2d U; Vec2d E; } EigenSol2d;
typedef struct { CMat3d U; Vec3d E; } EigenSol3d;
typedef struct { CMat4d U; Vec4d E; } EigenSol4d;
typedef struct { CMat2cd U; Vec2d E; } EigenSol2cd;
typedef struct { CMat3cd U; Vec3d E; } EigenSol3cd;
typedef struct { CMat4cd U; Vec4d E; } EigenSol4cd;

EigenSol2d eigensolve_2d( CMat2d H );
EigenSol3d eigensolve_3d( CMat3d H );
EigenSol4d eigensolve_4d( CMat4d H );
EigenSol2cd eigensolve_2cd( CMat2cd H );
EigenSol3cd eigensolve_3cd( CMat3cd H );
EigenSol4cd eigensolve_4cd( CMat4cd H );

Vec2d svd_2d( CMat2d H );
Vec3d svd_3d( CMat3d H );
Vec4d svd_4d( CMat4d H );
Vec2d svd_2cd( CMat2cd H );
Vec3d svd_3cd( CMat3cd H );
Vec4d svd_4cd( CMat4cd H );
