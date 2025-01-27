/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifndef LEAVE_EIGEN_BROKEN

#ifndef EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_PARALLELIZE
#endif

#ifndef EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_VECTORIZE
#endif

#ifndef EIGEN_MAX_ALIGN_BYTES
#define EIGEN_MAX_ALIGN_BYTES 0
#endif

#ifndef EIGEN_MAX_STATIC_ALIGN_BYTES
#define EIGEN_MAX_STATIC_ALIGN_BYTES 0
#endif

#endif // LEAVE_EIGEN_BROKEN

#ifndef SKIP_EIGEN3_INCLUDE_PREFIX
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif

#include <complex>
typedef Eigen::Vector2i Vec2i;
typedef Eigen::Vector3i Vec3i;
typedef Eigen::Vector4i Vec4i;
typedef Eigen::Matrix<int,5,1> Vec5i;

typedef Eigen::Vector2d Vec2d;
typedef Eigen::Vector3d Vec3d;
typedef Eigen::Vector4d Vec4d;
typedef Eigen::VectorXd VecXd;
typedef Eigen::Matrix2d Mat2d;
typedef Eigen::Matrix3d Mat3d;
typedef Eigen::Matrix4d Mat4d;
typedef Eigen::MatrixXd MatXd;

typedef Eigen::Vector2cd Vec2cd;
typedef Eigen::Vector3cd Vec3cd;
typedef Eigen::Vector4cd Vec4cd;
typedef Eigen::VectorXcd VecXcd;
typedef Eigen::Matrix2cd Mat2cd;
typedef Eigen::Matrix3cd Mat3cd;
typedef Eigen::Matrix4cd Mat4cd;
typedef Eigen::MatrixXcd MatXcd;
typedef Eigen::Matrix<std::complex<double>,-1,-1,Eigen::RowMajor> cMatXcd;

typedef Eigen::Rotation2Dd Rot2d;

using Eigen::Map;
using Eigen::Matrix;
