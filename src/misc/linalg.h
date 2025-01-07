/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#ifdef __cplusplus
#include <complex>
#include <cstdint>
typedef std::complex<double> complex128_t;
#else
#include <complex.h>
#include <stdint.h>
typedef _Complex double complex128_t;
#endif

#define LINALG_DATA_TYPE_d double
#define LINALG_DATA_TYPE_cd complex128_t
#define LINALG_DATA_TYPE_i int
#define LINALG_DATA_TYPE_ll int64_t

#ifndef LINALG_EXPORT
#define LINALG_EXPORT static inline
#endif

#ifndef LINALG_PREFIX
#define LINALG_PREFIX cla
#endif

/**
 * Linear Algebra for low dimensions in C
 * ======================================
 *
 * In many parts of divERGe, linear algebra for low-dimensional vector spaces is
 * required. We provide a collection of functions (inline accessible from C,
 * compiled accessible from both C and C++) that are implemented through a
 * macro. So, in your source code, you have to call the implementation macro
 * after including linalg.h:
 *
 * .. sourcecode:: C
 *
 *  #include "linalg.h"
 *  LINALG_IMPLEMENT
 *
 * This gives you basic linear algebra functionality for double (``d``), complex
 * double (``cd``), integer (``i``) and long long integer (``ll``) vectors and
 * matrices in 2, 3, and 4 dimensions. More dimensions can be implemented with
 *
 * .. sourcecode:: C
 *
 *  LINALG_IMPL( LINALG_PREFIX, 7 )
 *
 * where ``LINALG_PREFIX`` is expanded to ``cla`` (other prefices may be used by
 * either defining  ``LINALG_PREFIX`` or giving something else to
 * ``LINALG_IMPL``), and ``7`` represents the dimension. If you want to only
 * declare functions and not implement them (for e.g. C++), you can write
 *
 * .. sourcecode:: C++
 *
 *  LINALG_TYPES( LINALG_PREFIX, 7 )
 *
 * This statement declares linear algebra functions in dimension 7. Notably, the
 * default symbol visiblity is set to ``LINALG_EXPORT``, which expands to
 * ``static inline``. If you want to use the functions in C++, you have to
 * redefine ``LINALG_EXPORT`` to something else (how about ``extern "C"``?) in
 * your C++ file that includes linalg.h, as well as in your C file that includes
 * linalg.h in order to *define* the functions declared to C++.
 *
 * For enhanced portability, we provide the option to *disable* decomposition
 * support (that relies on LAPACKE). When ``LINALG_NO_LAPACKE`` is defined, the
 * decomposition functions are declared, but their definition is empty.
 *
 * The funcion names are made from a prefix ``P``, a vector/matrix indicator
 * ``X`` (can be ``V`` or ``M``), a dimension indicator ``D`` (an integer), and
 * a datatype indicator ``T`` (the four datatype indicators are ``d``:
 * ``double``, ``cd``: :c:type:`complex128_t`, ``i``: ``int``, ``ll``:
 * ``int64_t``). The convention then is ``PXDT``. For example, if the prefix is
 * ``cla`` (the default), all functions operating on three-dimensional double
 * vectors begin with ``claV3d``, and the three dimensional vector data type is
 * called ``claV3d_t``.
 *
 * Each macro invokation of ``LINALG_IMPL`` or ``LINALG_TYPES`` declares/defines
 * the following types and functions:
 *
 * Vectors: ``PVDT_t``
 * -------------------
 * The structure ``PVDT_t`` (e.g. ``claV3d_t`` for a three-dimensional vector
 * of doubles) has one static array of datatype ``T`` as member: ``T v[D]``.
 * This member holds the vector's data. The following vector functions are
 * available:
 *
 *   * ``PVDT_t PVDT_zero( void )``: generate a zero vector
 *   * ``PVDT_t PVDT_const( T val )``: generate a constant vector, where each entry is set to ``val``
 *   * ``PVDT_t PVDT_map( const T* p )``: map data to a vector (copy)
 *   * ``void PVDT_rmap( PVDT_t v, T* p )``: map vector to data (copy)
 *   * ``PVDT_t PVDT_scale( PVDT_t v, T s )``: return :math:`\boldsymbol{v} \, s`
 *   * ``PVDT_t PVDT_add( PVDT_t a, PVDT_t b )``: return :math:`\boldsymbol{a} + \boldsymbol{b}`
 *   * ``PVDT_t PVDT_sub( PVDT_t a, PVDT_t b )``: return :math:`\boldsymbol{a} - \boldsymbol{b}`
 *   * ``PVDT_t PVDT_conj( PVDT_t a )``: return :math:`\boldsymbol{a}^*`
 *   * ``PVDT_t PVDT_odot( PVDT_t a, PVDT_t b )``: return the element-wise product :math:`\boldsymbol{a}
 *     \circ \boldsymbol{b}`
 *   * ``T PVDT_dot( PVDT_t a, PVDT_t b )``: scalar product :math:`\boldsymbol{a} \cdot \boldsymbol{b}`
 *   * ``double PVDT_norm2( PVDT_t a )``: squared norm :math:`\boldsymbol{a} \cdot \boldsymbol{a}^*`
 *   * ``double PVDT_norm( PVDT_t a )``: square root of squared norm.
 *   * ``PVDT_t PVDT_cross( PVDT_t a, PVDT_t b )``: return the cross product
 *     :math:`\boldsymbol{a} \times \boldsymbol{b}`
 *
 * Matrices: ``PMDT_t``
 * --------------------
 * The structure ``PMDT_t`` (e.g. ``claM3d_t`` for a :math:`3\times3` matrix of
 * doubles) has one two-dimensional static array of datatype ``T`` as member:
 * ``T m[D][D]``. This member holds the matrix' data. The following matrix
 * (-vector) functions are available:
 *
 *   * ``PMDT_t PMDT_zero( void )``: zero-init a matrix
 *   * ``PMDT_t PMDT_id( void )``: indentity-init a matrix
 *   * ``PMDT_t PMDT_map( const T* p )``: map data to a matrix (copy)
 *   * ``void PMDT_rmap( PMDT_t m, T * p )``: map matrix to data (copy)
 *   * ``PMDT_t PMDT_scale( PMDT_t v, T s )``: return :math:`\hat{v} \, s`
 *   * ``PMDT_t PMDT_add( PMDT_t a, PMDT_t b )``: return :math:`\hat{a} + \hat{b}`
 *   * ``PMDT_t PMDT_sub( PMDT_t a, PMDT_t b )``: return :math:`\hat{a} - \hat{b}`
 *   * ``PMDT_t PMDT_conj( PMDT_t a )``: return :math:`\hat{a}^*`
 *   * ``PMDT_t PMDT_odot( PMDT_t a, PMDT_t b )``: return :math:`\hat{a} \circ \hat{b}`
 *   * ``T PMDT_dot( PMDT_t a, PMDT_t b )``: return :math:`\mathrm{Tr}(\hat{a} \circ \hat{b})`
 *   * ``double PMDT_norm2( PMDT_t a )``: return squared norm :math:`\mathrm{Tr}(\hat{a} \circ \hat{a}^*)`
 *   * ``double PMDT_norm( PMDT_t a )``: return square root of squared norm
 *   * ``PMDT_t PMDT_matmul( PMDT_t a, PMDT_t b )``: return matrix product :math:`\hat{a} \cdot \hat{b}`
 *   * ``T PMDT_trace( PMDT_t a )``: return trace :math:`\mathrm{Tr}(\hat a)`
 *   * ``PVDT_t PMDT_matvec( PMDT_t a, PVDT_t v )``: return matrix vector product
 *     :math:`\hat{a} \cdot \boldsymbol{v}`
 *   * ``PMDT_t PMDT_transpose( PMDT_t a )``: return transposed matrix :math:`\hat{a}^T`
 *   * ``PMDT_t PMDT_adjoint( PMDT_t a )``: return adjoint matrix :math:`\hat{a}^{T*}`
 *   * ``PVDT_t PMDT_vecmat( PVDT_t v, PMDT_t a )``: return vector matrix product
 *     :math:`(\boldsymbol{v}^T \cdot \hat{a})^T`
 *   * ``PVDd_t PMDT_eigvalsh( PMDT_t H )``: return eigenvalues of Hermitian
 *     matrix :math:`\hat{H}` (only floating point)
 *   * ``PVDd_t PMDT_eigh( PMDT_t H, PMDT_t* pU )``: return eigenvalues of Hermitian
 *     matrix :math:`\hat{H}` and eigenvectors in ``pU``.
 *   * ``PVDd_t PMDT_svd( PMDT_t H, PMDT_t* pU, PMDT_t* pVT )``: perform an SVD
 *     of the matrix :math:`\hat{H} = \hat{U} \cdot
 *     \mathrm{diag}(\boldsymbol{S}) \cdot \hat{V}^T`. return singular values
 *     :math:`\boldsymbol{S}` (floating point) and left singular vectors
 *     :math:`\hat{U}` (``pU``) as well as right singular vectors
 *     :math:`\hat{V}^T` (``pVT``).
 *
 * To get used to this mini-library, you can have a look at e.g.
 * src/misc/hoppings_to_supercell.c (mostly integers, i.e., ``claV3i_t``) or
 * src/diverge_symmetrize_generate_symm_maps.c (3 dimensional double vectors,
 * i.e., ``claV3d_t``). It feels a bit like stack-only Eigen3, but for C (so
 * functions and no overloads).
 */

#ifndef LINALG_IMPLEMENT
#define LINALG_IMPLEMENT \
    LINALG_IMPL( LINALG_PREFIX, 2 ) \
    LINALG_IMPL( LINALG_PREFIX, 3 ) \
    LINALG_IMPL( LINALG_PREFIX, 4 )
#endif

#define LINALG_TYPES( cla_prefix, dim ) \
    LINALG_VEC_TYPES( cla_prefix, dim, d ) \
    LINALG_VEC_TYPES( cla_prefix, dim, cd ) \
    LINALG_VEC_TYPES( cla_prefix, dim, i ) \
    LINALG_VEC_TYPES( cla_prefix, dim, ll )

#define LINALG_IMPL( cla_prefix, dim ) \
    LINALG_TYPES( cla_prefix, dim ) \
    LINALG_VEC_IMPL( cla_prefix, dim, d ) \
    LINALG_VEC_IMPL( cla_prefix, dim, cd ) \
    LINALG_VEC_IMPL( cla_prefix, dim, i ) \
    LINALG_VEC_IMPL( cla_prefix, dim, ll )

#define LINALG_VEC_TYPES( cla_prefix, dim, type ) \
typedef struct cla_prefix##V##dim##type##_t { \
    LINALG_DATA_TYPE_##type v[dim]; \
} cla_prefix##V##dim##type##_t; \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_zero( void ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_const( LINALG_DATA_TYPE_##type val ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_map( const LINALG_DATA_TYPE_##type * p ); \
LINALG_EXPORT void cla_prefix##V##dim##type##_rmap( cla_prefix##V##dim##type##_t v, LINALG_DATA_TYPE_##type * p ); \
LINALG_EXPORT int cla_prefix##V##dim##type##_equal( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_scale( cla_prefix##V##dim##type##_t v, LINALG_DATA_TYPE_##type s ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_add( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_sub( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_conj( cla_prefix##V##dim##type##_t a ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_odot( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ); \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla_prefix##V##dim##type##_dot( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ); \
LINALG_EXPORT double cla_prefix##V##dim##type##_norm2( cla_prefix##V##dim##type##_t a ); \
LINALG_EXPORT double cla_prefix##V##dim##type##_norm( cla_prefix##V##dim##type##_t a ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##V##dim##type##_cross( cla_prefix##V##dim##type##_t a, cla_prefix##V##dim##type##_t b ); \
\
typedef struct cla_prefix##M##dim##type##_t { \
    LINALG_DATA_TYPE_##type m [dim][dim]; \
} cla_prefix##M##dim##type##_t; \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_zero( void ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_id( void ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_map( const LINALG_DATA_TYPE_##type * p ); \
LINALG_EXPORT void cla_prefix##M##dim##type##_rmap( cla_prefix##M##dim##type##_t m, LINALG_DATA_TYPE_##type * p ); \
LINALG_EXPORT int cla_prefix##M##dim##type##_equal( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_scale( cla_prefix##M##dim##type##_t v, LINALG_DATA_TYPE_##type s ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_add( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_sub( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_conj( cla_prefix##M##dim##type##_t a ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_odot( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ); \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla_prefix##M##dim##type##_dot( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ); \
LINALG_EXPORT double cla_prefix##M##dim##type##_norm2( cla_prefix##M##dim##type##_t a ); \
LINALG_EXPORT double cla_prefix##M##dim##type##_norm( cla_prefix##M##dim##type##_t a ); \
\
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_matmul( cla_prefix##M##dim##type##_t a, cla_prefix##M##dim##type##_t b ); \
LINALG_EXPORT LINALG_DATA_TYPE_##type cla_prefix##M##dim##type##_trace( cla_prefix##M##dim##type##_t a ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##M##dim##type##_matvec( cla_prefix##M##dim##type##_t a, cla_prefix##V##dim##type##_t v ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_transpose( cla_prefix##M##dim##type##_t a ); \
LINALG_EXPORT cla_prefix##M##dim##type##_t cla_prefix##M##dim##type##_adjoint( cla_prefix##M##dim##type##_t a ); \
LINALG_EXPORT cla_prefix##V##dim##type##_t cla_prefix##M##dim##type##_vecmat( cla_prefix##V##dim##type##_t v, cla_prefix##M##dim##type##_t a ); \
LINALG_EXPORT cla_prefix##V##dim##d_t cla_prefix##M##dim##type##_eigvalsh( cla_prefix##M##dim##type##_t H ); \
LINALG_EXPORT cla_prefix##V##dim##d_t cla_prefix##M##dim##type##_eigh( cla_prefix##M##dim##type##_t H, cla_prefix##M##dim##type##_t* pU ); \
LINALG_EXPORT cla_prefix##V##dim##d_t cla_prefix##M##dim##type##_svd( cla_prefix##M##dim##type##_t H, cla_prefix##M##dim##type##_t* pU, cla_prefix##M##dim##type##_t* pVT );

#ifndef __cplusplus
#include "linalg_impl.h"
#endif
