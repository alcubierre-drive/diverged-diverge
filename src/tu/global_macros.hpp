/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once
#include "../diverge_common.h"
#include <numeric>
#include <vector>

#define DELETE_CONSTRUCTOR_RULE_OF_FIVE(CLASSNAME) \
        CLASSNAME() = delete; \
        CLASSNAME(const CLASSNAME&) = delete; \
        CLASSNAME& operator=(const CLASSNAME&) = delete; \
        CLASSNAME (CLASSNAME &&) = delete; \
        CLASSNAME& operator= (CLASSNAME &&) = delete


template <typename cmpl_x_1, typename cmpl_x_2>
constexpr inline bool approxEqual(cmpl_x_1 a, cmpl_x_2 b, double absEpsilon = absEps,
        double relEpsilon = relEps)
{
    return std::abs(a - b) <= std::max( absEpsilon, relEpsilon*std::max(
                (double)std::abs(a), (double)std::abs(b)));
}

struct vec3i_cmp {
    vec3i_cmp(Vec3i v)
    :val{v}{ }

    inline bool operator()(const Vec3i x) const {
        return (x-val).cwiseAbs().sum() == 0;
    }
private:
    const Vec3i val;
};

struct vec5i_cmp {
    vec5i_cmp(Vec5i v)
    :val{v}{ }

    inline bool operator()(const Vec5i x) const {
        return (x-val).cwiseAbs().sum() == 0;
    }
private:
    const Vec5i val;
};

struct vec_cmp {
    vec_cmp(Eigen::VectorXd v, double diff = 1e-3)
    :
        val{v},delta{diff}
    { }
    inline bool operator()(const Eigen::VectorXd x) const {
        return (x-val).cwiseAbs().sum()< delta;
    }
private:
    const Eigen::VectorXd val;
    const double delta;
};

struct cdbl_cmp {
    cdbl_cmp(complex128_t v)
    :
        val(v)
    { }
    inline bool operator()(const complex128_t x) const {
        return abs(x-val) < delta;
    }
private:
    complex128_t val;
    double delta = 1e-10;
};

#pragma omp declare reduction(+ : complex128_t : \
             omp_out += omp_in)


template <typename T>
inline double max_abs(const T* const start, const long length)
{
    return (double)std::abs(
                *std::max_element(start, start+length,
                [](T A, T B){return std::abs(A)<std::abs(B);})
            );
}


template <typename T>
vector<uindex_t> sort_permutation(
    const vector<T>& vec)
{
    vector<uindex_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
        [&](uindex_t i, uindex_t j){ return abs(vec[i])>abs(vec[j]); });
    return p;
}

template <typename T>
vector<uindex_t> sort_permutation_increasing(
    const vector<T>& vec)
{
    vector<uindex_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
        [&](uindex_t i, uindex_t j){ return abs(vec[i])<abs(vec[j]); });
    return p;
}


template <typename T, typename S>
void apply_permutation_in_place(
    vector<T>& vec,
    const vector<S>& p)
{
    vector<bool> done(vec.size());
    for (uindex_t i = 0; i < vec.size(); ++i)
    {
        if (done[i])
        {
            continue;
        }
        done[i] = true;
        uindex_t prev_j = i;
        uindex_t j = p[i];
        while (i != j)
        {
            std::swap(vec[prev_j], vec[j]);

            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}
