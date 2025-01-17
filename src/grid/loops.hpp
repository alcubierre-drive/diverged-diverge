/* Copyright (C) 2025 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "typedefs.hpp"

typedef struct diverge_model_t diverge_model_t;
typedef struct grid_symm_transform_t grid_symm_transform_t;

namespace grid {

class Loops {
    public:
        Loops( diverge_model_t* model );
        ~Loops();
        complex128_t* X_P(double Lambda);
        complex128_t* X_C(double Lambda);

        void toggle_symmetrizer( void ) { _SYM_enabled = !_SYM_enabled; }

        index_t get_qstart() const { return _qstart; }
        index_t get_qstop() const { return _qstart+_qsize; }
        index_t get_qsize() const { return _qsize; }

        array<double,2> get_max() { return {_max_P, _max_C}; }

        double* get_kfweights(void) { return _kfweights; }
        index_t get_nkf(void) { return _nkf; }
        index_t get_nk(void) { return _nk; }
        index_t get_nb(void) { return _nb; }
        index_t get_kzero(void);

        index_t get_nspin(void);

        complex128_t* get_ff(void) { return _formfac; }
        index_t get_nff(void) { return _nff; }

    private:
        complex128_t* symmetrize( complex128_t* loop, char chan );
        grid_symm_transform_t* _SYM = nullptr;
        bool _SYM_enabled = true;

        complex128_t* _X_P;
        complex128_t* _X_C;
        double _max_P = 0.0;
        double _max_C = 0.0;
        index_t _qstart, _qsize;
        index_t _nk, _nb, _Xsize;

        index_t _nkf, *_kfrev, *_k1m2;
        double* _kfweights;
        complex128_t* _formfac;
        index_t _nff;

        void _greens_cache_fill(double Lambda);
        double _greens_cache_lambda = 0;

        diverge_model_t* _dmod = nullptr;
        complex128_t* _greens_cache_p = nullptr;
        complex128_t* _greens_cache_m = nullptr;
        complex128_t* _greens_proj_cache_p = nullptr;
        complex128_t* _greens_proj_cache_m = nullptr;

};

}
