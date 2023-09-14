/* Copyright (C) 2023 Jonas B. Hauck, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once

#include "typedefs.hpp"

typedef struct diverge_model_t diverge_model_t;

namespace grid {

class LoopSymmetrizer;

class Loops {
    public:
        Loops( diverge_model_t* model );
        ~Loops();
        complex128_t* X_P(double Lambda);
        complex128_t* X_C(double Lambda);
        /* FRG_Data* get_frg_info() const { return _frg; } */
        index_t get_qstart() const { return _qstart; }
        index_t get_qstop() const { return _qstop; }
        index_t get_qsize() const { return _qstop-_qstart; }

        LoopSymmetrizer* get_symmetrizer(void) { return _SYM; }
        void set_symmetrizer( LoopSymmetrizer* S ) { _SYM = S; }
        complex128_t* symmetrize( complex128_t* loop, char chan );

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
        LoopSymmetrizer* _SYM = nullptr;
        bool _SYM_free = false;
        complex128_t* _X_P;
        complex128_t* _X_C;
        double _max_P = 0.0;
        double _max_C = 0.0;
        index_t _qstart, _qstop;
        index_t _nk, _nb, _qsize, _Xsize;

        index_t _nkf, *_kfrev, *_k1m2;
        double* _kfweights;
        bool _formfac_free = false;
        complex128_t* _formfac;
        index_t _nff;

        void _greens_cache_fill(double Lambda);
        double _greens_cache_lambda = 0;

        diverge_model_t* _dmod = nullptr;
        complex128_t* _greens_cache_p = nullptr;
        complex128_t* _greens_cache_m = nullptr;

};

}
