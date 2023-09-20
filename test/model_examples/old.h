#ifndef DIVERGE_SKIP_TESTS

#pragma once

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

typedef struct diverge_model_t diverge_model_t;

diverge_model_t* model_examples_hubbard( bool grid );
diverge_model_t* model_examples_kagome( double U, double V );
diverge_model_t* model_examples_graphene( void );

#ifdef __cplusplus
}
#endif

#endif // DIVERGE_SKIP_TESTS
