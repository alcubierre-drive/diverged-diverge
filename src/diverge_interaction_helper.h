#pragma once

#include "diverge_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Function: init_P_channel
// functions return whether buf was filled (1) or left as-is (0)
//
// buf is filled with the realspace vertices as defined in the model
//
// use as helper functions when creating a model on your own.
int init_P_channel(const diverge_model_t* model, complex128_t* buf);
// Function: init_C_channel
int init_C_channel(const diverge_model_t* model, complex128_t* buf);
// Function: init_D_channel
int init_D_channel(const diverge_model_t* model, complex128_t* buf);

#ifdef __cplusplus
}
#endif

