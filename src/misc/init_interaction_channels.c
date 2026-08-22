/* Copyright (C) 2026 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#include "init_interaction_channels.h"
#include "../diverge_internals_struct.h"
#include <assert.h>

static inline void init_P_channel_nonSU2(const diverge_model_t* model, complex128_t* buf);
static inline void init_C_channel_nonSU2(const diverge_model_t* model, complex128_t* buf);
static inline void init_D_channel_nonSU2(const diverge_model_t* model, complex128_t* buf);
static inline void init_channel_SU2( const diverge_model_t* model, complex128_t* buf, index_t offset, index_t n_elem );

int init_P_channel(const diverge_model_t* model, complex128_t* buf) {
    if (model->n_vert_chan[2] == 0)
        return 0;
    if (model->SU2 > 0)
        init_channel_SU2(model,buf,model->n_vert_chan[0] + model->n_vert_chan[1],model->n_vert_chan[2]);
    else {
        init_P_channel_nonSU2(model,buf);
    }
    return 1;
}

int init_C_channel(const diverge_model_t* model, complex128_t* buf) {
    if (model->SU2 > 0) {
        if (model->n_vert_chan[0] == 0)
            return 0;
        init_channel_SU2(model,buf,0,model->n_vert_chan[0]);
    } else {
        if (model->n_vert_chan[0] == 0 && model->n_vert_chan[1] == 0)
            return 0;
        init_C_channel_nonSU2(model,buf);
    }
    return 1;
}

int init_D_channel(const diverge_model_t* model, complex128_t* buf) {
    if (model->SU2 > 0) {
        if (model->n_vert_chan[1] == 0)
            return 0;
        init_channel_SU2(model,buf,model->n_vert_chan[0],model->n_vert_chan[1]);
    } else {
        if (model->n_vert_chan[0] == 0 && model->n_vert_chan[1] == 0)
            return 0;
        init_D_channel_nonSU2(model,buf);
    }
    return 1;
}

static inline void init_channel_SU2(const diverge_model_t* model, complex128_t* buf, index_t offset, index_t n_elem) {
    const index_t no = model->n_orb;
    const index_t* nk = model->nk;
    memset(buf,0,sizeof(complex128_t)*kdim(nk)*POW2(no));

    for (index_t idx=offset; idx<offset+n_elem; ++idx) {
        index_t o1 = model->vert[idx].o1,
                o2 = model->vert[idx].o2;
        const index_t* R = model->vert[idx].R;
        index_t z_id = R[2], y_id = R[1], x_id = R[0];
        while (z_id < 0) { z_id += nk[2]; } z_id %= nk[2];
        while (y_id < 0) { y_id += nk[1]; } y_id %= nk[1];
        while (x_id < 0) { x_id += nk[0]; } x_id %= nk[0];
        index_t Ridx = IDX3(x_id,y_id,z_id,nk[1],nk[2]);

        buf[IDX3(Ridx,o1,o2,no,no)] += model->vert[idx].V;
     }
}

static inline void init_C_channel_nonSU2(const diverge_model_t* model, complex128_t* buf) {
    const index_t no = model->n_orb,
                  ns = model->n_spin;
    const index_t* nk = model->nk;
    memset(buf,0,sizeof(complex128_t)*kdim(nk)*POW2(no*ns*ns));

    for (index_t idx=0; idx<model->n_vert_chan[0]; ++idx) {
        index_t o1 = model->vert[idx].o1,
                o2 = model->vert[idx].o2,
                s1 = model->vert[idx].s1,
                s2 = model->vert[idx].s2,
                s3 = model->vert[idx].s3,
                s4 = model->vert[idx].s4;
        const index_t* R = model->vert[idx].R;
        index_t z_id = R[2], y_id = R[1], x_id = R[0];
        while (z_id < 0) { z_id += nk[2]; } z_id %= nk[2];
        while (y_id < 0) { y_id += nk[1]; } y_id %= nk[1];
        while (x_id < 0) { x_id += nk[0]; } x_id %= nk[0];
        index_t Ridx = IDX3(x_id,y_id,z_id,nk[1],nk[2]);

        if (s1 == -1) {
            for (index_t si1=0; si1<2; si1++)
            for (index_t si2=0; si2<2; si2++) {
                s1 = si1, s2 = si2, s3 = si1, s4 = si2;
                buf[IDX7(Ridx,s1,s2,o1,s3,s4,o2,ns,ns,no,ns,ns,no)] += model->vert[idx].V;
            }
        } else {
            buf[IDX7(Ridx,s1,s2,o1,s3,s4,o2,ns,ns,no,ns,ns,no)] += model->vert[idx].V;
        }
    }
}

static inline void init_D_channel_nonSU2(const diverge_model_t* model, complex128_t* buf) {
    const index_t no = model->n_orb,
                  ns = model->n_spin;
    const index_t* nk = model->nk;
    memset(buf,0,sizeof(complex128_t)*kdim(nk)*POW2(no*ns*ns));

    for (index_t idx=model->n_vert_chan[0]; idx<model->n_vert_chan[0]+model->n_vert_chan[1]; ++idx) {
        index_t o1 = model->vert[idx].o1,
                o2 = model->vert[idx].o2,
                s1 = model->vert[idx].s1,
                s2 = model->vert[idx].s2,
                s3 = model->vert[idx].s3,
                s4 = model->vert[idx].s4;
        const index_t* R = model->vert[idx].R;
        index_t z_id = R[2], y_id = R[1], x_id = R[0];
        while (z_id < 0) { z_id += nk[2]; } z_id %= nk[2];
        while (y_id < 0) { y_id += nk[1]; } y_id %= nk[1];
        while (x_id < 0) { x_id += nk[0]; } x_id %= nk[0];
        index_t Ridx = IDX3(x_id,y_id,z_id,nk[1],nk[2]);

        if (s1 == -1) {
            for (index_t si1=0; si1<2; si1++)
            for (index_t si2=0; si2<2; si2++) {
                s1 = si1, s2 = si1, s3 = si2, s4 = si2;
                buf[IDX7(Ridx,s1,s2,o1,s3,s4,o2,ns,ns,no,ns,ns,no)] += model->vert[idx].V;
            }
        } else {
            buf[IDX7(Ridx,s1,s2,o1,s3,s4,o2,ns,ns,no,ns,ns,no)] += model->vert[idx].V;
        }
    }
}


static inline void init_P_channel_nonSU2(const diverge_model_t* model, complex128_t* buf) {
    const index_t offset = model->n_vert_chan[0]+model->n_vert_chan[1];
    const index_t no = model->n_orb,
                  ns = model->n_spin;
    const index_t* nk = model->nk;
    memset(buf,0,sizeof(complex128_t)*kdim(nk)*POW2(no*ns*ns));

    for(index_t idx = offset; idx < offset + model->n_vert_chan[2]; ++idx) {
        index_t o1 = model->vert[idx].o1,
                o2 = model->vert[idx].o2,
                s1 = model->vert[idx].s1,
                s2 = model->vert[idx].s2,
                s3 = model->vert[idx].s3,
                s4 = model->vert[idx].s4;
        const index_t* R = model->vert[idx].R;
        index_t z_id = R[2], y_id = R[1], x_id = R[0];
        while (z_id < 0) { z_id += nk[2]; } z_id %= nk[2];
        while (y_id < 0) { y_id += nk[1]; } y_id %= nk[1];
        while (x_id < 0) { x_id += nk[0]; } x_id %= nk[0];
        index_t Ridx = IDX3(x_id,y_id,z_id,nk[1],nk[2]);
        if (s1 == -1) {
            for (index_t si1=0; si1<2; si1++)
            for (index_t si2=0; si2<2; si2++) {
                s1 = si1, s2 = si2, s3 = si1, s4 = si2;
                buf[IDX7(Ridx,s1,s2,o1,s3,s4,o2,ns,ns,no,ns,ns,no)] += model->vert[idx].V;
            }
        } else {
            buf[IDX7(Ridx,s1,s2,o1,s3,s4,o2,ns,ns,no,ns,ns,no)] += model->vert[idx].V;
        }
    }
}
