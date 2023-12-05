#include "bs23.h"
#include <math.h>

#define BS23_SAFETY_FACTOR 1e-19

#ifndef MAX
#define MAX(A,B) ((A)>(B)?(A):(B))
#endif

double bs23_step( bs23_t* step, bs23_fun_t* fun, double t, double dt, void* userdata ) {
    const index_t size = step->ydim;
    double *k1 = step->k1, *k2 = step->k2, *k3 = step->k3, *yn = step->yn;

    if (!step->has_k1) (*fun)( t, yn, k1, size, userdata );
    // k1 filled

    #pragma omp parallel for
    for (index_t i=0; i<size; ++i) yn[i] = yn[i] + 0.5*dt*k1[i];
    // yn now contains yn + 0.5*dt*k1
    (*fun)( t+0.5*dt, yn, k2, size, userdata );
    // k2 filled

    #pragma omp parallel for
    for (index_t i=0; i<size; ++i) yn[i] = yn[i] - 0.5*dt*k1[i] + 0.75*dt*k2[i];
    // yn now contains yn+0.75*dt*k2
    (*fun)( t+0.75*dt, yn, k3, size, userdata );
    // k3 filled

    double *yn1 = yn, *k4 = k1;
    #pragma omp parallel for
    for (index_t i=0; i<size; ++i) yn1[i] = yn[i] + 2./9.*dt*k1[i] + (1./3.-0.75)*dt*k2[i] + 4./9.*dt*k3[i];
    // use (1./3.-0.75)*dt*k2 because yn already includes +0.75*dt*k2

    double *zn1 = k2;
    #pragma omp parallel for
    for (index_t i=0; i<size; ++i) zn1[i] = yn[i] + (7./24.-2./9.)*dt*k1[i] + (1.-1./3.)*dt*k2[i] + (1./3.-4./9.)*dt*k3[i];
    // zn1 contains all zn1 contributions except for the one from k4

    step->has_k1 = 1;
    (*fun)( t+dt, yn1, k4, size, userdata );
    // k4 is filled, i.e., k1!

    #pragma omp parallel for
    for (index_t i=0; i<size; ++i) zn1[i] = zn1[i] + 1./8.*dt*k4[i];
    // zn1 is done

    double squaredNormDiff = 0.0,
           yzmax = 0.0;
    #pragma omp parallel for reduction(+:squaredNormDiff) reduction(max:yzmax)
    for (index_t i=0; i<size; ++i) {
        squaredNormDiff += (zn1[i] - yn1[i])*(zn1[i] - yn1[i]);
        double yzmax_local = MAX(fabs(zn1[i]), fabs(yn1[i]));
        yzmax = MAX(yzmax, yzmax_local);
    }
    yzmax = MAX(yzmax, BS23_SAFETY_FACTOR);
    return sqrt(squaredNormDiff)/yzmax;
}
