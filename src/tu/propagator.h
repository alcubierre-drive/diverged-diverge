#pragma once

#include "../diverge_common.h"
#include "../diverge_model.h"
#include "../diverge_flow_step.h" // timing
#include "diverge_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tu_prop_t tu_prop_t;

tu_prop_t* tu_prop_init( diverge_model_t* model );
void tu_prop_free( tu_prop_t* t );

void tu_prop_ph_loop( tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu/*=NULL*/ );
void tu_prop_pp_loop( tu_prop_t* t, complex128_t* L, complex128_t* tmpgpu/*=NULL*/ );
void tu_prop_GF_zero_T( tu_prop_t* t, complex128_t lambda, complex128_t* self_energy/*=NULL*/ );
void tu_prop_GF_for_self( tu_prop_t* t, complex128_t* coarseGF );

/*
With a sharp frequency cutoff, it is possible to use FFTs in the loop integrals
of TUFRG. Therefore, we cannot include the default Katanin substitution: It
would invalidate the convolution property of the loop integrals that is present
due to the sharp frequency cutoff. It is therefore necessary to include it in a
different (approximate) manner. Recall that Katanin's substitution reads
\begin{equation}
  S^\Lambda \to \dot{G^\Lambda} = S^\Lambda + G^\Lambda \dot\Sigma^\Lambda
  G^\Lambda \equiv S^\Lambda + \mathcal{H}^\Lambda
\end{equation}
in the one-loop vertex diagrams. When plugged into the (integrated) loops, we
obtain
\begin{equation}
\begin{aligned}
  \dot K_{1234}^{\pm;\Lambda}(\bvec q,\bvec k) &{}= \int\frac{\dd\omega}{2\pi}
  \big[ \dot G^\Lambda_{13}(i\omega,\bvec k)
             G^\Lambda_{24}(\pm i\omega,\pm\bvec k\mp\bvec q) +
             G^\Lambda_{13}(i\omega,\bvec k)
        \dot G^\Lambda_{24}(\pm i\omega,\pm\bvec k\mp\bvec q) \big] \\
        &{}= \dot L_{1234}^{\pm;\Lambda}(\bvec q,\bvec k) +
        \underbrace{\int\frac{\dd\omega}{2\pi}
  \big[ \mathcal H^\Lambda_{13}(i\omega,\bvec k)
        G^\Lambda_{24}(\pm i\omega,\pm\bvec k\mp\bvec q) +
        G^\Lambda_{13}(i\omega,\bvec k)
        \mathcal H^\Lambda_{24}(\pm i\omega,\pm\bvec k\mp\bvec q)
        \big]}_{\mathcal{K}^{\pm;\Lambda}_{1234}} \,.
\end{aligned}
\end{equation}
We observe that the second contribution, $\mathcal K^{\pm;\Lambda}$ requires an
actual integration over Matsubara frequencies, which is highly problematic
considering runtime. The way out is to approximate the integral through a
summation over all previous scales, as each object in the integral comes with a
cutoff $\Theta(|i\omega|-\Lambda)$:
\begin{equation}
\begin{aligned}
  \mathcal K^{\pm;\Lambda}_{1234} &{}=
  \int_{|\omega|>\Lambda}\frac{\dd\omega}{2\pi} \big[ \mathcal
  H^\Lambda_{13}(i\omega,\bvec k) G^\Lambda_{24}(\pm i\omega,\pm \bvec k\mp\bvec
  q) + \dots \big] \approx \sum_{|\omega| \geq \Lambda}
  \frac{\Delta\omega}{2\pi} \big[ \mathcal H^\Lambda_{13}(i\omega,\bvec k)
  G^\Lambda_{24}(\pm i\omega,\pm \bvec k\mp\bvec q) + \dots \big] \\
  &{} \approx \sum_{i=1}^{n} \frac{\Delta\Lambda_i}{2\pi} \big[ \mathcal
  H^\Lambda_{13}(i\Lambda_i,\bvec k)G^\Lambda_{24}(\pm i\Lambda_i,\pm\bvec
  k\mp\bvec q) + \dots \big] \approx \sum_{i=1}^{n} \frac{\Delta\Lambda_i}{2\pi}
  \big[ \mathcal H^{\Lambda_i}_{13}(i\Lambda_i,\bvec k)G^{\Lambda_i}_{24}(\pm
  i\Lambda_i,\pm\bvec k\mp\bvec q) + \dots \big] \,.
\end{aligned}
\end{equation}
In the last expression, we approximate the kernels $\mathcal
H^\Lambda(i\Lambda_i) \approx \mathcal H^{\Lambda_i}(i\Lambda_i)$ with their
scale-local counterparts, effectively giving the self-energy a frequency
dependence as $\Sigma(i\omega) \approx \Sigma(i\Lambda_i)$. By this replacement,
all quantities on the right-hand side are scale-local, such that they can be
computed with the same complexity as a regular TUFRG + sharp cutoff loop.
Notably, the approximation of the self-energy's frequency dependence is never
worse than the static approximation (where we essentially apply
$\Sigma(i\omega)\equiv\Sigma(0)$), as the contributions to $\Sigma$ computed in
the present scheme are always of incremental nature.
*/
bool tu_prop_katanin( const tu_prop_t* t ); // enabled?
// fill second GF set with self_energy = \Sigma + d\Lambda \dot\Sigma
void tu_prop_GF_zero_T_kat( tu_prop_t* t, complex128_t lambda, complex128_t* self_energy );
void tu_prop_katanin_swap( tu_prop_t* t ); // exchange plain with Katanin GF

diverge_timing_t* tu_prop_get_extra_timings( tu_prop_t* t, int* ntimings ); // dont free
void tu_prop_set_extra_timings( tu_prop_t* t, bool enabled );

#ifdef __cplusplus
}
#endif
