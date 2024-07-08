TITLE Fluctuating conductances

COMMENT
-----------------------------------------------------------------------------

  Fluctuating conductance model for synaptic bombardment
  ======================================================

THEORY

  Synaptic bombardment is represented by a stochastic model containing
  two fluctuating conductances g_e(t) and g_i(t) descibed by:

     Isyn = g_e(t) * [V - E_e] + g_i(t) * [V - E_i]
     d g_e / dt = -(g_e - ge0) / tau_e + sqrt(D_e) * Ft
     d g_i / dt = -(g_i - gi0) / tau_i + sqrt(D_i) * Ft

  where E_e, E_i are the reversal potentials, ge0, gi0 are the average
  conductances, tau_e, tau_i are time constants, D_e, D_i are noise diffusion
  coefficients and Ft is a gaussian white noise of unit standard deviation.

  g_e and g_i are described by an Ornstein-Uhlenbeck (OU) stochastic process
  where tau_e and tau_i represent the "correlation" (if tau_e and tau_i are 
  zero, g_e and g_i are white noise).  The estimation of OU parameters can
  be made from the power spectrum:

     S(w) =  2 * D * tau^2 / (1 + w^2 * tau^2)

  and the diffusion coeffient D is estimated from the variance:

     D = 2 * sigma^2 / tau


NUMERICAL RESOLUTION

  The numerical scheme for integration of OU processes takes advantage 
  of the fact that these processes are gaussian, which led to an exact
  update rule independent of the time step dt (see Gillespie DT, Am J Phys 
  64: 225, 1996):

     x(t+dt) = x(t) * exp(-dt/tau) + A * N(0,1)

  where A = sqrt( D*tau/2 * (1-exp(-2*dt/tau)) ) and N(0,1) is a normal
  random number (avg=0, sigma=1)


IMPLEMENTATION

  This mechanism is implemented as a nonspecific current defined as a
  point process.


PARAMETERS

  The mechanism takes the following parameters:

     E_e = 0  (mV)    : reversal potential of excitatory conductance
     E_i = -75 (mV)    : reversal potential of inhibitory conductance

     ge0 = 0.0121 (umho)  : average excitatory conductance
     gi0 = 0.0573 (umho)  : average inhibitory conductance

     stde = 0.0030 (umho)  : standard dev of excitatory conductance
     stdi = 0.0066 (umho)  : standard dev of inhibitory conductance

     tau_e = 2.728 (ms)    : time constant of excitatory conductance
     tau_i = 10.49 (ms)    : time constant of inhibitory conductance

REFERENCE

  Destexhe, A., Rudolph, M., Fellous, J-M. and Sejnowski, T.J.  
  Fluctuating synaptic conductances recreate in-vivo--like activity in
  neocortical neurons. Neuroscience 107: 13-24 (2001).

  (electronic copy available at http://cns.iaf.cnrs-gif.fr)

  A. Destexhe, 1999

-----------------------------------------------------------------------------

20150414 -- Ted Carnevale
Fixed so that zero value for tau_e or tau_i results in 
"white noise" fluctuation of g_e or g_i.
In the previous implementation, tau_e or tau_i == 0 
made g_e or g_i equal to ge0 or gi0.

The fix involved
1.  restructuring conditional statements so that 
zero value for tau_e or tau_i had desired effect,
and
2.  moving all calls to normrand from the BREAKPOINT block
into PROCEDURE oup.

The latter was necessary because code in the BREAKPOINT block
is executed twice per time step in order to estimate di/dv
(the slope conductance of this "channel") for the Jacobian matrix.
The estimate is simply
di/dv ~ (i(v+0.001) - i(v))/0.001
where the current i is calculated by a statement of the form
i = f(v)
and f() is an algebraic expression that involves v.

Clearly if the algebraic expression involves terms that change 
from one call to the next, the estimate of di/dv will be incorrect,
and the solution will be corrupted.
The most obvious symptom of this is an occasional, abrupt, large 
jump of v, but even step-to-step small fluctuations of v will be 
incorrect.

The solution is to relegate calls to random number generators
either to a PROCEDURE that is SOLVEd, or to a BEFORE BREAKPOINT block,
because code in those blocks is executed only once per advance.
ENDCOMMENT

NEURON {
  POINT_PROCESS Gfluct
  RANGE g_e, g_i, E_e, E_i, ge0, gi0, g_e1, g_i1
  RANGE stde, stdi, tau_e, tau_i, D_e, D_i
  RANGE new_seed
  NONSPECIFIC_CURRENT i
}

UNITS {
  (nA) = (nanoamp) 
  (mV) = (millivolt)
  (umho) = (micromho)
}

PARAMETER {
  dt    (ms)

  E_e  = 0   (mV)  : reversal potential of excitatory conductance
  E_i  = -75   (mV)  : reversal potential of inhibitory conductance

  ge0  = 0.0121 (umho)  : average excitatory conductance
  gi0  = 0.0573 (umho)  : average inhibitory conductance

  stde  = 0.0030 (umho)  : standard dev of excitatory conductance
  stdi  = 0.0066 (umho)  : standard dev of inhibitory conductance

  tau_e  = 2.728  (ms)  : time constant of excitatory conductance
  tau_i  = 10.49  (ms)  : time constant of inhibitory conductance
}

ASSIGNED {
  v  (mV)    : membrane voltage
  i   (nA)    : fluctuating current
  g_e  (umho)    : total excitatory conductance
  g_i  (umho)    : total inhibitory conductance
  g_e1  (umho)    : fluctuating excitatory conductance
  g_i1  (umho)    : fluctuating inhibitory conductance
  D_e  (umho umho /ms) : excitatory diffusion coefficient
  D_i  (umho umho /ms) : inhibitory diffusion coefficient
  exp_e
  exp_i
  amp_e  (umho)
  amp_i  (umho)
}

INITIAL {
  g_e1 = 0
  g_i1 = 0
  if(tau_e != 0) {
    D_e = 2 * stde * stde / tau_e
    exp_e = exp(-dt/tau_e)
    amp_e = stde * sqrt( (1-exp(-2*dt/tau_e)) )
  }
  if(tau_i != 0) {
    D_i = 2 * stdi * stdi / tau_i
    exp_i = exp(-dt/tau_i)
    amp_i = stdi * sqrt( (1-exp(-2*dt/tau_i)) )
  }
}

BREAKPOINT {
  SOLVE oup
  i = g_e * (v - E_e) + g_i * (v - E_i)
}

PROCEDURE oup() {    : use Scop function normrand(mean, std_dev)
  if(tau_e==0) {
    g_e = ge0 + stde * normrand(0,1)
  } else {
    g_e1 = exp_e * g_e1 + amp_e * normrand(0,1)
    g_e = ge0 + g_e1
  }
  if (g_e < 0) { g_e = 0 }
  if(tau_i==0) {
    g_i = gi0 + stdi * normrand(0,1)
  } else {
    g_i1 = exp_i * g_i1 + amp_i * normrand(0,1)
    g_i = gi0 + g_i1
  }
  if (g_i < 0) { g_i = 0 }
}

PROCEDURE new_seed(seed) {    : procedure to set the seed
  set_seed(seed)
  VERBATIM
    printf("Setting random generator with seed = %g\n", _lseed);
  ENDVERBATIM
}