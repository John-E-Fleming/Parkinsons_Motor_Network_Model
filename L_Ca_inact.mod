TITLE Motoneuron L-type Calcium channels

COMMENT
	
	Comments from Original Implementation:
	
	The parameters for this current come from V. Booth et al. J Neurophysiol 78:3371-3385, 1997
	Iterative equations
	Modified by RP to provide calcium to a separate pool (caL)and to have adjustable equilibrium
	potential vca

	Model Reference: 
  
	Powers, R.K. and Heckman, C.J., 2017. 
	"Synaptic control of the shape of the motoneuron 
	pool input-output function." 
	Journal of neurophysiology, 117(3), pp.1171-1184.
  
    Original Code Link:
      https://senselab.med.yale.edu/ModelDB/showmodel?model=239582

ENDCOMMENT


NEURON {
	SUFFIX L_Ca_inact
	USEION caL READ ecaL WRITE icaL VALENCE 2
	RANGE gcabar,icaL,m_inf,m,h
	RANGE theta_m
	RANGE tau_m
	RANGE theta_h
	RANGE tau_h
	RANGE kappa_h
	GLOBAL vca,kappa_m
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gcabar  = 0.0003  (mho/cm2)
	ecaL		(mV)	: eca can't be set here, only in hoc
:	celsius = 36	(degC)
	dt		(ms)
	tau_m	= 40	(ms)
	v		(mV)
        vca=80		(mV)
	theta_m = -30   (mV)
	kappa_m = -6	(-mV)
	tau_h	= 2500	(ms)
	theta_h = 14   (mV)
	kappa_h = 5	(-mV)
}

STATE {
	m
	h
}

ASSIGNED {
	icaL		(mA/cm2)
	m_inf
	h_inf
	tadj
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	icaL = gcabar * m *h* (v - vca)  :I have tried this as m*m also
}

DERIVATIVE states {
	evaluate_fct(v)
	m' = (m_inf - m) / tau_m
	h' = (h_inf - h) / tau_h
}

UNITSOFF
INITIAL {

:
:  Q10 was assumed to be 3 for both currents
:
:	tadj = 3.0 ^ ((celsius-36)/ 10 )

	evaluate_fct(v)
	m = m_inf
	h = h_inf
}

PROCEDURE evaluate_fct(v(mV)) {

	m_inf = 1 / (1 + (Exp((v - theta_m)/ kappa_m))): / tadj
	h_inf = 1 / (1 + (Exp((v - theta_h)/ kappa_h))): / tadj

}

FUNCTION vtrap(x,y) {
	if (fabs(x/y) < 1e-6) {
		vtrap = y*(1 - x/y/2)
	}else{
		vtrap = x/(Exp(x/y)-1)
	}
}

FUNCTION Exp(x) {
	if (x < -100) {
		Exp = 0
	}else{
		Exp = exp(x)
	}
} 

