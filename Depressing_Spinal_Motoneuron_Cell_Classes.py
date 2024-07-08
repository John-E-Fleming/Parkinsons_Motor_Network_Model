# -*- coding: utf-8 -*-
"""
Created on April 03 14:27:26 2020

Depressing_Spinal_Motoneuron_Cell_Classes.py

This file contains the Spinal_Motoneuron class describing a five-compartment
motoneuron structure (single soma and four dendritic compartments) each
containing hodgkin-huxley type ionic mechanisms. Each dendritic compartment
receives a synaptic input due to correlated and independent synaptic currents.

Implemented by: John Fleming, john.fleming@ucdconnect.ie,
				Sageanne Senneff, sageanne.senneff@ucdconnect.ie
		 
Edits:	06-11-19 : John Fleming - Converted Sageanne's implementation of the motoneuron NEURON model into correct format for PyNN
		04-06-20 : John Fleming - Updated the motoneuron class to have synapse mechanisms for correlated and independent synaptic inputs on the dendrites

"""
from neuron import h
from nrnutils import Mechanism, Section
from pyNN.neuron import NativeCellType
from pyNN.parameters import Sequence
import numpy as np

# Import global variables for generating the seeds for the dendritic noise 
import Global_Variables as GV

# Load ranstream hoc template for generating random streams
h.load_file("ranstream.hoc")

try:
	reduce
except NameError:
	from functools import reduce

def _new_property(obj_hierarchy, attr_name):
	"""
	Returns a new property, mapping attr_name to obj_hierarchy.attr_name.

	For example, suppose that an object of class A has an attribute b which
	itself has an attribute c which itself has an attribute d. Then placing
		e = _new_property('b.c', 'd')
	in the class definition of A makes A.e an alias for A.b.c.d
	"""

	def set(self, value):
		obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
		setattr(obj, attr_name, value)

	def get(self):
		obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
		return getattr(obj, attr_name)
	return property(fset=set, fget=get)

class Spinal_Motoneuron_Synaptic_Bombardment_Point_Process(object):

	def __init__(self, **parameters):
		
		# Create Soma Compartments
		self.soma = Section(diam = parameters['soma_diam'], L = parameters['soma_L'], Ra = parameters['soma_Ra'], cm = parameters['soma_cm'], 
		mechanisms = (Mechanism('na3rp', gbar = parameters['soma_na3rp_gbar'], sh = parameters['soma_na3rp_sh'], ar = parameters['soma_na3rp_ar'], qinf = parameters['soma_na3rp_qinf'], thinf = parameters['soma_na3rp_thinf']), 
					  Mechanism('naps', gbar = parameters['soma_naps_gbar'], sh = parameters['soma_naps_sh'], ar = parameters['soma_naps_ar'], vslope = parameters['soma_naps_vslope'], asvh = parameters['soma_naps_asvh'], bsvh = parameters['soma_naps_bsvh']), 
					  Mechanism('kdrRL', gMax = parameters['soma_kdrRL_gMax'], tmin = parameters['soma_kdrRL_tmin'], taumax = parameters['soma_kdrRL_taumax'], mVh = parameters['soma_kdrRL_mVh']), 
					  Mechanism('mAHP', gcamax = parameters['soma_mAHP_gcamax'], gkcamax = parameters['soma_mAHP_gkcamax'], taur = parameters['soma_mAHP_taur'], mtauca = parameters['soma_mAHP_mtauca'], mvhalfca = parameters['soma_mAHP_mvhalfca'])))
		
		# Insert passive mechanism in soma
		self.soma.insert('pas')
		for seg in self.soma:
			seg.pas.g = parameters['soma_pas_g']
			seg.pas.e = parameters['soma_pas_e']
		
		# Insert h mechanism in soma
		self.soma.insert('gh')
		for seg in self.soma:
			seg.gh.ghbar = parameters['soma_gh_ghbar']
			seg.gh.half = parameters['soma_gh_half']
			seg.gh.htau = parameters['soma_gh_htau']
		
		# Set threshold netcons on the neuron
		self.soma.v_thresh = parameters['v_thres']
		
		# Add bias current to MU soma - current amplitude is in terms of original model paper, nA 
		self.soma_bias = h.IClamp(0.5, sec=self.soma)
		self.soma_bias.delay = 0
		self.soma_bias.dur = 1e12
		self.soma_bias.amp = parameters['soma_bias_current_amp']
		
		# Create Dendritic Compartments
		self.d1 = Section(diam = parameters['dendrite_diam'], L = parameters['dendrite_L'], Ra = parameters['dendrite_Ra'], cm = parameters['dendrite_cm'], 
							mechanisms = (Mechanism('pas', g = parameters['dendrite_pas_g'], e = parameters['dendrite_pas_e']), 
											Mechanism('gh', ghbar = parameters['dendrite_gh_ghbar'], half = parameters['dendrite_gh_half'], htau = parameters['dendrite_gh_htau'])))
		
		self.d2 = Section(diam = parameters['dendrite_diam'], L = parameters['dendrite_L'], Ra = parameters['dendrite_Ra'], cm = parameters['dendrite_cm'], 
							mechanisms = (Mechanism('pas', g = parameters['dendrite_pas_g'], e = parameters['dendrite_pas_e']), 
											Mechanism('gh', ghbar = parameters['dendrite_gh_ghbar'], half = parameters['dendrite_gh_half'], htau = parameters['dendrite_gh_htau'])))
				
		self.d3 = Section(diam = parameters['dendrite_diam'], L = parameters['dendrite_L'], Ra = parameters['dendrite_Ra'], cm = parameters['dendrite_cm'], 
							mechanisms = (Mechanism('pas', g = parameters['dendrite_pas_g'], e = parameters['dendrite_pas_e']), 
											Mechanism('gh', ghbar = parameters['dendrite_gh_ghbar'], half = parameters['dendrite_gh_half'], htau = parameters['dendrite_gh_htau'])))
						
		self.d4 = Section(diam = parameters['dendrite_diam'], L = parameters['dendrite_L'], Ra = parameters['dendrite_Ra'], cm = parameters['dendrite_cm'], 
							mechanisms = (Mechanism('pas', g = parameters['dendrite_pas_g'], e = parameters['dendrite_pas_e']), 
											Mechanism('gh', ghbar = parameters['dendrite_gh_ghbar'], half = parameters['dendrite_gh_half'], htau = parameters['dendrite_gh_htau'])))

		# Connect the dendritic compartments to their respective parts of the motoneuron soma
		# Note connect only takes one parameter
		self.d1.connect(self.soma(1))
		self.d2.connect(self.soma(1))
		self.d3.connect(self.soma(0))
		self.d4.connect(self.soma(0))

		# Insert the persistent inward currents on the dendrites & set their values
		# Dendrite 1:
		self.d1.insert('L_Ca_inact')
		for seg in self.d1:
			seg.L_Ca_inact.gcabar = parameters['dendrite_1_L_Ca_inact_gcabar']
			seg.L_Ca_inact.theta_m = parameters['dendrite_L_Ca_inact_theta_m']
			seg.L_Ca_inact.tau_m = parameters['dendrite_L_Ca_inact_tau_m']
			seg.L_Ca_inact.theta_h = parameters['dendrite_L_Ca_inact_theta_h']
			seg.L_Ca_inact.tau_h = parameters['dendrite_L_Ca_inact_tau_h']
			seg.L_Ca_inact.kappa_h = parameters['dendrite_L_Ca_inact_kappa_h']
		
		# Dendrite 2:
		self.d2.insert('L_Ca_inact')
		for seg in self.d2:
			seg.L_Ca_inact.gcabar = parameters['dendrite_2_L_Ca_inact_gcabar']
			seg.L_Ca_inact.theta_m = parameters['dendrite_L_Ca_inact_theta_m']
			seg.L_Ca_inact.tau_m = parameters['dendrite_L_Ca_inact_tau_m']
			seg.L_Ca_inact.theta_h = parameters['dendrite_L_Ca_inact_theta_h']
			seg.L_Ca_inact.tau_h = parameters['dendrite_L_Ca_inact_tau_h']
			seg.L_Ca_inact.kappa_h = parameters['dendrite_L_Ca_inact_kappa_h']
			
		# Dendrite 3:
		self.d3.insert('L_Ca_inact')
		for seg in self.d3:
			seg.L_Ca_inact.gcabar = parameters['dendrite_3_L_Ca_inact_gcabar']
			seg.L_Ca_inact.theta_m = parameters['dendrite_L_Ca_inact_theta_m']
			seg.L_Ca_inact.tau_m = parameters['dendrite_L_Ca_inact_tau_m']
			seg.L_Ca_inact.theta_h = parameters['dendrite_L_Ca_inact_theta_h']
			seg.L_Ca_inact.tau_h = parameters['dendrite_L_Ca_inact_tau_h']
			seg.L_Ca_inact.kappa_h = parameters['dendrite_L_Ca_inact_kappa_h']
			
		# Dendrite 4:
		self.d4.insert('L_Ca_inact')
		for seg in self.d4:
			seg.L_Ca_inact.gcabar = parameters['dendrite_4_L_Ca_inact_gcabar']
			seg.L_Ca_inact.theta_m = parameters['dendrite_L_Ca_inact_theta_m']
			seg.L_Ca_inact.tau_m = parameters['dendrite_L_Ca_inact_tau_m']
			seg.L_Ca_inact.theta_h = parameters['dendrite_L_Ca_inact_theta_h']
			seg.L_Ca_inact.tau_h = parameters['dendrite_L_Ca_inact_tau_h']
			seg.L_Ca_inact.kappa_h = parameters['dendrite_L_Ca_inact_kappa_h']
		
		# Add iclamps for playing in dendritic bias currents
		# Dendrite 1:															
		self.d1_bias_current = h.IClamp(0.5, sec=self.d1)							# IClamp used to play bias current into the dendrite
		self.d1_bias_current.delay = 0
		self.d1_bias_current.dur = 1e12
		self.d1_bias_current.amp = parameters['dendrite_bias_current_amp']	# Updated in main script to play a ramped bias current through this IClamp
		
		# Dendrite 2:															
		self.d2_bias_current = h.IClamp(0.5, sec=self.d2)							# IClamp used to play bias current into the dendrite
		self.d2_bias_current.delay = 0
		self.d2_bias_current.dur = 1e12
		self.d2_bias_current.amp = parameters['dendrite_bias_current_amp']	# Updated in main script to play a ramped bias current through this IClamp
		
		# Dendrite 3:															
		self.d3_bias_current = h.IClamp(0.5, sec=self.d3)							# IClamp used to play bias current into the dendrite
		self.d3_bias_current.delay = 0
		self.d3_bias_current.dur = 1e12
		self.d3_bias_current.amp = parameters['dendrite_bias_current_amp']	# Updated in main script to play a ramped bias current through this IClamp
		
		# Dendrite 4:															
		self.d4_bias_current = h.IClamp(0.5, sec=self.d4)							# IClamp used to play bias current into the dendrite
		self.d4_bias_current.delay = 0
		self.d4_bias_current.dur = 1e12
		self.d4_bias_current.amp = parameters['dendrite_bias_current_amp']	# Updated in main script to play a ramped bias current through this IClamp
		
		# Append the gaussian IClamps to global list so they can be updated in the main script
		GV.MU_dendritic_bias_current_iclamps.append([self.d1_bias_current, self.d2_bias_current, self.d3_bias_current, self.d4_bias_current])
		
		# Add Double Exponential Synapses to soma and each dendritic compartment for correlated inputs from cortex
		# Correlated Inputs - Excitatory Synapse:
		# Soma:
		self.soma_correlated_exc = h.ExpSyn(0.5, sec=self.soma)
		self.soma_correlated_exc.e = parameters['correlated_e_rev_exc']			# Excitatory reversal potential
		self.soma_correlated_exc.tau = parameters['correlated_tau']			# Excitatory decay time
		
		# Dendrite 1:
		self.d1_correlated_exc = h.ExpSyn(0.5, sec=self.d1)
		self.d1_correlated_exc.e = parameters['correlated_e_rev_exc']			# Excitatory reversal potential
		self.d1_correlated_exc.tau = parameters['correlated_tau']				# Excitatory decay time

		# Dendrite 2:
		self.d2_correlated_exc = h.ExpSyn(0.5, sec=self.d2)
		self.d2_correlated_exc.e = parameters['correlated_e_rev_exc']			# Excitatory reversal potential
		self.d2_correlated_exc.tau = parameters['correlated_tau']				# Excitatory decay time

		# Dendrite 3:
		self.d3_correlated_exc = h.ExpSyn(0.5, sec=self.d3)
		self.d3_correlated_exc.e = parameters['correlated_e_rev_exc']			# Excitatory reversal potential
		self.d3_correlated_exc.tau = parameters['correlated_tau']				# Excitatory decay time
		
		# Dendrite 4:
		self.d4_correlated_exc = h.ExpSyn(0.5, sec=self.d4)
		self.d4_correlated_exc.e = parameters['correlated_e_rev_exc']			# Excitatory reversal potential
		self.d4_correlated_exc.tau = parameters['correlated_tau']				# Excitatory decay time
		
		# Independent Inputs - Excitatory Synapse:
		# Dendrite 1:
		self.d1_independent_exc = h.Exp2Syn(0.5, sec=self.d1)
		self.d1_independent_exc.e = parameters['independent_e_rev_exc']			# Excitatory reversal potential
		self.d1_independent_exc.tau1 = parameters['independent_tau1']			# Excitatory rise time
		self.d1_independent_exc.tau2 = parameters['independent_tau2']			# Excitatory decay time

		# Dendrite 2:
		self.d2_independent_exc = h.Exp2Syn(0.5, sec=self.d2)
		self.d2_independent_exc.e = parameters['independent_e_rev_exc']			# Excitatory reversal potential
		self.d2_independent_exc.tau1 = parameters['independent_tau1']			# Excitatory rise time
		self.d2_independent_exc.tau2 = parameters['independent_tau2']			# Excitatory decay time

		# Dendrite 3:
		self.d3_independent_exc = h.Exp2Syn(0.5, sec=self.d3)
		self.d3_independent_exc.e = parameters['independent_e_rev_exc']			# Excitatory reversal potential
		self.d3_independent_exc.tau1 = parameters['independent_tau1']			# Excitatory rise time
		self.d3_independent_exc.tau2 = parameters['independent_tau2']			# Excitatory decay time
		
		# Dendrite 4:
		self.d4_independent_exc = h.Exp2Syn(0.5, sec=self.d4)
		self.d4_independent_exc.e = parameters['independent_e_rev_exc']			# Excitatory reversal potential
		self.d4_independent_exc.tau1 = parameters['independent_tau1']			# Excitatory rise time
		self.d4_independent_exc.tau2 = parameters['independent_tau2']			# Excitatory decay time
		
		# Synaptic Bombardment Noise Inputs to Dendrites:
		# Dendrite 1:
		self.d1_Synaptic_Bombardment = h.Gfluct(0.5, sec=self.d1)
		self.d1_Synaptic_Bombardment.ge0 = parameters['dendrite_Gfluct_ge0']
		self.d1_Synaptic_Bombardment.gi0 = parameters['dendrite_Gfluct_gi0']
		self.d1_Synaptic_Bombardment.stde = parameters['dendrite_Gfluct_stde']
		self.d1_Synaptic_Bombardment.stdi = parameters['dendrite_Gfluct_stdi']
		self.d1_Synaptic_Bombardment.tau_e = parameters['dendrite_Gfluct_tau_e']
		self.d1_Synaptic_Bombardment.tau_i = parameters['dendrite_Gfluct_tau_i']
		self.d1_Synaptic_Bombardment.E_e = parameters['dendrite_Gfluct_E_e']
		self.d1_Synaptic_Bombardment.E_i = parameters['dendrite_Gfluct_E_i']
		
		# Set new seed for the process
		d1_r = h.Random()						# Make random number for seed
		d1_r.Random123(GV.stream_index, 0, 0)
		#self.d1_Synaptic_Bombardment.new_seed = d1_r
		
		# Dendrite 2:
		self.d2_Synaptic_Bombardment = h.Gfluct(0.5, sec=self.d2)
		self.d2_Synaptic_Bombardment.ge0 = parameters['dendrite_Gfluct_ge0']
		self.d2_Synaptic_Bombardment.gi0 = parameters['dendrite_Gfluct_gi0']
		self.d2_Synaptic_Bombardment.stde = parameters['dendrite_Gfluct_stde']
		self.d2_Synaptic_Bombardment.stdi = parameters['dendrite_Gfluct_stdi']
		self.d2_Synaptic_Bombardment.tau_e = parameters['dendrite_Gfluct_tau_e']
		self.d2_Synaptic_Bombardment.tau_i = parameters['dendrite_Gfluct_tau_i']
		self.d2_Synaptic_Bombardment.E_e = parameters['dendrite_Gfluct_E_e']
		self.d2_Synaptic_Bombardment.E_i = parameters['dendrite_Gfluct_E_i']
		
		# Set new seed for the process
		d2_r = h.Random()						# Make random number for seed
		d2_r.Random123(GV.stream_index, 1, 0)
		#self.d2_Synaptic_Bombardment.new_seed = d2_r
			
		# Dendrite 3:
		self.d3_Synaptic_Bombardment = h.Gfluct(0.5, sec=self.d3)
		self.d3_Synaptic_Bombardment.ge0 = parameters['dendrite_Gfluct_ge0']
		self.d3_Synaptic_Bombardment.gi0 = parameters['dendrite_Gfluct_gi0']
		self.d3_Synaptic_Bombardment.stde = parameters['dendrite_Gfluct_stde']
		self.d3_Synaptic_Bombardment.stdi = parameters['dendrite_Gfluct_stdi']
		self.d3_Synaptic_Bombardment.tau_e = parameters['dendrite_Gfluct_tau_e']
		self.d3_Synaptic_Bombardment.tau_i = parameters['dendrite_Gfluct_tau_i']
		self.d3_Synaptic_Bombardment.E_e = parameters['dendrite_Gfluct_E_e']
		self.d3_Synaptic_Bombardment.E_i = parameters['dendrite_Gfluct_E_i']
		
		# Set new seed for the process
		d3_r = h.Random()						# Make random number for seed
		d3_r.Random123(GV.stream_index, 2, 0)
		#self.d3_Synaptic_Bombardment.new_seed = d3_r
		
		# Dendrite 4:
		self.d4_Synaptic_Bombardment = h.Gfluct(0.5, sec=self.d4)
		self.d4_Synaptic_Bombardment.ge0 = parameters['dendrite_Gfluct_ge0']
		self.d4_Synaptic_Bombardment.gi0 = parameters['dendrite_Gfluct_gi0']
		self.d4_Synaptic_Bombardment.stde = parameters['dendrite_Gfluct_stde']
		self.d4_Synaptic_Bombardment.stdi = parameters['dendrite_Gfluct_stdi']
		self.d4_Synaptic_Bombardment.tau_e = parameters['dendrite_Gfluct_tau_e']
		self.d4_Synaptic_Bombardment.tau_i = parameters['dendrite_Gfluct_tau_i']
		self.d4_Synaptic_Bombardment.E_e = parameters['dendrite_Gfluct_E_e']
		self.d4_Synaptic_Bombardment.E_i = parameters['dendrite_Gfluct_E_i']
		
		# Append the noisy conductances to global list so they can be updated in the main script
		GV.MU_dendritic_noisy_conductances.append([self.d1_Synaptic_Bombardment, self.d2_Synaptic_Bombardment, self.d3_Synaptic_Bombardment, self.d4_Synaptic_Bombardment])
		
		# Set new seed for the process
		d4_r = h.Random()						# Make random number for seed
		d4_r.Random123(GV.stream_index, 3, 0)
		#self.d4_Synaptic_Bombardment.new_seed = d4_r
		
		# Update the stream index for generating other random number seeds
		GV.stream_index = GV.stream_index + 1
		
		# needed for PyNN
		self.source_section = self.soma
		self.source = self.soma(0.5)._ref_v
		self.rec = h.NetCon(self.source, None, sec=self.source_section)		# Needed to clear the simulator
		self.spike_times = h.Vector(0)
		self.traces = {}
		self.recording_time = False
		self.parameter_names = ('soma_diam', 'soma_L', 'soma_Ra', 'soma_cm', 'soma_ek', 'v_thres',
								'dendrite_diam', 'dendrite_L', 'dendrite_Ra', 'dendrite_cm', 
								'soma_na3rp_gbar', 'soma_na3rp_sh', 'soma_na3rp_ar', 'soma_na3rp_qinf', 'soma_na3rp_thinf',
								'soma_naps_gbar', 'soma_naps_sh', 'soma_naps_ar', 'soma_naps_vslope', 'soma_naps_asvh', 'soma_naps_bsvh',
								'soma_kdrRL_gMax', 'soma_kdrRL_tmin', 'soma_kdrRL_taumax', 'soma_kdrRL_mVh', 
								'soma_mAHP_gcamax', 'soma_mAHP_gkcamax', 'soma_mAHP_taur', 'soma_mAHP_mtauca', 'soma_mAHP_mvhalfca',
								'soma_gh_ghbar', 'soma_gh_half', 'soma_gh_htau',
								'soma_pas_g', 'soma_pas_e', 
								'dendrite_pas_g', 'dendrite_pas_e',
								'dendrite_1_L_Ca_inact_gcabar', 'dendrite_2_L_Ca_inact_gcabar', 'dendrite_3_L_Ca_inact_gcabar', 'dendrite_4_L_Ca_inact_gcabar', 'dendrite_L_Ca_inact_theta_m', 'dendrite_L_Ca_inact_tau_m', 'dendrite_L_Ca_inact_theta_h', 'dendrite_L_Ca_inact_tau_h', 'dendrite_L_Ca_inact_kappa_h',
								'dendrite_gh_ghbar', 'dendrite_gh_half', 'dendrite_gh_htau', 
								'correlated_e_rev_exc', 'correlated_tau', 
								'independent_e_rev_exc', 'independent_tau1', 'independent_tau2',
								'soma_bias_current_amp', 'dendrite_bias_current_amp',
								'dendrite_Gfluct_ge0', 'dendrite_Gfluct_gi0', 'dendrite_Gfluct_stde', 'dendrite_Gfluct_stdi', 'dendrite_Gfluct_tau_e', 'dendrite_Gfluct_tau_i', 'dendrite_Gfluct_E_e', 'dendrite_Gfluct_E_i'								
								)
		self.traces = {}
		self.recording_time = False
		
	def memb_init(self):
		for seg in self.soma:				# Initialize all compartment membrane voltages
			seg.v = self.v_init
		for seg in self.d1:
			seg.v = self.v_init
		for seg in self.d2:
			seg.v = self.v_init
		for seg in self.d3:
			seg.v = self.v_init
		for seg in self.d4:
			seg.v = self.v_init
	
	def get_threshold(self):				# Initialize the soma membrane threshold for netcons
		return self.soma.v_thresh
	
	# Need to make getters and setters for changing motoneuron parameters
	# Soma -
	# diam Setters & Getters:
	def _set_soma_diam(self, value):
		self.soma.diam = value
	def _get_soma_diam(self):
		return self.soma.diam
	soma_diam = property(fget=_get_soma_diam, fset=_set_soma_diam)
	
	# L Setters & Getters:
	def _set_soma_L(self, value):
		self.soma.L = value
	def _get_soma_L(self):
		return self.soma.L
	soma_L = property(fget=_get_soma_L, fset=_set_soma_L)
	
	# cm Setters & Getters:
	def _set_soma_cm(self, value):
		self.soma.cm = value
	def _get_soma_cm(self):
		return self.soma.cm
	soma_cm = property(fget=_get_soma_cm, fset=_set_soma_cm)
	
	# Ra Setters & Getters:
	def _set_soma_Ra(self, value):
		self.soma.Ra = value
	def _get_soma_Ra(self):
		return self.soma.Ra
	soma_Ra = property(fget=_get_soma_Ra, fset=_set_soma_Ra)
	
	# na3rp Setters & Getters:
	def _set_soma_na3rp_gbar(self, value):
		for seg in self.soma:
			seg.na3rp.gbar = value
	def _get_soma_na3rp_gbar(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.soma(0.5).na3rp.gbar
	soma_na3rp_gbar = property(fget=_get_soma_na3rp_gbar, fset=_set_soma_na3rp_gbar)
	
	def _set_soma_na3rp_sh(self, value):
		for seg in self.soma:
			seg.na3rp.sh = value
	def _get_soma_na3rp_sh(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.soma(0.5).na3rp.sh
	soma_na3rp_sh = property(fget=_get_soma_na3rp_sh, fset=_set_soma_na3rp_sh)
	
	def _set_soma_na3rp_ar(self, value):
		for seg in self.soma:
			seg.na3rp.ar = value
	def _get_soma_na3rp_ar(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.soma(0.5).na3rp.ar
	soma_na3rp_ar = property(fget=_get_soma_na3rp_ar, fset=_set_soma_na3rp_ar)
	
	def _set_soma_na3rp_qinf(self, value):
		for seg in self.soma:
			seg.na3rp.qinf = value
	def _get_soma_na3rp_qinf(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.soma(0.5).na3rp.qinf
	soma_na3rp_qinf = property(fget=_get_soma_na3rp_qinf, fset=_set_soma_na3rp_qinf)
	
	def _set_soma_na3rp_thinf(self, value):
		for seg in self.soma:
			seg.na3rp.thinf = value
	def _get_soma_na3rp_thinf(self):
		# Return only the middle soma parameter
		return self.soma(0.5).na3rp.thinf
	soma_na3rp_thinf = property(fget=_get_soma_na3rp_thinf, fset=_set_soma_na3rp_thinf)

	# naps Setters & Getters:
	def _set_soma_naps_gbar(self, value):
		for seg in self.soma:
			seg.naps.gbar = value
	def _get_soma_naps_gbar(self):
		# Return only the middle soma parameter
		return self.soma(0.5).naps.gbar
	soma_naps_gbar = property(fget=_get_soma_naps_gbar, fset=_set_soma_naps_gbar)

	def _set_soma_naps_sh(self, value):
		for seg in self.soma:
			seg.naps.sh = value
	def _get_soma_naps_sh(self):
		# Return only the middle soma parameter
		return self.soma(0.5).naps.sh
	soma_naps_sh = property(fget=_get_soma_naps_sh, fset=_set_soma_naps_sh)

	def _set_soma_naps_ar(self, value):
		for seg in self.soma:
			seg.naps.ar = value
	def _get_soma_naps_ar(self):
		# Return only the middle soma parameter
		return self.soma(0.5).naps.ar
	soma_naps_ar = property(fget=_get_soma_naps_ar, fset=_set_soma_naps_ar)

	def _set_soma_naps_vslope(self, value):
		for seg in self.soma:
			seg.naps.vslope = value
	def _get_soma_naps_vslope(self):
		# Return only the middle soma parameter
		return self.soma(0.5).naps.vslope
	soma_naps_vslope = property(fget=_get_soma_naps_vslope, fset=_set_soma_naps_vslope)

	def _set_soma_naps_asvh(self, value):
		for seg in self.soma:
			seg.naps.asvh = value
	def _get_soma_naps_asvh(self):
		# Return only the middle soma parameter
		return self.soma(0.5).naps.asvh
	soma_naps_asvh = property(fget=_get_soma_naps_asvh, fset=_set_soma_naps_asvh)

	def _set_soma_naps_bsvh(self, value):
		for seg in self.soma:
			seg.naps.bsvh = value
	def _get_soma_naps_bsvh(self):
		# Return only the middle soma parameter
		return self.soma(0.5).naps.bsvh
	soma_naps_bsvh = property(fget=_get_soma_naps_bsvh, fset=_set_soma_naps_bsvh)

	# kdrRL Setters & Getters:
	def _set_soma_kdrRL_gMax(self, value):
		for seg in self.soma:
			seg.kdrRL.gMax = value
	def _get_soma_kdrRL_gMax(self):
		# Return only the middle soma parameter
		return self.soma(0.5).kdrRL.gMax
	soma_kdrRL_gMax = property(fget=_get_soma_kdrRL_gMax, fset=_set_soma_kdrRL_gMax)

	def _set_soma_kdrRL_tmin(self, value):
		for seg in self.soma:
			seg.kdrRL.tmin = value
	def _get_soma_kdrRL_tmin(self):
		# Return only the middle soma parameter
		return self.soma(0.5).kdrRL.tmin
	soma_kdrRL_tmin = property(fget=_get_soma_kdrRL_tmin, fset=_set_soma_kdrRL_tmin)

	def _set_soma_kdrRL_taumax(self, value):
		for seg in self.soma:
			seg.kdrRL.taumax = value
	def _get_soma_kdrRL_taumax(self):
		# Return only the middle soma parameter
		return self.soma(0.5).kdrRL.taumax
	soma_kdrRL_taumax = property(fget=_get_soma_kdrRL_taumax, fset=_set_soma_kdrRL_taumax)

	def _set_soma_kdrRL_mVh(self, value):
		for seg in self.soma:
			seg.kdrRL.mVh = value
	def _get_soma_kdrRL_mVh(self):
		# Return only the middle soma parameter
		return self.soma(0.5).kdrRL.mVh
	soma_kdrRL_mVh = property(fget=_get_soma_kdrRL_mVh, fset=_set_soma_kdrRL_mVh)

	# mAHP Setters & Getters:
	def _set_soma_mAHP_gcamax(self, value):
		for seg in self.soma:
			seg.mAHP.gcamax = value
	def _get_soma_mAHP_gcamax(self):
		# Return only the middle soma parameter
		return self.soma(0.5).mAHP.gcamax
	soma_mAHP_gcamax = property(fget=_get_soma_mAHP_gcamax, fset=_set_soma_mAHP_gcamax)

	def _set_soma_mAHP_gkcamax(self, value):
		for seg in self.soma:
			seg.mAHP.gkcamax = value
	def _get_soma_mAHP_gkcamax(self):
		# Return only the middle soma parameter
		return self.soma(0.5).mAHP.gkcamax
	soma_mAHP_gkcamax = property(fget=_get_soma_mAHP_gkcamax, fset=_set_soma_mAHP_gkcamax)

	def _set_soma_mAHP_taur(self, value):
		for seg in self.soma:
			seg.mAHP.taur = value
	def _get_soma_mAHP_taur(self):
		# Return only the middle soma parameter
		return self.soma(0.5).mAHP.taur
	soma_mAHP_taur = property(fget=_get_soma_mAHP_taur, fset=_set_soma_mAHP_taur)

	def _set_soma_mAHP_mtauca(self, value):
		for seg in self.soma:
			seg.mAHP.mtauca = value
	def _get_soma_mAHP_mtauca(self):
		# Return only the middle soma parameter
		return self.soma(0.5).mAHP.mtauca
	soma_mAHP_mtauca = property(fget=_get_soma_mAHP_mtauca, fset=_set_soma_mAHP_mtauca)

	def _set_soma_mAHP_mvhalfca(self, value):
		for seg in self.soma:
			seg.mAHP.mvhalfca = value
	def _get_soma_mAHP_mvhalfca(self):
		# Return only the middle soma parameter
		return self.soma(0.5).mAHP.mvhalfca
	soma_mAHP_mvhalfca = property(fget=_get_soma_mAHP_mvhalfca, fset=_set_soma_mAHP_mvhalfca)

	# gh Setters & Getters:
	def _set_soma_gh_ghbar(self, value):
		for seg in self.soma:
			seg.gh.ghbar = value
	def _get_soma_gh_ghbar(self):
		# Return only the middle soma parameter
		return self.soma(0.5).gh.ghbar
	soma_gh_ghbar = property(fget=_get_soma_gh_ghbar, fset=_set_soma_gh_ghbar)

	def _set_soma_gh_half(self, value):
		for seg in self.soma:
			seg.gh.half = value
	def _get_soma_gh_half(self):
		# Return only the middle soma parameter
		return self.soma(0.5).gh.half
	soma_gh_half = property(fget=_get_soma_gh_half, fset=_set_soma_gh_half)

	def _set_soma_gh_htau(self, value):
		for seg in self.soma:
			seg.gh.htau = value
	def _get_soma_gh_htau(self):
		# Return only the middle soma parameter
		return self.soma(0.5).gh.htau
	soma_gh_htau = property(fget=_get_soma_gh_htau, fset=_set_soma_gh_htau)

	# pas Setters & Getters:
	def _set_soma_pas_g(self, value):
		for seg in self.soma:
			seg.pas.g = value
	def _get_soma_pas_g(self):
		# Return only the middle soma parameter
		return self.soma(0.5).pas.g
	soma_pas_g = property(fget=_get_soma_pas_g, fset=_set_soma_pas_g)

	def _set_soma_pas_e(self, value):
		for seg in self.soma:
			seg.pas.e = value
	def _get_soma_pas_e(self):
		# Return only the middle soma parameter
		return self.soma(0.5).pas.e
	soma_pas_e = property(fget=_get_soma_pas_e, fset=_set_soma_pas_e)
	
	# Dendrites -
	# diam Setters & Getters:
	def _set_dendrite_diam(self, value):
		self.d1.diam = value
		self.d2.diam = value
		self.d3.diam = value
		self.d4.diam = value
	def _get_dendrite_diam(self):
		return self.d1.diam
	dendrite_diam = property(fget=_get_dendrite_diam, fset=_set_dendrite_diam)
	
	# L Setters & Getters:
	def _set_dendrite_L(self, value):
		self.d1.L = value
		self.d2.L = value
		self.d3.L = value
		self.d4.L = value
	def _get_dendrite_L(self):
		return self.d1.L
	dendrite_L = property(fget=_get_dendrite_L, fset=_set_dendrite_L)
	
	# cm Setters & Getters:
	def _set_dendrite_cm(self, value):
		self.d1.cm = value
		self.d2.cm = value
		self.d3.cm = value
		self.d4.cm = value
	def _get_dendrite_cm(self):
		return self.d1.cm
	dendrite_cm = property(fget=_get_dendrite_cm, fset=_set_dendrite_cm)
	
	# Ra Setters & Getters:
	def _set_dendrite_Ra(self, value):
		self.d1.Ra = value
		self.d2.Ra = value
		self.d3.Ra = value
		self.d4.Ra = value
	def _get_dendrite_Ra(self):
		return self.d1.Ra
	dendrite_Ra = property(fget=_get_dendrite_Ra, fset=_set_dendrite_Ra)
	
	# pas mechanism - g value:
	def _set_dendrite_pas_g(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		for seg in self.d1:
			seg.pas.g = value
		for seg in self.d2:
			seg.pas.g = value
		for seg in self.d3:
			seg.pas.g = value
		for seg in self.d4:
			seg.pas.g = value
	def _get_dendrite_pas_g(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1(0.5).pas.g
	dendrite_pas_g = property(fget=_get_dendrite_pas_g, fset=_set_dendrite_pas_g)

	# pas mechanism - e value:
	def _set_dendrite_pas_e(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		for seg in self.d1:
			seg.pas.e = value
		for seg in self.d2:
			seg.pas.e = value
		for seg in self.d3:
			seg.pas.e = value
		for seg in self.d4:
			seg.pas.e = value
	def _get_dendrite_pas_e(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1(0.5).pas.e
	dendrite_pas_e = property(fget=_get_dendrite_pas_e, fset=_set_dendrite_pas_e)
	
	# PIC conductance values are different on each dendrite compartment, so there are 4 getters and setters corresponding to each conductance
	# L_Ca_inact mechanism - gcabar value for dendrite 1:	
	def _set_dendrite_1_L_Ca_inact_gcabar(self, value):
		for seg in self.d1:
			seg.L_Ca_inact.gcabar = value
	def _get_dendrite_1_L_Ca_inact_gcabar(self):
		return self.d1(0.5).L_Ca_inact.gcabar
	dendrite_1_L_Ca_inact_gcabar = property(fget=_get_dendrite_1_L_Ca_inact_gcabar, fset=_set_dendrite_1_L_Ca_inact_gcabar)
	
	# L_Ca_inact mechanism - gcabar value for dendrite 2:	
	def _set_dendrite_2_L_Ca_inact_gcabar(self, value):
		for seg in self.d2:
			seg.L_Ca_inact.gcabar = value
	def _get_dendrite_2_L_Ca_inact_gcabar(self):
		return self.d2(0.5).L_Ca_inact.gcabar
	dendrite_2_L_Ca_inact_gcabar = property(fget=_get_dendrite_2_L_Ca_inact_gcabar, fset=_set_dendrite_2_L_Ca_inact_gcabar)
	
	# L_Ca_inact mechanism - gcabar value for dendrite 3:	
	def _set_dendrite_3_L_Ca_inact_gcabar(self, value):
		for seg in self.d3:
			seg.L_Ca_inact.gcabar = value
	def _get_dendrite_3_L_Ca_inact_gcabar(self):
		return self.d3(0.5).L_Ca_inact.gcabar
	dendrite_3_L_Ca_inact_gcabar = property(fget=_get_dendrite_3_L_Ca_inact_gcabar, fset=_set_dendrite_3_L_Ca_inact_gcabar)
	
	# L_Ca_inact mechanism - gcabar value for dendrite 4:	
	def _set_dendrite_4_L_Ca_inact_gcabar(self, value):
		for seg in self.d4:
			seg.L_Ca_inact.gcabar = value
	def _get_dendrite_4_L_Ca_inact_gcabar(self):
		return self.d4(0.5).L_Ca_inact.gcabar
	dendrite_4_L_Ca_inact_gcabar = property(fget=_get_dendrite_4_L_Ca_inact_gcabar, fset=_set_dendrite_4_L_Ca_inact_gcabar)
	
	# L_Ca_inact mechanism - theta_m value:
	def _set_dendrite_L_Ca_inact_theta_m(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		for seg in self.d1:
			seg.L_Ca_inact.theta_m = value
		for seg in self.d2:
			seg.L_Ca_inact.theta_m = value
		for seg in self.d3:
			seg.L_Ca_inact.theta_m = value
		for seg in self.d4:
			seg.L_Ca_inact.theta_m = value
	def _get_dendrite_L_Ca_inact_theta_m(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1(0.5).L_Ca_inact.theta_m
	dendrite_L_Ca_inact_theta_m = property(fget=_get_dendrite_L_Ca_inact_theta_m, fset=_set_dendrite_L_Ca_inact_theta_m)

	# L_Ca_inact mechanism - tau_m value:
	def _set_dendrite_L_Ca_inact_tau_m(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		for seg in self.d1:
			seg.L_Ca_inact.tau_m = value
		for seg in self.d2:
			seg.L_Ca_inact.tau_m = value
		for seg in self.d3:
			seg.L_Ca_inact.tau_m = value
		for seg in self.d4:
			seg.L_Ca_inact.tau_m = value
	def _get_dendrite_L_Ca_inact_tau_m(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1(0.5).L_Ca_inact.tau_m
	dendrite_L_Ca_inact_tau_m = property(fget=_get_dendrite_L_Ca_inact_tau_m, fset=_set_dendrite_L_Ca_inact_tau_m)

	# L_Ca_inact mechanism - theta_h value:
	def _set_dendrite_L_Ca_inact_theta_h(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		for seg in self.d1:
			seg.L_Ca_inact.theta_h = value
		for seg in self.d2:
			seg.L_Ca_inact.theta_h = value
		for seg in self.d3:
			seg.L_Ca_inact.theta_h = value
		for seg in self.d4:
			seg.L_Ca_inact.theta_h = value
	def _get_dendrite_L_Ca_inact_theta_h(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1(0.5).L_Ca_inact.theta_h
	dendrite_L_Ca_inact_theta_h = property(fget=_get_dendrite_L_Ca_inact_theta_h, fset=_set_dendrite_L_Ca_inact_theta_h)
	
	# L_Ca_inact mechanism - tau_h value:
	def _set_dendrite_L_Ca_inact_tau_h(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		for seg in self.d1:
			seg.L_Ca_inact.tau_h = value
		for seg in self.d2:
			seg.L_Ca_inact.tau_h = value
		for seg in self.d3:
			seg.L_Ca_inact.tau_h = value
		for seg in self.d4:
			seg.L_Ca_inact.tau_h = value
	def _get_dendrite_L_Ca_inact_tau_h(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1(0.5).L_Ca_inact.tau_h
	dendrite_L_Ca_inact_tau_h = property(fget=_get_dendrite_L_Ca_inact_tau_h, fset=_set_dendrite_L_Ca_inact_tau_h)
	
	# L_Ca_inact mechanism - kappa_h value:
	def _set_dendrite_L_Ca_inact_kappa_h(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		for seg in self.d1:
			seg.L_Ca_inact.kappa_h = value
		for seg in self.d2:
			seg.L_Ca_inact.kappa_h = value
		for seg in self.d3:
			seg.L_Ca_inact.kappa_h = value
		for seg in self.d4:
			seg.L_Ca_inact.kappa_h = value
	def _get_dendrite_L_Ca_inact_kappa_h(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1(0.5).L_Ca_inact.kappa_h
	dendrite_L_Ca_inact_kappa_h = property(fget=_get_dendrite_L_Ca_inact_kappa_h, fset=_set_dendrite_L_Ca_inact_kappa_h)
	
	# gh mechanism - ghbar value:
	def _set_dendrite_gh_ghbar(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		for seg in self.d1:
			seg.gh.ghbar = value
		for seg in self.d2:
			seg.gh.ghbar = value
		for seg in self.d3:
			seg.gh.ghbar = value
		for seg in self.d4:
			seg.gh.ghbar = value
	def _get_dendrite_gh_ghbar(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1(0.5).gh.ghbar
	dendrite_gh_ghbar = property(fget=_get_dendrite_gh_ghbar, fset=_set_dendrite_gh_ghbar)
	
	# gh mechanism - half value:
	def _set_dendrite_gh_half(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		for seg in self.d1:
			seg.gh.half = value
		for seg in self.d2:
			seg.gh.half = value
		for seg in self.d3:
			seg.gh.half = value
		for seg in self.d4:
			seg.gh.half = value
	def _get_dendrite_gh_half(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1(0.5).gh.half
	dendrite_gh_half = property(fget=_get_dendrite_gh_half, fset=_set_dendrite_gh_half)
	
	# gh mechanism - htau value:
	def _set_dendrite_gh_htau(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		for seg in self.d1:
			seg.gh.htau = value
		for seg in self.d2:
			seg.gh.htau = value
		for seg in self.d3:
			seg.gh.htau = value
		for seg in self.d4:
			seg.gh.htau = value
	def _get_dendrite_gh_htau(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1(0.5).gh.htau
	dendrite_gh_htau = property(fget=_get_dendrite_gh_htau, fset=_set_dendrite_gh_htau)
	
	# Getters and Setters for dendritic synapses
	# Correlated Inputs - Excitatory Inputs:
	def _get_correlated_e_rev_exc(self):			# Reversal Potential
		return self.d1_correlated_exc.e
	def _set_correlated_e_rev_exc(self, value):
		self.soma_correlated_exc.e = value
		self.d1_correlated_exc.e = value
		self.d2_correlated_exc.e = value
		self.d3_correlated_exc.e = value
		self.d4_correlated_exc.e = value
	correlated_e_rev_exc = property(fget=_get_correlated_e_rev_exc, fset=_set_correlated_e_rev_exc)
	
	def _get_correlated_tau(self):					# Decay time (ms)
		return self.d1_correlated_exc.tau
	def _set_correlated_tau(self, value):
		self.soma_correlated_exc.tau = value
		self.d1_correlated_exc.tau = value
		self.d2_correlated_exc.tau = value
		self.d3_correlated_exc.tau = value
		self.d4_correlated_exc.tau = value
	correlated_tau = property(fget=_get_correlated_tau, fset=_set_correlated_tau)
	
	# Independent Inputs - Excitatory Inputs:
	def _get_independent_e_rev_exc(self):			# Reversal Potential
		return self.d1_independent_exc.e
	def _set_independent_e_rev_exc(self, value):
		self.d1_independent_exc.e = value
		self.d2_independent_exc.e = value
		self.d3_independent_exc.e = value
		self.d4_independent_exc.e = value
	independent_e_rev_exc = property(fget=_get_independent_e_rev_exc, fset=_set_independent_e_rev_exc)

	def _get_independent_tau1(self):					# Rise time (ms)
		return self.d1_independent_exc.tau1
	def _set_independent_tau1(self, value):
		# Excitatory Independent Inputs
		self.d1_independent_exc.tau1 = value
		self.d2_independent_exc.tau1 = value
		self.d3_independent_exc.tau1 = value
		self.d4_independent_exc.tau1 = value
	independent_tau1 = property(fget=_get_independent_tau1, fset=_set_independent_tau1)
	
	def _get_independent_tau2(self):					# Decay time (ms)
		return self.d1_independent_exc.tau2
	def _set_independent_tau2(self, value):
		# Excitatory Independent Inputs
		self.d1_independent_exc.tau2 = value
		self.d2_independent_exc.tau2 = value
		self.d3_independent_exc.tau2 = value
		self.d4_independent_exc.tau2 = value
	independent_tau2 = property(fget=_get_independent_tau2, fset=_set_independent_tau2)
	
	def _get_soma_bias_current_amp(self):			# Getters/Setters for the soma bias current (nA)
		return self.soma_bias.amp
	def _set_soma_bias_current_amp(self, value):
		self.soma_stim.bias = value
	soma_bias_current_amp = property(fget=_get_soma_bias_current_amp, fset=_set_soma_bias_current_amp)
	
	def _get_dendrite_bias_current_amp(self):		# Getters/Setters for the dendrite bias current (nA)
		return self.d1_bias_current.amp
	def _set_dendrite_bias_current_amp(self, value):
		self.d1_bias_current.amp = value
		self.d2_bias_current.amp = value
		self.d3_bias_current.amp = value
		self.d4_bias_current.amp = value
	dendrite_bias_current_amp = property(fget=_get_dendrite_bias_current_amp, fset=_set_dendrite_bias_current_amp)
	
	# Getters and Setters for dendritic stochastic conductance - simulates synaptic bombardment from other structures 
	# Gfluct mechanism - ge0 value:
	def _set_dendrite_Gfluct_g_e0(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		self.d1_Synaptic_Bombardment.ge0 = value
		self.d2_Synaptic_Bombardment.ge0 = value
		self.d3_Synaptic_Bombardment.ge0 = value
		self.d4_Synaptic_Bombardment.ge0 = value
		
	def _get_dendrite_Gfluct_g_e0(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1_Synaptic_Bombardment.ge0
	dendrite_Gfluct_ge0 = property(fget=_get_dendrite_Gfluct_g_e0, fset=_set_dendrite_Gfluct_g_e0)
	
	# Gfluct mechanism - gi0 value:
	def _set_dendrite_Gfluct_g_i0(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		self.d1_Synaptic_Bombardment.gi0 = value
		self.d2_Synaptic_Bombardment.gi0 = value
		self.d3_Synaptic_Bombardment.gi0 = value
		self.d4_Synaptic_Bombardment.gi0 = value
		
	def _get_dendrite_Gfluct_g_i0(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1_Synaptic_Bombardment.gi0
	dendrite_Gfluct_gi0 = property(fget=_get_dendrite_Gfluct_g_i0, fset=_set_dendrite_Gfluct_g_i0)
	
	# Gfluct mechanism - stde value:
	def _set_dendrite_Gfluct_std_e(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		self.d1_Synaptic_Bombardment.stde = value
		self.d2_Synaptic_Bombardment.stde = value
		self.d3_Synaptic_Bombardment.stde = value
		self.d4_Synaptic_Bombardment.stde = value
		
	def _get_dendrite_Gfluct_std_e(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1_Synaptic_Bombardment.stde
	dendrite_Gfluct_stde = property(fget=_get_dendrite_Gfluct_std_e, fset=_set_dendrite_Gfluct_std_e)
	
	# Gfluct mechanism - stdi value:
	def _set_dendrite_Gfluct_std_i(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		self.d1_Synaptic_Bombardment.stdi = value
		self.d2_Synaptic_Bombardment.stdi = value
		self.d3_Synaptic_Bombardment.stdi = value
		self.d4_Synaptic_Bombardment.stdi = value
		
	def _get_dendrite_Gfluct_std_i(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1_Synaptic_Bombardment.stdi
	dendrite_Gfluct_stdi = property(fget=_get_dendrite_Gfluct_std_i, fset=_set_dendrite_Gfluct_std_i)
	
	# Gfluct mechanism - tau_e value:
	def _set_dendrite_Gfluct_tau_e(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		self.d1_Synaptic_Bombardment.tau_e = value
		self.d2_Synaptic_Bombardment.tau_e = value
		self.d3_Synaptic_Bombardment.tau_e = value
		self.d4_Synaptic_Bombardment.tau_e = value
		
	def _get_dendrite_Gfluct_tau_e(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1_Synaptic_Bombardment.tau_e
	dendrite_Gfluct_tau_e = property(fget=_get_dendrite_Gfluct_tau_e, fset=_set_dendrite_Gfluct_tau_e)

	# Gfluct mechanism - tau_i value:
	def _set_dendrite_Gfluct_tau_i(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		self.d1_Synaptic_Bombardment.tau_i = value
		self.d2_Synaptic_Bombardment.tau_i = value
		self.d3_Synaptic_Bombardment.tau_i = value
		self.d4_Synaptic_Bombardment.tau_i = value
		
	def _get_dendrite_Gfluct_tau_i(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1_Synaptic_Bombardment.tau_i
	dendrite_Gfluct_tau_i = property(fget=_get_dendrite_Gfluct_tau_i, fset=_set_dendrite_Gfluct_tau_i)
	
	# Gfluct mechanism - E_e value:
	def _set_dendrite_Gfluct_E_e(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		self.d1_Synaptic_Bombardment.E_e = value
		self.d2_Synaptic_Bombardment.E_e = value
		self.d3_Synaptic_Bombardment.E_e = value
		self.d4_Synaptic_Bombardment.E_e = value
		
	def _get_dendrite_Gfluct_E_e(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1_Synaptic_Bombardment.E_e
	dendrite_Gfluct_E_e = property(fget=_get_dendrite_Gfluct_E_e, fset=_set_dendrite_Gfluct_E_e)
	
	# Gfluct mechanism - E_i value:
	def _set_dendrite_Gfluct_E_i(self, value):
		# Set each dendrite on the motoneuron to have the same parameter value
		self.d1_Synaptic_Bombardment.E_i = value
		self.d2_Synaptic_Bombardment.E_i = value
		self.d3_Synaptic_Bombardment.E_i = value
		self.d4_Synaptic_Bombardment.E_i = value
		
	def _get_dendrite_Gfluct_E_i(self):
		# Return only the first dendrite parameter, since other dendrites use same value
		return self.d1_Synaptic_Bombardment.E_i
	dendrite_Gfluct_E_i = property(fget=_get_dendrite_Gfluct_E_i, fset=_set_dendrite_Gfluct_E_i)

class Spinal_Motoneuron_Synaptic_Bombardment_Point_Process_Type(NativeCellType):
	
	# Default parameters for the neurons:
	default_parameters = {'soma_diam': 22.0, 'soma_L': 2952.0, 'soma_Ra': 0.001, 'soma_cm': 1.35546, 'soma_ek': -80.0, 'v_thres' : -20,
						 'dendrite_diam': 8.73071, 'dendrite_L': 1794.13, 'dendrite_Ra': 51.038, 'dendrite_cm': 0.867781}
	# Soma default parameters:
	# na3rp:
	default_parameters['soma_na3rp_gbar'] = 0.01
	default_parameters['soma_na3rp_sh'] = 1.0
	default_parameters['soma_na3rp_ar'] = 1.0
	default_parameters['soma_na3rp_qinf'] = 8.0
	default_parameters['soma_na3rp_thinf'] = -50.0
	
	# naps:
	default_parameters['soma_naps_gbar'] = 2.6e-5
	default_parameters['soma_naps_sh'] = 5.0
	default_parameters['soma_naps_ar'] = 1.0
	default_parameters['soma_naps_vslope'] = 5.0
	default_parameters['soma_naps_asvh'] = -90.0
	default_parameters['soma_naps_bsvh'] = -22.0
	
	# kdrRL:
	default_parameters['soma_kdrRL_gMax'] = 0.015
	default_parameters['soma_kdrRL_tmin'] = 0.8
	default_parameters['soma_kdrRL_taumax'] = 20.0
	default_parameters['soma_kdrRL_mVh'] = -21.0
	
	# mAHP:
	default_parameters['soma_mAHP_gcamax'] = 6.4e-06
	default_parameters['soma_mAHP_gkcamax'] = 0.00045
	default_parameters['soma_mAHP_taur'] = 90.0
	default_parameters['soma_mAHP_mtauca'] = 2.0
	default_parameters['soma_mAHP_mvhalfca'] = -20.0
	
	# gh:
	default_parameters['soma_gh_ghbar'] = 3e-5
	default_parameters['soma_gh_half'] = -77.0
	default_parameters['soma_gh_htau'] = 30.0
	
	# pas:
	default_parameters['soma_pas_g'] = 8.11e-05
	default_parameters['soma_pas_e'] = -71.0
	
	# Dendrite default parameters:
	# pas:
	default_parameters['dendrite_pas_g'] = 7.93e-05
	default_parameters['dendrite_pas_e'] = -71.0
	
	# L_Ca_inact:
	default_parameters['dendrite_1_L_Ca_inact_gcabar'] = 5.2e-6
	default_parameters['dendrite_2_L_Ca_inact_gcabar'] = 4.25e-6
	default_parameters['dendrite_3_L_Ca_inact_gcabar'] = 6.62e-6
	default_parameters['dendrite_4_L_Ca_inact_gcabar'] = 8.03e-6
	default_parameters['dendrite_L_Ca_inact_theta_m'] = -42.0
	default_parameters['dendrite_L_Ca_inact_tau_m'] = 40.0
	default_parameters['dendrite_L_Ca_inact_theta_h'] = 10.0
	default_parameters['dendrite_L_Ca_inact_tau_h'] = 2500.0
	default_parameters['dendrite_L_Ca_inact_kappa_h'] = 5.0
	
	# gh:
	default_parameters['dendrite_gh_ghbar'] = 3e-5
	default_parameters['dendrite_gh_half'] = -77.0
	default_parameters['dendrite_gh_htau'] = 30.0
	
	# Dendritic synaptic parameters - 
	# Correlated Input Parameters:
	default_parameters['correlated_e_rev_exc'] = 0.0
	default_parameters['correlated_tau'] = 25.0
	
	# Independent Input Parameters:
	default_parameters['independent_e_rev_exc'] = 0.0
	default_parameters['independent_tau1'] = 4.0
	default_parameters['independent_tau2'] = 20.0
	
	# Bias Current Values
	default_parameters['soma_bias_current_amp'] = 0.0
	default_parameters['dendrite_bias_current_amp'] = 0.0
	
	# Stochastic conductance parameters to simulate inputs from other structures
	default_parameters['dendrite_Gfluct_ge0'] = 0.15			# mean value of excitatory conductance - representing input from other neural structures
	default_parameters['dendrite_Gfluct_stde'] = 0.25			# standard deviation of excitatory conductance - representing input from other neural structures
	default_parameters['dendrite_Gfluct_gi0'] = 0.3			# mean value of inhibitory conductance - representing input from other neural structures
	default_parameters['dendrite_Gfluct_stdi'] = 0.25			# standard deviation of inhibitory conductance - representing input from other neural structures
	default_parameters['dendrite_Gfluct_tau_e'] = 2.728			# Excitatory Current Time constant - In Powers et al. 2012 Paper value is 2.728 ms to simulate uncorrelated/independent inputs
	default_parameters['dendrite_Gfluct_tau_i'] = 10.49			# Inhibitory Current Time constant - In Powers et al. 2012 Paper value is 10.49 ms to simulate uncorrelated/independent inputs
	default_parameters['dendrite_Gfluct_E_e'] = 0.0
	default_parameters['dendrite_Gfluct_E_i'] = -75.0
	
	default_initial_values = {'v': -65.0}			# Can record soma voltage and synaptic currents on each dendrite
	recordable = ['soma(0.5).v', 'd1(0.5).v', 'd2(0.5).v', 'd3(0.5).v', 'd4(0.5).v', 'soma_correlated_exc.i', 'd1_correlated_exc.i', 'd2_correlated_exc.i', 'd3_correlated_exc.i', 'd4_correlated_exc.i', 'd1_independent_exc.i', 'd2_independent_exc.i', 'd3_independent_exc.i', 'd4_independent_exc.i', 'd1_Synaptic_Bombardment.i', 'd2_Synaptic_Bombardment.i', 'd3_Synaptic_Bombardment.i', 'd4_Synaptic_Bombardment.i', 'd1_Synaptic_Bombardment.g_e', 'd2_Synaptic_Bombardment.g_e', 'd3_Synaptic_Bombardment.g_e', 'd4_Synaptic_Bombardment.g_e', 'd1_Synaptic_Bombardment.g_i', 'd2_Synaptic_Bombardment.g_i', 'd3_Synaptic_Bombardment.g_i', 'd4_Synaptic_Bombardment.g_i']
	units = {'soma(0.5).v' : 'mV', 'd1(0.5).v' : 'mV', 'd2(0.5).v' : 'mV', 'd3(0.5).v' : 'mV', 'd4(0.5).v' : 'mV', 'soma_correlated_exc.i' : 'nA', 'd1_correlated_exc.i' : 'nA', 'd2_correlated_exc.i' : 'nA', 'd3_correlated_exc.i' : 'nA', 'd4_correlated_exc.i' : 'nA', 'd1_independent_exc.i' : 'nA', 'd2_independent_exc.i' : 'nA', 'd3_independent_exc.i' : 'nA', 'd4_independent_exc.i' : 'nA', 'd1_Synaptic_Bombardment.i' : 'nA', 'd2_Synaptic_Bombardment.i' : 'nA', 'd3_Synaptic_Bombardment.i' : 'nA', 'd4_Synaptic_Bombardment.i' : 'nA', 'd1_Synaptic_Bombardment.g_e' : 'uS', 'd2_Synaptic_Bombardment.g_e' : 'uS', 'd3_Synaptic_Bombardment.g_e' : 'uS', 'd4_Synaptic_Bombardment.g_e' : 'uS', 'd1_Synaptic_Bombardment.g_i' : 'uS', 'd2_Synaptic_Bombardment.g_i' : 'uS', 'd3_Synaptic_Bombardment.g_i' : 'uS', 'd4_Synaptic_Bombardment.g_i' : 'uS'}    
	receptor_types = ['soma_correlated_exc', 'd1_correlated_exc', 'd2_correlated_exc', 'd3_correlated_exc', 'd4_correlated_exc', 'd1_independent_exc', 'd2_independent_exc', 'd3_independent_exc', 'd4_independent_exc']
	model = Spinal_Motoneuron_Synaptic_Bombardment_Point_Process

class Spinal_Interneuron(object):
	
	def __init__(self, **parameters):
		
		# Create single compartment Destexhe Interneuron cell section, i.e. soma section
		self.soma = Section(L=parameters['L'], diam=parameters['diam'], nseg=parameters['nseg'], Ra=parameters['Ra'], cm=parameters['cm'],
							mechanisms=())
							#mechanisms=(Mechanism('hh')))
							#mechanisms=(Mechanism('interneuron_i_leak'), Mechanism('interneuron_i_na'), Mechanism('interneuron_i_k')))
		
		# Insert the basic HH mechanisms to the soma
		self.soma.insert('hh')
		
		# Set the default mechanism parameters
		for seg in self.soma:
			seg.hh.gnabar = 120e-3
			seg.hh.gkbar = 100e-3
			seg.hh.gl = 0.51e-3
			seg.hh.el = -64
		
		# Add bias current to neuron model - current amplitude is in terms of original model paper, nA 
		self.soma_bias_current = h.IClamp(0.5, sec=self.soma)
		self.soma_bias_current.delay = 0
		self.soma_bias_current.dur = 1e12
		self.soma_bias_current.amp = parameters['bias_current_amp']	# nA
		
		# Append the gaussian IClamps to global list so they can be updated in the main script
		GV.Spinal_Interneuron_bias_current_iclamps.append(self.soma_bias_current)
		
		# Add excitatory synapse to the cell, i.e. add to the soma section
		self.synapse_exc = h.ExpSyn(0.5, sec=self.soma)
		self.synapse_exc.e = parameters['e_rev']
		self.synapse_exc.tau = parameters['tau']
		
		# needed for PyNN
		self.source_section = self.soma
		self.source = self.soma(0.5)._ref_v
		self.rec = h.NetCon(self.source, None, sec=self.source_section)		# Needed to clear the simulator
		self.spike_times = h.Vector(0)
		self.parameter_names = ('L', 'diam', 'nseg', 'Ra', 'cm', 'bias_current_amp', 'e_rev', 'tau')
		self.traces = {}
		self.recording_time = False

	L = _new_property('soma', 'L')
	diam = _new_property('soma', 'diam')
	nseg = _new_property('soma', 'nseg')
	Ra = _new_property('soma', 'Ra')
	cm = _new_property('soma', 'cm')
	bias_current_amp = _new_property('soma_bias_current', 'amp')
	
	def area(self):
		"""Membrane area in µm²"""
		return pi * self.soma.L * self.soma.diam
	
	def memb_init(self):
		for seg in self.soma:
			seg.v = self.v_init

class Spinal_Interneuron_Type(NativeCellType):
	default_parameters = {'L': 100, 'diam': 500, 'nseg': 1, 'Ra': 150, 'cm': 1, 'bias_current_amp': 0, 'e_rev': -10, 'tau': 5}
	default_initial_values = {'v': -68.0}
	recordable = ['soma(0.5).v', 'synapse_exc.i']
	units = {'soma(0.5).v' : 'mV', 'synapse_exc.i' : 'nA'}    
	receptor_types = ['synapse_exc']
	model = Spinal_Interneuron

# Function to generate a skewed parameter values for a single parameter value
def generate_skewed_parameter_values(Pop_size, lower_parameter_bound, upper_parameter_bound, nonlin_ind):
	
	# Set Up Nonlinear Interpolation Function - skew
	nonlin_ind = motoneuron_pool_parameter_bounds['nonlin_ind']
	fraction = np.zeros(Pop_size)
	for cell_index in range(Pop_size):
		fraction[cell_index] = float(cell_index)/(Pop_size - 1)
		if nonlin_ind >= 0:
			skew = fraction**(1 + nonlin_ind)
		else:
			skew = 1 - (1-fraction)**(1 - nonlin_ind)

	# Calculate the skewed parameter
	parameter_values = lower_parameter_bound + (upper_parameter_bound - lower_parameter_bound)*skew

	# return the skewed parameter values
	return parameter_values
	
# Function to generate the distribution of parameter values across the motoneurons in the pool
def generate_motoneuron_population_parameters(Pop_size, motoneuron_pool_parameter_bounds):
	
	# Create dictionary for holding the parameters
	motoneuron_parameters = {}
	
	# Set Up Nonlinear Interpolation Function - skew
	nonlin_ind = motoneuron_pool_parameter_bounds['nonlin_ind']
	fraction = np.zeros(Pop_size)
	for cell_index in range(Pop_size):
		fraction[cell_index] = float(cell_index)/(Pop_size - 1)
		if nonlin_ind >= 0:
			skew = fraction**(1 + nonlin_ind)
		else:
			skew = 1 - (1-fraction)**(1 - nonlin_ind)
	
	# Get the motoneuron sections to update parameters for
	motoneuron_sections = list(motoneuron_pool_parameter_bounds.keys())
	
	# Increment over the sections
	for i in np.arange(1,len(motoneuron_sections)):         # Skip the first skew parameter
		section_parameters = list(motoneuron_pool_parameter_bounds[motoneuron_sections[i]].keys())
		# Increment over the parameters in the section
		for j in np.arange(0,len(section_parameters)):  
			section_subparameters = list(motoneuron_pool_parameter_bounds[motoneuron_sections[i]][section_parameters[j]].keys())
			# Checking/setting cell geometry parameters
			if section_subparameters == ['lower', 'upper']:
				parameter_key = motoneuron_sections[i]+"_"+section_parameters[j]
				lower_parameter_bound = motoneuron_pool_parameter_bounds[motoneuron_sections[i]][section_parameters[j]]['lower']
				upper_parameter_bound = motoneuron_pool_parameter_bounds[motoneuron_sections[i]][section_parameters[j]]['upper']
				
				# Calculate the skewed parameters
				parameter_values = lower_parameter_bound + (upper_parameter_bound - lower_parameter_bound)*skew
				
				# Add the parameter values to the parameter dictionary
				motoneuron_parameters[parameter_key] = parameter_values
				
			else:
				for k in np.arange(0,len(section_subparameters)):
					mechanism_parameters = list(motoneuron_pool_parameter_bounds[motoneuron_sections[i]][section_parameters[j]][section_subparameters[k]].keys())
					parameter_key = motoneuron_sections[i]+"_"+section_parameters[j]+"_"+section_subparameters[k]
					lower_parameter_bound = motoneuron_pool_parameter_bounds[motoneuron_sections[i]][section_parameters[j]][section_subparameters[k]]['lower']
					upper_parameter_bound = motoneuron_pool_parameter_bounds[motoneuron_sections[i]][section_parameters[j]][section_subparameters[k]]['upper']
					
					# Calculate the skewed parameters
					parameter_values = lower_parameter_bound + (upper_parameter_bound - lower_parameter_bound)*skew
					
					# Add the parameter values to the parameter dictionary
					motoneuron_parameters[parameter_key] = parameter_values
					
	# Return the updated motoneuron population
	return motoneuron_parameters

# Motoneuron parameter bounds for the pool
motoneuron_pool_parameter_bounds = {
	"nonlin_ind" : 0.8,
	"soma"        : {
		"diam"        : {"lower" : 22, "upper" : 30},
		"L"            : {"lower" : 2952, "upper" : 3665},   
		"cm"        : {"lower" : 1.35546, "upper" : 1.87853},   
		"pas"       : {
			"g"    : {"lower" : 8.11e-05, "upper" : 3.77e-04}, 
			"e"    : {"lower" : -71, "upper" : -72}
		},
		"na3rp"        : {
			"gbar"        : {"lower" : 0.01, "upper" : 0.022},
			"sh"		  : {"lower" : 1.0, "upper" : 1.0},
			"ar"		  : {"lower" : 1.0, "upper" : 1.0},
			"qinf"		  : {"lower" : 8.0, "upper" : 8.0},
			"thinf"		  : {"lower" : -50.0, "upper" : -50.0}
		},
		"naps"        : {
			"gbar"        : {"lower" : 2.60e-05, "upper" : 2.00e-05},
			"sh"          : {"lower" : 5.0, "upper" : 5.0},
			"ar"		  : {"lower" : 1.0, "upper" : 1.0},
			"vslope"	  : {"lower" : 5.0, "upper" : 5.0},
			"asvh"	      : {"lower" : -90.0, "upper" : -90.0},
			"bsvh"	      : {"lower" : -22.0, "upper" : -22.0}
		},
		"kdrRL"        : {
			"gMax"        : {"lower" : 0.015, "upper" : 0.02},
			"mVh"         : {"lower" : -21, "upper" : -21},
			"tmin"        : {"lower" : 0.8, "upper" : 0.8},
			"taumax"      : {"lower" : 20.0, "upper" : 20.0}
		},
		"mAHP"        : {
			"gcamax"      : {"lower" : 6.40e-06, "upper" : 1.05e-05},
			"gkcamax"     : {"lower" : 0.00045, "upper" : 0.0006},
			"taur"        : {"lower" : 90, "upper" : 40},
			"mvhalfca"    : {"lower" : -20, "upper" : -20},
			"mtauca"      : {"lower" : 2, "upper" : 2}
		},
		"gh"        : {
			"ghbar"        : {"lower" : 3.00e-05, "upper" : 2.30e-04},
			"htau"         : {"lower" : 30, "upper" : 30}
		},
	},
	"dendrite"    : {
		"diam"    : {"lower" : 8.73071, "upper" : 11.9055},
		"L"       : {"lower" : 1794.13, "upper" : 2226.91},
		"Ra"      : {"lower" : 51.038, "upper" : 40.755},
		"cm"      : {"lower" : 0.867781, "upper" : 0.880407},
		"pas"    : {
			"g"        : {"lower" : 7.93e-05, "upper" : 1.75e-04},
			"e"        : {"lower" : -71, "upper" : -72}
		},
		"gh"    : {
			"ghbar"    : {"lower" : 3.00e-05, "upper" : 2.30e-04},
			"htau"     : {"lower" : 30, "upper" : 30}
		},
		"L_Ca_inact"    : {
			"theta_m"        : {"lower" : -42, "upper" : -32},
			"tau_m"          : {"lower" : 40, "upper" : 40},
			"theta_h"        : {"lower" : 10, "upper" : -10},
			"tau_h"          : {"lower" : 2500, "upper" : 2500},
			"kappa_h"        : {"lower" : 5, "upper" : 5}
		}
	},
	"dendrite_1"    : {
		"L_Ca_inact"    : {
			"gcabar"    : {"lower" : 5.20e-06, "upper" : 5.44e-06}
		}
	},
	"dendrite_2"    : {
		"L_Ca_inact"    : {
			"gcabar"    : {"lower" : 4.25e-06, "upper" : 6.24e-06}
		}
	},
	"dendrite_3"    : {
		"L_Ca_inact"    : {
			"gcabar"    : {"lower" : 6.62e-06, "upper" : 7.04e-06}
		}
	},
	"dendrite_4"    : {
		"L_Ca_inact"    : {
			"gcabar"    : {"lower" : 8.03e-06, "upper" : 8.24e-06}
		}
	}
}