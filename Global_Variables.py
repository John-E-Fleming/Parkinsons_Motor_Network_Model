# -*- coding: utf-8 -*-
"""
Created on Wed April 03 14:27:26 2019

Description: Define arrays for holding global references to the GPe DBS 
			 stimulation signals.

			Paper Ref . . . 
.
			

@author: John Fleming, john.fleming@ucdconnect.ie
"""

# Define DBS GPe signal globally
GPe_DBS_Signals = []
GPe_stimulation_iclamps = []

# Define global list of MU dendritic bias current signals
MU_dendritic_bias_current_iclamps = []

# Define global list of noise currents for spinal interneurons
Spinal_Interneuron_bias_current_iclamps = []

# Define global list of MU dendritic noisy conductances
MU_dendritic_noisy_conductances = []

# List to append the random streams to (so won't disappear after initialization) for MU independent input currents on dendrites
rslist = []

# Variable to hold the offset of the random streams
random_stream_offset = 0

# Index variable for generating the independent random stream for each neuron
stream_index = 0