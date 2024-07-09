# Parkinsonian Motor Network Model Code
---

## General Information

This repository contains python code for simulating a Hodgkin-Huxley based neural network model of the parkinsonian motor network during deep brain stimulation (DBS). The motor network is comprised of a model of the cortico-basal ganglia network coupled to a motoneuron pool. The model simulates clinically relevant output signals from the motor network including a model of the local field potential in the vicinity of the DBS electrode, and electromyography and force signals generated due to activation of the motoneuron pool. The model can be used to explore tremor- and beta-based closed-loop DBS control strategies. 

<br/>

<br/>

![alt text](https://content.cld.iop.org/journals/1741-2552/20/5/056029/revision2/jneacfbfaf2_hr.jpg)

<sup><sub>Fleming, J.E., Senneff, S. and Lowery, M.M., 2023. Multivariable closed-loop control of deep brain stimulation for Parkinson’s disease. Journal of Neural Engineering, 20(5), p.056029. </sup></sub>

<br/>
<br/>

## Citations
The code contained in this repository for simulation of the motor network model and multivariable control DBS approaches is companion to the paper:  
> "Multivariable closed-loop control of deep brain stimulation for Parkinson’s disease"  
>   John E. Fleming, Sageanne Senneff, Madeleine M. Lowery
>    October 4, 2023 : 056029  
>    DOI: 10.1088/1741-2552/acfbfa  

which should be cited for academic use of this code.  
<br/>

A journal paper describing the model has been published at https://iopscience.iop.org/article/10.1088/1741-2552/acfbfa/meta. If using the model, reference it's original publication as follows:

Fleming, J.E., Senneff, S. and Lowery, M.M., 2023. Multivariable closed-loop control of deep brain stimulation for Parkinson’s disease. Journal of Neural Engineering, 20(5), p.056029.

Model Requirements: - Model is simulated using PyNN with NEURON as it's backend simulator, thus follow their installation instructions at: 1) Neuron - https://www.neuron.yale.edu/neuron/download 2) PyNN - https://pypi.org/project/PyNN/ - http://neuralensemble.org/docs/PyNN/

Model Setup: 1) Copy the included PyNN files from the downloaded model folder to their corresponding location on your computer (i.e. the directory of your PyNN instatllation - Updated PyNN files are needed for correct simulation of the multicompartmental cortical neurons and for loading model simulations from a presimulated steady state.
			 2) Compile the NEURON model mod files using either mknrndll or nrnivmodl, for windows or Linux, respectively.
			 3) Run run_CBG_MU_Pool_Model_to_SS.py Example: 1) From the command line/terminal navigate to the folder containing the model. 2) Execute "python run_CBG_Model_to_SS.py neuron"

    Explanation: There is an initial transient period in the model (~8 seconds). This model simulation runs the model for 
    the transient period and creates a binary file (steady_state.bin) at the end of the simulation. This binary file captures 
    the state of the model at the end of this transient simulation (i.e. after the model has reasched the steady state)
    Subsequent runs of the model can use run_CBG_MU_Pool_Model_from_SS_Dual_Amplitude_Pulse_Duration_Modulation.py to load the previously saved model steady state and run a 
    model simulation from this steady state point, and subsequently simulates closed-loop DBS.

Running the Model: - Once the steady state of the model has been saved you can run the model by navigating to the model directory in the command line and typing:

"python run_CBG_MU_Pool_Model_from_SS_Dual_Amplitude_Pulse_Duration_Modulation.py neuron"

Output files of the simulation are then written to a "Results" folder when the simulation is finished.  Model outputs are structured using the neo file format as detailed in https://neo.readthedocs.io/en/stable/.
