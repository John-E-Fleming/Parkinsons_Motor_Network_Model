# -*- coding: utf-8 -*-
"""
Created on Wed April 03 14:27:26 2019

Description: Controller and scheduler class implementations 

@author: John Fleming, john.fleming@ucdconnect.ie
"""

import math

class Controller_Scheduler:
	"""Controller Scheduler Class"""

	def __init__(self, Controller_Bank=[], Scheduler_Bounds=[], MinParameterValue=0.0, MaxParameterValue=1e9, RampDuration=0.25, Ts=0.02, Scheduler_Label='Scheduler', Controller_Labels=['Tremor_ARV', 'Beta_ARV']):
		
		# Initial Scheduler Properties
		self.Controller_Bank = Controller_Bank			# A list of the controllers to be implemented by the scheduler
		self.Scheduler_Bounds = Scheduler_Bounds		# A list of the parameter bounds for the schduling variable corresponding to each controller 
		self.MinParameterValue = MinParameterValue		# Lower bound on the modulated parameter value
		self.MaxParameterValue = MaxParameterValue		# Upper bound on the modulated parameter value
		self.RampDuration = RampDuration	# should be defined in sec, i.e. 0.25 sec
		self.Ts = Ts						# should be in sec as per above
		self.Scheduler_Label = Scheduler_Label
		self.Controller_Labels = Controller_Labels
		
		# Parameter for tracking the controllers utilized from the controller bank
		self.Selected_Controller_id = 0
		self.Last_Selected_Controller_id = 0
		self.Transitioning_Controllers = False
		
		# Calculate how much controller output value will change each controller call
		self.maxParameterValueIncrement = (self.MaxParameterValue - self.MinParameterValue)/math.ceil(self.RampDuration/self.Ts)
		
		# Initialize the output value of the controller
		self.LastParameterOutputValue = 0
		
		# Lists for tracking scheduler history - Updated every scheduler update call
		self.scheduler_state_history = []
		self.scheduler_classification_history = []
		self.scheduler_output_history = []
		self.scheduler_sample_times = []
		
		# Lists for tracking controller bank history - Updated when controllers change or method called
		self.controller_state_history = [[] for _ in range(len(self.Controller_Bank))]
		self.controller_error_history = [[] for _ in range(len(self.Controller_Bank))]
		self.controller_output_history = [[] for _ in range(len(self.Controller_Bank))]
		self.controller_sample_times = [[] for _ in range(len(self.Controller_Bank))]


	def clear(self):
		"""Clears scheduler output values and history"""

		# Reset the selected controller index being used 
		self.Selected_Controller_id = 0
		self.Last_Selected_Controller_id = 0
		self.Transitioning_Controllers = True 	# Setting transition to True will force the scheduler to switch off DBS before restarting
		
		# Clear the output value of the scheduler
		self.LastParameterOutputValue = 0
		
		# Clear the scheduler history lists
		self.scheduler_state_history = []
		self.scheduler_classification_history = []
		self.scheduler_output_history = []
		self.scheduler_sample_times = []
		
		# Clear the controller bank history lists
		self.controller_state_history = [[] for _ in range(len(self.Controller_Bank))]
		self.controller_error_history = [[] for _ in range(len(self.Controller_Bank))]
		self.controller_output_history = [[] for _ in range(len(self.Controller_Bank))]
		self.controller_sample_times = [[] for _ in range(len(self.Controller_Bank))]

	def update(self, scheduler_state_value, controller_state_values, current_time):
		"""Determines which bounds the scheduling variable is in to implement the appropriate controller
			
		"""
		
		# 1) Check if scheduler is transitioning between controllers
		if self.Transitioning_Controllers == True:

			# Scheduler is transitioning between controllers - 
			# Ramp down the output parameter at the max rate
			ParameterOutputValue = self.LastParameterOutputValue - self.maxParameterValueIncrement 
			
			# Check if the parameter value is <= 0, if so transition is complete
			if ParameterOutputValue <= 0.0:
				ParameterOutputValue = 0	 	 	# Bound the output value at zero
				self.Transitioning_Controllers = False 	# Set the transition flag to off
		else:
			# 2) Scheduler is not transitioning between controllers -   
			# 2.1) Check which bounds the scheduling variable is in 
			for bound_id, bound_values in enumerate(self.Scheduler_Bounds):
				# Select the corresponding controller based on the scheduler parameter bounds
				if scheduler_state_value >= bound_values[0] and scheduler_state_value < bound_values[1]:
					self.Selected_Controller_id = bound_id
					
				else:
					pass
			
			# Above code assumes that incorrect values are not entered to the scheduler, i.e. a value outside the defined scheduler bounds
			
			# 2.2) Check if the selected controller has changed from the last scheduler call
			if self.Selected_Controller_id == self.Last_Selected_Controller_id:
				
				# 2.2.1) Controller has not changed since last call, update the output using this controller
				ParameterOutputValue = self.Controller_Bank[self.Selected_Controller_id].update(state_value=controller_state_values[self.Selected_Controller_id], current_time=current_time)
				
				# Update the last selected controller property
				self.Last_Selected_Controller_id = self.Selected_Controller_id
				self.Transitioning_Controllers = False
				
			else:
			
				# 2.2.2) Controller has changed since the last call
				#	- Get the previously used controller data
				self.controller_state_history[self.Last_Selected_Controller_id].append(self.Controller_Bank[self.Last_Selected_Controller_id].get_state_history())
				self.controller_error_history[self.Last_Selected_Controller_id].append(self.Controller_Bank[self.Last_Selected_Controller_id].get_error_history())
				self.controller_output_history[self.Last_Selected_Controller_id].append(self.Controller_Bank[self.Last_Selected_Controller_id].get_output_history())
				self.controller_sample_times[self.Last_Selected_Controller_id].append(self.Controller_Bank[self.Last_Selected_Controller_id].get_sample_times())
				
				# And reset the controller for use later on initial change into transition
				if self.Transitioning_Controllers == False:
					# 	- Reset the controller for use later
					self.Controller_Bank[self.Last_Selected_Controller_id].clear()
					# 	- Set the scheduler transition property to True
					self.Transitioning_Controllers = True
				else:
					pass
					
				# 	- Ramp down the output parameter at the max rate
				ParameterOutputValue = self.LastParameterOutputValue - self.maxParameterValueIncrement 
				
				# Check if the parameter value is <= 0, if so transition is complete
				if ParameterOutputValue <= 0.0:
					ParameterOutputValue = 0	 	 	# Bound the output value at zero
					self.Transitioning_Controllers = False 	# Set the transition flag to off
					
				else:
					pass
				
				self.Last_Selected_Controller_id = self.Selected_Controller_id
			
		# Update the LastParameterOutputValue property to the current output
		self.LastParameterOutputValue = ParameterOutputValue
		
		# Record scheduler state, classification, output, and sample time values
		self.scheduler_state_history.append(scheduler_state_value)
		self.scheduler_classification_history.append(self.Selected_Controller_id)
		self.scheduler_output_history.append(ParameterOutputValue)
		self.scheduler_sample_times.append(current_time/1000.0)
		
		# Return parameter output value
		return ParameterOutputValue
	
	def stop(self):
		"""Method stops the scheduler and writes current controller data to respective class properties"""
		self.controller_state_history[self.Last_Selected_Controller_id].append(self.Controller_Bank[self.Last_Selected_Controller_id].get_state_history())
		self.controller_error_history[self.Last_Selected_Controller_id].append(self.Controller_Bank[self.Last_Selected_Controller_id].get_error_history())
		self.controller_output_history[self.Last_Selected_Controller_id].append(self.Controller_Bank[self.Last_Selected_Controller_id].get_output_history())
		self.controller_sample_times[self.Last_Selected_Controller_id].append(self.Controller_Bank[self.Last_Selected_Controller_id].get_sample_times())
		self.Controller_Bank[self.Last_Selected_Controller_id].clear()
		
	def set_MaxParameterValue(self, MaxParameterValue):
		"""Sets the upper bound for the controller output"""
		self.MaxParameterValue = MaxParameterValue
		self.maxParameterValueIncrement = (self.MaxParameterValue - self.MaxParameterValue)/math.ceil(self.RampDuration/self.Ts)
	
	def set_MinParameterValue(self, MinParameterValue):
		"""Sets the lower bound for the controller output"""
		self.MinParameterValue = MinParameterValue
		self.maxParameterValueIncrement = (self.MaxParameterValue - self.MaxParameterValue)/(self.RampDuration/self.Ts)
	
	def set_RampDuration(self, ramp_duration):
		"""Sets the how long the controller output takes to reach it's max value"""
		self.RampDuration = ramp_duration
		self.maxParameterValueIncrement = (self.MaxParameterValue - self.MaxParameterValue)/(self.RampDuration/self.Ts)
	
	def set_Ts(self, Ts):
		"""Sets the sampling rate of the controller"""
		self.Ts = Ts
		self.maxParameterValueIncrement = (self.MaxParameterValue - self.MaxParameterValue)/(self.RampDuration/self.Ts)
	
	def set_Scheduler_Label(self, label):
		"""Sets the Scheduler Label"""
		self.Scheduler_Label = label
	
	def set_Controller_Labels(self, label):
		"""Sets the Controller Labels"""
		self.Controller_Labels = label
		
	def get_scheduler_state_history(self):
		return self.scheduler_state_history
		
	def get_scheduler_classification_history(self):
		return self.scheduler_classification_history
	
	def get_scheduler_output_history(self):
		return self.scheduler_output_history
	
	def get_scheduler_sample_times(self):
		return self.scheduler_sample_times
		
	def get_controller_state_history(self):
		return self.controller_state_history
		
	def get_controller_error_history(self):
		return self.controller_error_history
	
	def get_controller_output_history(self):
		return self.controller_output_history
	
	def get_controller_sample_times(self):
		return self.controller_sample_times

	def get_Scheduler_Label(self):
		return self.Scheduler_Label
	
	def get_Controller_Labels(self):
		return self.Controller_Labels

class State_Observer:
	"""State Observer Class"""

	def __init__(self, Ts=0.0):
		# Initial Controller Values
		self.Ts = Ts						# should be in sec as per above
		
		# Lists for tracking controller history
		self.state_history = []
		self.sample_times = []
		
		#self.clear()

	def clear(self):
		"""Clears current On-Off controller output value and history"""
		self.state_history = []
		self.sample_times = []
		
	def update(self, state_value, current_time):
		# Records current state measurement
		self.state_history.append(state_value)
		self.sample_times.append(current_time/1000)			# Convert from msec to sec

	def setTs(self, Ts):
		"""Sets the sampling rate of the controller"""
		self.Ts = Ts
			
	def get_state_history(self):
		return self.state_history
	
	def get_sample_times(self):
		return self.sample_times

class Constant_Controller:
	"""Constant DBS Parameter Controller Class"""

	def __init__(self, SetPoint=0.0, MinValue=0.0, MaxValue=1e9, ConstantValue=0.0, Ts=0.0, units='mA'):
		# Initial Controller Values
		self.SetPoint = SetPoint
		self.MaxValue = MaxValue
		self.MinValue = MinValue
		self.ConstantValue = ConstantValue
		self.Ts = Ts						# should be in sec as per above
		self.units = units
		self.label = ('Constant_Controller/%f%s' % (self.ConstantValue, self.units))
		
		# Set output value
		self.OutputValue = 0
		
		# Lists for tracking controller history
		self.state_history = []
		self.error_history = []
		self.output_history = []
		self.sample_times = []

	def clear(self):
		"""Clears current On-Off controller output value and history"""

		self.state_history = []
		self.error_history = []
		self.output_history = []
		self.sample_times = []
		
		self.OutputValue = 0.0

	def update(self, state_value, current_time):
		"""Calculates biomarker for constant DBS value
			
			u = self.ConstantValue
			
		"""
		
		# Calculate Error - if SetPoint > 0.0, then normalize error with respect to setpoint
		if self.SetPoint==0.0:
			error = state_value - self.SetPoint
		else:
			error = (state_value - self.SetPoint)/self.SetPoint 
		
		# Bound the controller output (between MinValue - MaxValue)
		if self.ConstantValue > self.MaxValue:
			self.OutputValue = self.MaxValue
		elif self.ConstantValue < self.MinValue:
			self.OutputValue = self.MinValue
		else:
			self.OutputValue = self.ConstantValue
		
		# Record state, error and sample time values
		self.state_history.append(state_value)
		self.error_history.append(error)
		self.output_history.append(self.OutputValue)
		self.sample_times.append(current_time/1000)			# Convert from msec to sec
		
		return self.OutputValue
		
	def setMaxValue(self, max_value):
		"""Sets the upper bound for the controller output"""
		self.MaxValue = max_value

	def setMinValue(self, min_value):
		"""Sets the lower bound for the controller output"""
		self.MinValue = min_value
	
	def setConstantValue(self, constant_value):
		"""Sets the constant controller output"""
		self.ConstantValue = ConstantValue
	
	def setTs(self, Ts):
		"""Sets the sampling rate of the controller"""
		self.Ts = Ts
	
	def setLabel(self, label):
		"""Sets the label of the controller"""
		self.label = label
	
	def setSetPoint(self, set_point):
		self.SetPoint = set_point
		
	def get_state_history(self):
		return self.state_history
		
	def get_error_history(self):
		return self.error_history
	
	def get_output_history(self):
		return self.output_history
	
	def get_sample_times(self):
		return self.sample_times

	def get_label(self):
		return self.label

class ON_OFF_Controller:
	"""On-Off Controller Class"""

	def __init__(self, SetPoint=0.0, MinValue=0.0, MaxValue=1e9, RampDuration=0.25, Ts=0.02):
		# Initial Controller Values
		self.SetPoint = SetPoint
		self.MaxValue = MaxValue
		self.MinValue = MinValue
		self.RampDuration = RampDuration	# should be defined in sec, i.e. 0.25 sec
		self.Ts = Ts						# should be in sec as per above
		self.label = 'On_Off_Controller'
		
		# Calculate how much controller output value will change each controller call
		self.OutputValueIncrement = (self.MaxValue - self.MinValue)/math.ceil(self.RampDuration/self.Ts)
		
		# Initialize the output value of the controller
		self.LastOutputValue = MinValue
		self.OutputValue = MinValue
		
		# Lists for tracking controller history
		self.state_history = []
		self.error_history = []
		self.output_history = []
		self.sample_times = []

	def clear(self):
		"""Clears current On-Off controller output value and history"""

		self.state_history = []
		self.error_history = []
		self.output_history = []
		self.sample_times = []
		
		self.LastOutputValue = 0.0
		self.OutputValue = 0.0

	def update(self, state_value, current_time):
		"""Calculates updated controller output value for given reference feedback
			
			y(t) = y(t-1) + u(t)
		
			where:
		
			u(t) = MaxValue / (RampDuration/Ts) if e(t) > SetPoint or -MaxValue / (RampDuration/Ts) if e(t) < SetPoint

		"""
		
		# Calculate Error - if SetPoint > 0.0, then normalize error with respect to setpoint
		if self.SetPoint==0.0:
			error = state_value - self.SetPoint
			increment = 0.0
		else:
			error = (state_value - self.SetPoint)/self.SetPoint 
			if error > 0.0:
				increment = self.OutputValueIncrement
			else:
				increment = -self.OutputValueIncrement
		
		# Bound the controller output (between MinValue - MaxValue)
		if self.LastOutputValue+increment > self.MaxValue:
			self.OutputValue = self.MaxValue
		elif self.LastOutputValue+increment < self.MinValue:
			self.OutputValue = self.MinValue
		else:
			self.OutputValue = self.LastOutputValue+increment
		
		# Record state, error and sample time values
		self.state_history.append(state_value)
		self.error_history.append(error)
		self.output_history.append(self.OutputValue)
		self.sample_times.append(current_time/1000)			# Convert from msec to sec
		
		self.LastOutputValue = self.OutputValue
		
		return self.OutputValue
		
	def setMaxValue(self, max_value):
		"""Sets the upper bound for the controller output"""
		self.MaxValue = max_value
		self.OutputValueIncrement = (self.MaxValue - self.MinValue)/(self.RampDuration/self.Ts)

	def setMinValue(self, min_value):
		"""Sets the lower bound for the controller output"""
		self.MinValue = min_value
		self.OutputValueIncrement = (self.MaxValue - self.MinValue)/(self.RampDuration/self.Ts)
	
	def setRampDuration(self, ramp_duration):
		"""Sets the how long the controller output takes to reach it's max value"""
		self.RampDuration = ramp_duration
		self.OutputValueIncrement = (self.MaxValue - self.MinValue)/(self.RampDuration/self.Ts)
	
	def setTs(self, Ts):
		"""Sets the sampling rate of the controller"""
		self.Ts = Ts
		self.OutputValueIncrement = (self.MaxValue - self.MinValue)/(self.RampDuration/self.Ts)
	
	def setLabel(self, label):
		"""Sets the label of the controller"""
		self.label = label
	
	def setSetPoint(self, set_point):
		self.SetPoint = set_point
		
	def get_state_history(self):
		return self.state_history
		
	def get_error_history(self):
		return self.error_history
	
	def get_output_history(self):
		return self.output_history
	
	def get_sample_times(self):
		return self.sample_times

	def get_label(self):
		return self.label

class Dual_Threshold_Controller:
	"""Dual-Threshold Controller Class"""

	def __init__(self, LowerThreshold=0.0, UpperThreshold=0.1, MinValue=0.0, MaxValue=1e9, RampDuration=0.25, Ts=0.02):
		# Initial Controller Values
		self.UpperThreshold = UpperThreshold
		self.LowerThreshold = LowerThreshold
		self.MaxValue = MaxValue
		self.MinValue = MinValue
		self.RampDuration = RampDuration		# should be defined in sec, i.e. 0.25 sec
		self.Ts = Ts							# should be in sec as per above
		self.label = 'Dual_Threshold_Controller'
		
		# Calculate how much controller output value will change each controller call
		self.OutputValueIncrement = (self.MaxValue - self.MinValue)/math.ceil(self.RampDuration/self.Ts)
		
		# Initialize the output value of the controller
		self.LastOutputValue = MinValue
		self.OutputValue = MinValue
		
		# Lists for tracking controller history
		self.state_history = []
		self.error_history = []
		self.output_history = []
		self.sample_times = []

	def clear(self):
		"""Clears current dual-threshold controller output value and history"""

		self.state_history = []
		self.error_history = []
		self.output_history = []
		self.sample_times = []
		
		self.LastOutputValue = 0.0
		self.OutputValue = 0.0

	def update(self, state_value, current_time):
		"""Calculates updated controller output value for given reference feedback
			
			if state_value > upper_threshold:
				y(t) = y(t-1) + u(t)
			elif state_value < lower_threshold:
				y(t) = y(t-1) + u(t)
			else:
				y(t) = y(t-1)
			where:
		
			u(t) = MaxValue / (RampDuration/Ts) if state_value(t) > UpperThreshold or -MaxValue / (RampDuration/Ts) if state_value(t) < LowerThreshold

		"""
		
		# Check how to update controller value and calculate error with respect to upper/lower threshold
		if state_value > self.UpperThreshold:			# Increase if above upper threshold
			error = (state_value - self.UpperThreshold)/self.UpperThreshold
			increment = self.OutputValueIncrement
		elif state_value < self.LowerThreshold:			# Decrease if below lower threshold
			error = (state_value - self.LowerThreshold)/self.LowerThreshold
			increment = -self.OutputValueIncrement
		else:											# Do nothing when within upper and lower thresholds
			error = 0
			increment = 0								
		
		# Bound the controller output (between MinValue - MaxValue)
		if self.LastOutputValue+increment > self.MaxValue:
			self.OutputValue = self.MaxValue
		elif self.LastOutputValue+increment < self.MinValue:
			self.OutputValue = self.MinValue
		else:
			self.OutputValue = self.LastOutputValue+increment
			
		# Record state, error and sample time values
		self.state_history.append(state_value)
		self.error_history.append(error)
		self.output_history.append(self.OutputValue)
		self.sample_times.append(current_time/1000)		# Convert from msec to sec
		
		self.LastOutputValue = self.OutputValue
		
		return self.OutputValue

	def setUpperThreshold(self, upper_threshold):
		"""Sets the upper threshold for the measured state"""
		self.UpperThreshold = upper_threshold
	
	def setLowerThreshold(self, lower_threshold):
		"""Sets the lower threshold for the measured state"""
		self.LowerThreshold = lower_threshold
	
	def setMaxValue(self, max_value):
		"""Sets the upper bound for the controller output"""
		self.MaxValue = max_value
		self.OutputValueIncrement = (self.MaxValue - self.MinValue)/(self.RampDuration/self.Ts)

	def setMinValue(self, min_value):
		"""Sets the lower bound for the controller output"""
		self.MinValue = min_value
		self.OutputValueIncrement = (self.MaxValue - self.MinValue)/(self.RampDuration/self.Ts)
		
	def setRampDuration(self, ramp_duration):
		"""Sets the how long the controller output takes to reach it's max value"""
		self.RampDuration = ramp_duration
		self.OutputValueIncrement = (self.MaxValue - self.MinValue)/(self.RampDuration/self.Ts)
	
	def setTs(self, Ts):
		"""Sets the sampling rate of the controller"""
		self.Ts = Ts
		self.OutputValueIncrement = (self.MaxValue - self.MinValue)/(self.RampDuration/self.Ts)
	
	def setLabel(self, label):
		"""Sets the label of the controller"""
		self.label = label
	
	def setSetPoint(self, set_point):
		self.SetPoint = set_point
		
	def get_state_history(self):
		return self.state_history
	
	def get_error_history(self):
		return self.error_history
	
	def get_output_history(self):
		return self.output_history
	
	def get_sample_times(self):
		return self.sample_times
		
	def get_label(self):
		return self.label

class standard_PID_Controller:
	"""Standard PID Controller Class"""

	def __init__(self, SetPoint=0.0, Kp=0.0, Ti=0.0, Td=0.0, Ts=0.02, MinValue=0.0, MaxValue=1e9):

		self.SetPoint = SetPoint
		self.Kp = Kp
		self.Ti = Ti
		self.Td = Td
		
		# Set output value bounds
		self.MinValue = MinValue
		self.MaxValue = MaxValue
		
		self.label = "standard_PID_Controller/Kp=%f, Ti=%f, Td=%f" % (self.Kp, self.Ti, self.Td)
		
		self.Ts = Ts
		self.current_time = 0.0 # (sec)
		self.last_time = 0.0	
		
		# Initialize controller terms
		self.ITerm = 0.0
		self.DTerm = 0.0
		self.last_error = 0.0
		self.last_OutputValue = 0.0
		
		# Initialize the output value of the controller
		self.OutputValue = 0.0
		
		self.state_history = []
		self.error_history = []
		self.output_history = []
		self.sample_times = []

	def clear(self):
		"""Clears PID computations and coefficients"""

		self.ITerm = 0.0
		self.DTerm = 0.0
		self.last_error = 0.0

		self.state_history = []
		self.error_history = []
		self.output_history = []
		self.sample_times = []
		
		self.OutputValue = 0.0

	def update(self, state_value, current_time):
		"""Calculates controller output signal for given reference feedback
		
			where:
		
			u(t) = K_p (e(t) + (1/T_i)* \int_{0}^{t} e(t)dt + T_d {de}/{dt})		
			
			where the error calculated is the tracking error (r(t) - y(t))

		"""

		# Calculate Error - if SetPoint > 0.0, then normalize error with respect to setpoint
		if self.SetPoint==0.0:
			error = state_value - self.SetPoint
		else:
			error = (state_value - self.SetPoint)/self.SetPoint 
		
		self.current_time = current_time/1000.0 		# Converting from msec to sec
		delta_time = self.Ts
		delta_error = error - self.last_error

		self.ITerm += error * delta_time
		
		self.DTerm = 0.0
		if delta_time > 0:
			self.DTerm = delta_error / delta_time

		# Remember last time and last error for next calculation
		self.last_time = self.current_time
		self.last_error = error
		
		# Calculate u(t) - catch potential division by zero error
		try:
			u = self.Kp * (error + ((1.0/self.Ti) * self.ITerm) + (self.Td * self.DTerm))
		except ZeroDivisionError:
			u = self.Kp * (error + (0.0 * self.ITerm) + (self.Td * self.DTerm))
				
		# Bound the controller output if necessary (between MinValue - MaxValue) 
		if u > self.MaxValue:
			self.OutputValue = self.MaxValue
			self.ITerm -= error * delta_time 	# Back-calculate the integral error
		elif u < self.MinValue:
			self.OutputValue = self.MinValue
			self.ITerm -= error * delta_time 	# Back-calculate the integral error
		else:
			self.OutputValue = u
		
		# Update the last output value
		self.last_OutputValue = self.OutputValue
		
		# Record state, error, y(t), and sample time values
		self.state_history.append(state_value)
		self.error_history.append(error)
		self.output_history.append(self.OutputValue)
		self.sample_times.append(current_time/1000)		# Convert from msec to sec
		
		# Return controller output
		return self.OutputValue
		
	def setKp(self, proportional_gain):
		"""Determine how aggressively the controller reacts to the current error with setting Proportional Gain"""
		self.Kp = proportional_gain
		self.label = "standard_PID_Controller/Kp=%f, Ti=%f, Td=%f" % (self.Kp, self.Ti, self.Td)

	def setTi(self, Ti):
		"""Determine how fast the controller integrates the error history by setting Integral Time Constant"""
		self.Ti = Ti
		self.label = "standard_PID_Controller/Kp=%f, Ti=%f, Td=%f" % (self.Kp, self.Ti, self.Td)

	def setTd(self, Td):
		"""Determine far into the future the controller predicts future errors setting Derivative Time Constant"""
		self.Td = Td
		self.label = "standard_PID_Controller/Kp=%f, Ti=%f, Td=%f" % (self.Kp, self.Ti, self.Td)

	def setSetPoint(self, set_point):
		"""Set target setpoint value"""
		self.SetPoint = set_point

	def setMaxValue(self, max_value):
		"""Sets the upper bound for the controller output"""
		self.MaxValue = max_value
		
	def setMinValue(self, min_value):
		"""Sets the lower bound for the controller output"""
		self.MinValue = min_value
		
	def get_state_history(self):
		return self.state_history
		
	def get_error_history(self):
		return self.error_history
		
	def get_output_history(self):
		return self.output_history
		
	def get_sample_times(self):
		return self.sample_times
		
	def get_label(self):
		return self.label
		

class standard_PID_Controller_Plus_Bias:
	"""Standard PID Controller plus Constant Output Bias Class"""

	def __init__(self, SetPoint=0.0, Kp=0.0, Ti=0.0, Td=0.0, Bias=0.0, Ts=0.02, MinValue=0.0, MaxValue=1e9):

		self.SetPoint = SetPoint
		self.Kp = Kp
		self.Ti = Ti
		self.Td = Td
		self.Bias = Bias
		
		# Set output value bounds
		self.MinValue = MinValue
		self.MaxValue = MaxValue
		
		self.label = "standard_PID_Controller_Plus_Bias/Kp=%f, Ti=%f, Td=%f, Bias=%f" % (self.Kp, self.Ti, self.Td, self.Bias)
		
		self.Ts = Ts
		self.current_time = 0.0 # (sec)
		self.last_time = 0.0	
		
		# Initialize controller terms
		self.ITerm = 0.0
		self.DTerm = 0.0
		self.last_error = 0.0
		self.last_OutputValue = 0.0
		
		# Initialize the output value of the controller
		self.OutputValue = 0.0
		
		self.state_history = []
		self.error_history = []
		self.output_history = []
		self.sample_times = []

	def clear(self):
		"""Clears PID computations and coefficients"""

		self.ITerm = 0.0
		self.DTerm = 0.0
		self.last_error = 0.0

		self.state_history = []
		self.error_history = []
		self.output_history = []
		self.sample_times = []
		
		self.OutputValue = 0.0

	def update(self, state_value, current_time):
		"""Calculates controller output signal for given reference feedback
		
			where:
		
			u(t) = K_p (e(t) + (1/T_i)* \int_{0}^{t} e(t)dt + T_d {de}/{dt})		
			
			where the error calculated is the tracking error (r(t) - y(t))

		"""

		# Calculate Error - if SetPoint > 0.0, then normalize error with respect to setpoint
		if self.SetPoint==0.0:
			error = state_value - self.SetPoint
		else:
			error = (state_value - self.SetPoint)/self.SetPoint 
		
		self.current_time = current_time/1000.0 		# Converting from msec to sec
		delta_time = self.Ts
		delta_error = error - self.last_error

		self.ITerm += error * delta_time
		
		self.DTerm = 0.0
		if delta_time > 0:
			self.DTerm = delta_error / delta_time

		# Remember last time and last error for next calculation
		self.last_time = self.current_time
		self.last_error = error
		
		# Calculate u(t) - catch potential division by zero error
		try:
			u = self.Bias + self.Kp * (error + ((1.0/self.Ti) * self.ITerm) + (self.Td * self.DTerm))
		except ZeroDivisionError:
			u = self.Bias + self.Kp * (error + (0.0 * self.ITerm) + (self.Td * self.DTerm))
				
		# Bound the controller output if necessary (between MinValue - MaxValue) 
		if u > self.MaxValue:
			self.OutputValue = self.MaxValue
			self.ITerm -= error * delta_time 	# Back-calculate the integral error
		elif u < self.MinValue:
			self.OutputValue = self.MinValue
			self.ITerm -= error * delta_time 	# Back-calculate the integral error
		else:
			self.OutputValue = u
		
		# Update the last output value
		self.last_OutputValue = self.OutputValue
		
		# Record state, error, y(t), and sample time values
		self.state_history.append(state_value)
		self.error_history.append(error)
		self.output_history.append(self.OutputValue)
		self.sample_times.append(current_time/1000)		# Convert from msec to sec
		
		# Return controller output
		return self.OutputValue
		
	def setKp(self, proportional_gain):
		"""Determine how aggressively the controller reacts to the current error with setting Proportional Gain"""
		self.Kp = proportional_gain
		self.label = "standard_PID_Controller_Plus_Bias/Kp=%f, Ti=%f, Td=%f, Bias=%f" % (self.Kp, self.Ti, self.Td, self.Bias)
	
	def setTi(self, Ti):
		"""Determine how fast the controller integrates the error history by setting Integral Time Constant"""
		self.Ti = Ti
		self.label = "standard_PID_Controller_Plus_Bias/Kp=%f, Ti=%f, Td=%f, Bias=%f" % (self.Kp, self.Ti, self.Td, self.Bias)

	def setTd(self, Td):
		"""Determine far into the future the controller predicts future errors setting Derivative Time Constant"""
		self.Td = Td
		self.label = "standard_PID_Controller_Plus_Bias/Kp=%f, Ti=%f, Td=%f, Bias=%f" % (self.Kp, self.Ti, self.Td, self.Bias)
		
	def setBias(self, Bias):
		"""Set the controller output bias"""
		self.Bias = Bias
		self.label = "standard_PID_Controller_Plus_Bias/Kp=%f, Ti=%f, Td=%f, Bias=%f" % (self.Kp, self.Ti, self.Td, self.Bias)
	
	def setSetPoint(self, set_point):
		"""Set target setpoint value"""
		self.SetPoint = set_point
	
	def setMaxValue(self, max_value):
		"""Sets the upper bound for the controller output"""
		self.MaxValue = max_value
		
	def setMinValue(self, min_value):
		"""Sets the lower bound for the controller output"""
		self.MinValue = min_value
		
	def get_state_history(self):
		return self.state_history
		
	def get_error_history(self):
		return self.error_history
		
	def get_output_history(self):
		return self.output_history
		
	def get_sample_times(self):
		return self.sample_times
		
	def get_label(self):
		return self.label

