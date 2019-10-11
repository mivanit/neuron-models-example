import numpy as np

from numba import njit

from scipy import constants as spConst
from scipy.integrate import odeint

import sympy as sym
import sympy.physics.units as u

import matplotlib.pyplot as plt


 ######   #######  ##    ##  ######  ########
##    ## ##     ## ###   ## ##    ##    ##
##       ##     ## ####  ## ##          ##
##       ##     ## ## ## ##  ######     ##
##       ##     ## ##  ####       ##    ##
##    ## ##     ## ##   ### ##    ##    ##
 ######   #######  ##    ##  ######     ##


#* time constants
# NOTE: can be overriden in function calls
_t_min = 0.0
_t_max = 1500.0
_dt = 0.05



#* physical consts
# _R = spConst.R * u.joules / (u.mol * u.kelvin)
# _F = spConst.physical_constants['Faraday constant'][0] * u.coulombs / u.mol
_R = spConst.R
_F = spConst.physical_constants['Faraday constant'][0]

#* util
def C_to_Kelvin(temp, use_unit = False):
	if use_unit:
		return (temp + 273.15) * u.kelvin
	else:
		return (temp + 273.15)

def get_unit(expr):
	return expr.as_coeff_Mul()[1]

def get_val(expr):
	return expr.as_coeff_Mul()[0]


def check_unit(expr, unit):
	if expr.as_coeff_Mul()[1] == unit:
		# if dimensions equal, return expr
		return expr
	else:
		# if dimensions not equal, throw error
		raise ValueError('units should be \t %s, given %s' % ( str(unit), str(get_unit(expr)) ))


 ######  ########  #### ##    ## ########
##    ## ##     ##  ##  ##   ##  ##
##       ##     ##  ##  ##  ##   ##
 ######  ########   ##  #####    ######
      ## ##         ##  ##  ##   ##
##    ## ##         ##  ##   ##  ##
 ######  ##        #### ##    ## ########


@njit
def get_first_index_lb(A, k):
	''' idx of first element greater than k '''
	for i in range(len(A)):
		if A[i] > k:
			return i
	return -1

@njit
def get_first_index_ub(A, k):
	''' idx of first element less than k '''
	for i in range(len(A)):
		if A[i] < k:
			return i
	return -1

def get_spikes(arr_T, arr_Vm, thresh = 0.0, startTime = 0.0):
	'''
	until end of array reached,
	add every spike (time, amp) pair to the list
	'''

	lst_spikes = []
	idx = np.argmax(arr_T > startTime)
	N_pts = len(arr_T)

	while idx < N_pts:

		# find first pt above thresh
		idx_min = get_first_index_lb( arr_Vm[idx:], thresh ) + idx
		# break if no such point
		if idx_min == -1:
			break

		# find first pt below thresh
		idx_max = get_first_index_ub( arr_Vm[idx_min:], thresh ) + idx_min
		if idx_max <= idx_min:
			break

		# find max in that range
		idx_spike = arr_Vm[ idx_min:idx_max ].argmax()

		lst_spikes.append( ( arr_T[idx_spike], arr_Vm[idx_spike] ) )
		
		idx = idx_max + 1

	return lst_spikes





 ######  ######## #### ##     ##
##    ##    ##     ##  ###   ###
##          ##     ##  #### ####
 ######     ##     ##  ## ### ##
      ##    ##     ##  ##     ##
##    ##    ##     ##  ##     ##
 ######     ##    #### ##     ##


#* stim funcs
def stimFunc_constPulse(
		pulse_amp,
		pulse_start = 100,
		pulse_end = 1100,
	):
	def stim(t):
		if (t > pulse_start) and (t < pulse_end):
			return pulse_amp
		else:
			return 0.0

	return stim

def stimFunc_regPulse(
		pulse_amp,
		pulse_len = 1.0,
		pulse_delay = 10.0,
		pulse_count = 2,
		pulse_start = 100.0,
	):
	def stim(t):
		if t < pulse_start:
			return 0.0
		elif t > (pulse_start + pulse_count * pulse_delay + pulse_len):
			return 0.0
		else:
			# if we are less than `pulse_delay` past a pulse initation, return a pulse
			if ( (t - pulse_start) % pulse_delay ) < pulse_len:
				return pulse_amp
			else:
				return 0.0

	return stim


def stimFunc_pulseList(lst_pulses):
	'''
	pulse is a tuple (time, amp, length)
	NOTE: assumed to be sorted by time, and disjoint
	'''
	def stim(t):
		for p in lst_pulses:
			if t > p[0]:
				if t < p[0] + p[2]:
					return p[1]

		return 0.0

	return stim



 #######  ########        ##
##     ## ##     ##       ##
##     ## ##     ##       ##
##     ## ########        ##
##     ## ##     ## ##    ##
##     ## ##     ## ##    ##
 #######  ########   ######


#* obj
# base neural membrane model class
class NM_model(object):

       ######  ########  #######  ########
      ##    ##    ##    ##     ## ##     ##
      ##          ##    ##     ## ##     ##
      ##          ##    ##     ## ########
      ##          ##    ##     ## ##   ##
      ##    ##    ##    ##     ## ##    ##
       ######     ##     #######  ##     ##

	#* initializing a model

	def __init__(
			self,
			name_in,
			model_naming_in,
			model_expr_in,
			lst_vars_in,
			dict_syms,
			stim_sym_in,
			stim_func_in,
			dict_units = None,
			stabilization_period_in = 100.0,
			steady_in = None,
		):
		'''
		name 		:  name of the model
		lst_naming  :  name of each equation (left hand side, as a string)
		model_expr  :  model as list of sympy expressions
		lst_vars  	:  of the vars being solved for (sympy symbols)
		syms  		:  set of variables in the expression, mapped to their values (default to np.nan)
		units  		:  map symbols to units
		stim_sym 	:  symbol for applied current over time (to be substituted)
		stim_func 	:  function returning applied current over time
			NOTE: assumes only one stim variable
		sys_N		:  number of equations in the system
		'''
		self.name = name_in
		self.lst_naming = model_naming_in
		self.model_exprs = model_expr_in
		self.lst_vars = lst_vars_in
		self.syms = dict_syms
		self.units = dict_units
		self.stim_sym = stim_sym_in
		self.stim_func = stim_func_in
		
		self.sys_N = len(self.model_exprs)

		self.steady = steady_in

		self.stabilization_period = stabilization_period_in

		self.model_exprs_subs = None
		self.model_funcs = None





      ########     ###     ######  ########
      ##     ##   ## ##   ##    ## ##
      ##     ##  ##   ##  ##       ##
      ########  ##     ##  ######  ######
      ##     ## #########       ## ##
      ##     ## ##     ## ##    ## ##
      ########  ##     ##  ######  ########

	#* base functions, for substituting, lambdifying, and solving


	def subs_model(self):
		'''
		evaluate the model by substituting in `syms` values into `model_expr`
		(`LHS_expr` should be in reduced form)
		'''
		self.model_exprs_subs = [None] * self.sys_N
		for i in range(self.sys_N):
			self.model_exprs_subs[i] = self.model_exprs[i].subs( [ (sym, val) for sym, val in self.syms.items() ] )

	def get_funcs(self):
		'''
		lambdify all expressions
		for most models, `lst_vars` will be something like
		[ V_m, ...ion_currents...]
		onto which we add on [i_A]
		so the returned functions will take in voltage, ion currents, and current stimulus
		'''
		self.model_funcs = []
		for temp_expr in self.model_exprs_subs:
			temp_func = sym.lambdify(self.lst_vars + [ self.stim_sym ], temp_expr)
			self.model_funcs.append(temp_func)
		
	def solve(
			self,
			IC = None,
			T = np.arange( _t_min, _t_max, _dt ), 
		):
		'''
		solve the model for the given initial conditions and timesteps
		returns:
			( <timePoints>, <solutions>, <initial_conditions_used> )
		'''

		# set IC to steady state if one is found, otherwise use all 0's
		if IC is None:
			if self.steady is None:
				IC = np.array([0.0] * self.sys_N)
			else:
				IC = self.steady

		# make sure model is evaluated and lambdified
		if self.model_exprs_subs is None:
			self.subs_model()
		
		if self.model_funcs is None:
			self.get_funcs()

		# compute the derivatives for each of the functions
		def _compute_derivatives(_y, _t):
			_dy = np.full((len(IC),), np.nan, dtype = float)
			
			for idx in range(self.sys_N):
				# list of arguments to pass to the function
				# 	list of derivatives, applied current at this timepoint
				# 	plus I_A
				lst_args_temp = list(_y) + [ self.stim_func( _t ) ]

				# compute and store derivative
				_dy[idx] = self.model_funcs[idx]( *lst_args_temp )

			return _dy

		# Solve ODE system, store
		_Vy = odeint(_compute_derivatives, IC, T)
		
		# solution structure: ( time_arr, sln_arr, initial_conditions )
		self.sln = (T, _Vy, IC)

		return _Vy

	def find_stable(stabilization_period = 500.0, timeStep = 0.1):
		# no stim current
		self.stim_func = lambda t : 0.0
		# solve with zero initial conditions
		self.solve(
			T = np.arange(0.0, stabilization_period, timeStep),
			IC = np.array([0.0] * self.sys_N),
		)
		
		# get stable condition by looking at last timepoint
		self.steady = self.sln[1][-1]

		# wipe everything
		self.stim_func = None
		self.sln = None


      ########  ##        #######  ########
      ##     ## ##       ##     ##    ##
      ##     ## ##       ##     ##    ##
      ########  ##       ##     ##    ##
      ##        ##       ##     ##    ##
      ##        ##       ##     ##    ##
      ##        ########  #######     ##

	#* plotting various values

	def plot_AP(
			self, 
			plot_stim = True, 
			bln_show = True, 
			plot_title = None, 
			figure_tup = None, 
		):

		if plot_title is None:
			plot_title = self.name

		# Neuron potential
		if figure_tup is None:
			fig, ax = plt.subplots(figsize=(12, 7))
		else:
			fig, ax = figure_tup
		
		ax.grid()
		line, = ax.plot(self.sln[0], self.sln[1][:, 0])
		ax.set_xlabel('Time (ms)')
		ax.set_ylabel('Vm (mV)')
		ax.set_title(plot_title)

		# stimulus
		if plot_stim:
			ax2 = ax.twinx()
			Idv = [ self.stim_func(t) for t in self.sln[0] ]
			ax2.plot(self.sln[0], Idv, 'r-')
			ax2.set_ylabel(r'Stimulus Current density (uA/$cm^2$)')

		if bln_show:
			plt.show()

		return line


	def plot_fI(self, rampI_arr = None, switch_idx = None, rampI_seq = None, rampI_step = 0.1, freq_max = None, test_time = 1000, test_dt = 0.05):
		'''
		usage:
			`rampI_arr`   :  array of all the values to check
			`switch_idx`  :  (optional but useful) list of indecies at which ramp up/down switches
			`rampI_seq`   :  sequence of values to ramp up/down to, in order 
			`rampI_step`  :  (optional) step to use if `rampI_seq` is being used
			`freq_max`    :  (optional) frequency to start ramping back down at if reached (only if `rampI_seq` is being used)
			`test_time`   :  (optional) time to run for to test frequency
			`test_dt`     :  (optional) resolution at which to compute waveform
		'''

		# currents to try (order matters)
		if rampI_arr is None:
			# ramp every element
			list_rampI_temp = [ 
				np.arange(rampI_seq[idx-1], rampI_seq[idx]) for idx in range(1,len(rampI_seq)) 
			]
			# reverse every other element
			for x in range(len(list_rampI_temp)):
				if x % 2 == 0:
					list_rampI_temp[x] = list_rampI_temp[::-1]

			arr_rampI = np.concatenate(list_rampI_temp)

		# initial conditions, array to hold results
		IC = self.stable
		arr_freq = np.full(len(arr_rampI), np.nan, float)

		for idx in range(len(arr_rampI)):

			# for every current, measure freq
			self.stim_func = lambda t : arr_rampI[idx]
			self.solve(IC = IC, T = np.arange(0.0, test_time, test_dt))

			# get and store frequency
			lst_spikes = get_spikes( self.sln[0], self.sln[1][:,0] )
			arr_freq[idx] = len(lst_spikes)

			# save final state as IC for next step
			IC = self.sln[1][-1]

			# TODO: frequency maxing out stuff

			# print('stimulus =\t%.2f \t N_spikes = \t%d' % (arr_rampI[idx], arr_freq[idx]))
		
		# get switching indecies for plotting
		if (rampI_seq is not None) and (switch_idx is None):
			switch_idx = []
			for x in rampI_seq:
				switch_idx.append(len(x) + switch_idx[-1])
		
		# plot everything
		lines = []
		lines_lbl = []
		for idx in range(1, len(switch_idx)):
			# this (labels) is a bit useless unless you want way different colors 
			# (and corresponding labels)
			
			if idx % 2 == 1:
				fmt, lbl = 'ro-', 'ramp up'
			else:
				fmt, lbl = 'bo-', 'ramp down'

			line_temp, = plt.plot(
				arr_rampI[ switch_idx[idx-1] : switch_idx[idx] ], 
				arr_freq[ switch_idx[idx-1] : switch_idx[idx] ],
				fmt
			)

			lines.append(line_temp)
			lines_lbl.append(lbl)
		
		plt.xlabel(r'Stimulus Current density (uA/$cm^2$)')
		plt.ylabel('frequency (Hz)')
		plt.title('frequency-current relations for Hodgkin-Huxley')
		
		if len(lines) > 1:
			plt.legend(lines[:2], lines_lbl[:2])

		plt.show()


	def plot_refracPeriod(
			self, 
			pulseDims = (1.0, 10.0), 
			delayVals = np.arange(0.0, 10.0, 0.1),
		):
		'''
		show the timing and amplitude of the second spike
		as the timing of the second spike changes
		'''

		# stabilize initial 
		'''
		model_HH = HH.model_HH
		# no stim
		model_HH.stim_func = lambda t : 0.0
		# model_HH.stim = ( model_HH.stim[0], lambda t : 0.0 )
		# solve
		model_HH.solve(T = np.arange(0.0, 500.0, 0.01))
		# get stable condition by looking at last timepoint
		IC_stable = model_HH.sln[1][-1]

		pulse_amps = np.arange(3.0, 15.0, 3.0)

		pulse_delays = np.arange(0.0, 10.0, 0.5)

		pulse_length = 1.0

		arr_T = np.arange(100.0, 150.0, 0.01)

		fig1, ax1 = plt.subplots(figsize=(12, 7))
		fig2, ax2 = plt.subplots(figsize=(12, 7))

		ax1.set_title('pulse delay vs spike delay')
		ax1.set_xlabel('second pulse delay (ms)')
		ax1.set_ylabel('second spike peak delay (ms after pulse start)')

		ax2.set_title('pulse delay vs amplitude of second spike')
		ax2.set_xlabel('second pulse delay (ms)')
		ax2.set_ylabel('second spike peak voltage (mV)')

		lines_1 = []
		lines_2 = []
		
		# for each pulse amp
		for amp in pulse_amps:
			print('calculating for amp = %.2f' % amp)
			spikes = []

			# get spikes for the given amp and delay
			for dly in pulse_delays:
				# print('\tcalculating for dly = %.2f' % dly)
				
				stim_func = nm.stimFunc_regPulse(
					amp,
					pulse_len = pulse_length,
					pulse_delay = dly,
					pulse_count = 2,
					pulse_start = 100.0,
				)

				model_HH.stim_func = stim_func

				model_HH.solve(IC = IC_stable, T = arr_T)

				spikes_temp = nm.get_spikes( model_HH.sln[0], model_HH.sln[1][:,0] )

				print(spikes_temp)

				# if two spikes, add the second spike
				if len(spikes_temp) == 2:
					spikes.append( ( spikes_temp[1][0] - 100.0 - dly , spikes_temp[1][1]) )

				# if more than 2 spikes, something has gone wrong
				elif len(spikes_temp) > 2:
					print('ERROR: more than two spikes')
					print(spikes_temp)
					spikes.append((np.nan, np.nan))

				else:
					# if a single spike, then dont do anything
					spikes.append((np.nan, np.nan))

			print(spikes)

			# plot a line of stim delay vs spike delay
			line1, = ax1.plot( pulse_delays, [ x[0] for x in spikes ], 'o-' )
			lines_1.append(line1)

			# plot a line of stim delay vs second spike amp
			line2, = ax1.plot( pulse_delays, [ x[1] for x in spikes ], 'o-' )
			lines_2.append(line2)

		ax1.legend(lines_1, ['amp = %.2f' % x for x in pulse_amps])
		ax2.legend(lines_2, ['amp = %.2f' % x for x in pulse_amps])

		plt.show()
		'''


	def get_refracMinamp(
			self, 
			pulseLen = 2.0, 
			delayVals = np.arange(2.9, 6.0, 0.05)[::-1],
			maxAmp = 2000.0,
			maxIterations = 100,
			ampPrecision = 1.0,
			A_default = 10.0,
		):
		'''
		figure out the current needed to cause a spike at the given delayVals
		'''
		# minAmp = np.full(len(delayVals), np.nan, float)
		minAmp = []

		test_time_arr = np.arange(0.0, 150.0, 0.01)

		# for each time delay, find to within `ampPrecision`
		# the minimum input spike voltage required for a spike to happen
		for test_t in delayVals:
			print('testing for delay = \t%.2f' % test_t)
			
			# set bounds
			Abd_L = 0.0
			Abd_U = maxAmp
	
			# set to false when satisfied
			bln_loop = True
			count_loops = 0
			# loop until minAmp found
			while bln_loop and (count_loops < maxIterations):
				# get estimate
				A_test = (Abd_U + Abd_L) / 2

				# print('\t\t estimating amp = \t%.2f \t N_loops = \t%d' % (A_test, count_loops) )

				# create pulses
				test_pulses = [
					(100.0, A_default, pulseLen), 
					(100.0 + test_t, A_test, pulseLen)
				]
				
				# test if there is a second spike
				self.stim_func = stimFunc_pulseList(test_pulses)
				# self.stim = ( self.stim_sym, lambda t : 10.0 )

				self.solve(T = test_time_arr)

				spikes = get_spikes( self.sln[0], self.sln[1][:,0], startTime = 100.0 )

				# pair = find_range_max(compute(None, bln_plot=False, use_spikes = True, spikes=test_spike)[:,0], r_t=(max(3.5, test_t), test_t + 6), adj_dt = dt * 250.0 )

				if len(spikes) < 2:
					Abd_L = A_test
				else:
					Abd_U = A_test
				
				if (Abd_U - Abd_L) < ampPrecision:
					bln_loop = False
				
				count_loops = count_loops + 1
			
			minAmp.append(A_test)
			print('\t minAmp = \t%.2f' % A_test)

		# max_Vs = np.array(max_Vs) / 30.0
		# minAmp = [ (x - 5.5) for x in minAmp ]

		fig, ax1 = plt.subplots(figsize=(12, 7))
		plt.grid()
		ax1.plot(delayVals, minAmp, 'r.-')
			
		ax1.set_xlabel('second pulse delay (ms)')
		ax1.set_ylabel(r'current density required for second pulse (uA/$cm^2$)')

		plt.show()
		

	


