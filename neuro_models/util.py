import numpy as np

from scipy import constants as spConst
from scipy.integrate import odeint

import sympy as sym
import sympy.physics.units as u

import matplotlib.pyplot as plt

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




#* obj
# base neural membrane model class
class NM_model(object):

	def __init__(
			self,
			name_in,
			model_naming_in,
			model_expr_in,
			lst_vars_in,
			dict_syms,
			stim_in,
			dict_units = None,
			stabilization_period_in = 100.0,
		):
		'''
		name 		:  name of the model
		lst_naming  :  name of each equation (left hand side, as a string)
		model_expr  :  model as list of sympy expressions
		lst_vars  	:  of the vars being solved for (sympy symbols)
		syms  		:  set of variables in the expression, mapped to their values (default to np.nan)
		units  		:  map symbols to units
		stim 		:  tuple (symbol, func) for applied current over time
			NOTE: assumes only one stim variable
		sys_N		:  number of equations in the system
		'''
		self.name = name_in
		self.lst_naming = model_naming_in
		self.model_exprs = model_expr_in
		self.lst_vars = lst_vars_in
		self.syms = dict_syms
		self.units = dict_units
		self.stim = stim_in

		self.sys_N = len(self.model_exprs)

		self.stabilization_period = stabilization_period_in

	def subs_model(self):
		'''
		evaluate the model by substituting in `syms` values into `model_expr`
		(`LHS_expr` should be in reduced form)
		'''
		self.model_exprs_subs = self.model_exprs.subs( [ (sym, val) for sym, val in self.syms.iteritems() ] )

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
			temp_func = sym.lambdify(self.lst_vars + [ self.stim[0] ], temp_expr)
			self.model_funcs.append(temp_func)
		
	def solve(
			self,
			IC = None,
			T = np.arange( _t_min, _t_max, _dt ), 
		):

		if IC is None:
			IC = np.zeros((self.sys_N,), dtype = float)

		if self.model_exprs_subs is None:
			self.subs_model()
		
		if self.model_funcs is None:
			self.get_funcs()

		# compute the derivatives for each of the functions
		def _compute_derivatives(_y, _t):
			_dy = np.zeros((np.nan,), dtype = float)
			
			for idx in range(self.sys_N):
				# list of arguments to pass to the function
				# list of derivatives, applied current at this timepoint
				lst_args_temp = _y + self.stim[1]( _t )

				# compute and store derivative
				_dy[idx] = self.model_funcs[idx]( *lst_args_temp )

			return _dy

		# Solve ODE system, store
		_Vy = odeint(_compute_derivatives, IC, T)
		
		# solution structure: ( time_arr, sln_arr, initial_conditions )
		self.sln = (T, _Vy, IC)

		return _Vy


	def plot_AP(self, plot_stim = False, bln_show = True, plot_title = None):
		fig, ax = plt.subplots(figsize=(12, 7))

		if plot_title is None:
			plot_title = self.name

		# stimulus
		if plot_stim:
			ax2 = ax.twinx()
			Idv = [ self.stim(t) for t in self.sln[0] ]
			ax2.plot(self.sln[0], Idv, 'r-')
			ax2.set_ylabel(r'Stimulus Current density (uA/$cm^2$)')

		# Neuron potential
		fig, ax = plt.subplots(figsize=(12, 7))
		ax.plot(self.sln[0], self.sln[1][:, 0], 'b-')
		ax.set_xlabel('Time (ms)')
		ax.set_ylabel('Vm (mV)')
		ax.set_title(plot_title)
		plt.grid()

		if bln_show:
			plt.show()

	
	# def plot_fI(self):



	def plot_refracPeriod(
			self, 
			pulseDims = (1.0, 10.0), 
			delayVals = np.arange(0.0, 10.0, 0.1),
		):
		'''
		show the timing and amplitude of the second spike
		as the timing of the second spike changes
		'''


	def get_refracMinamp(
			self, 
			pulseLen = 1.0, 
			delayVals = np.arange(0.0, 10.0, 0.1),
			maxAmp = 10000.0,
			maxIterations = 100.0,
			ampPrecision = 0.01,
		):
		'''
		figure out the current needed to cause a spike at the given delayVals
		'''
		# minAmp = np.full(len(delayVals), np.nan, float)
		minAmp = []

		# for each time delay, find to within `ampPrecision`
		# the minimum input spike voltage required for a spike to happen
		for test_t in delayVals:
			
			# set bounds
			Abd_L = 0.0
			Abd_U = maxAmp
	
			# set to false when satisfied
			bln_loop = True
			# loop until minAmp found
			while bln_loop:
				# get estimate
				A_test = (Abd_U + Abd_L) / 2

				test_spike = [(0.0, A_default), (test_t, A_test)]
				
				pair = find_range_max(compute(None, bln_plot=False, use_spikes = True, spikes=test_spike)[:,0], r_t=(max(3.5, test_t), test_t + 6), adj_dt = dt * 250.0 )

				if pair[1] < spike_thresh:
					Abd_L = A_test
				else:
					Abd_U = A_test
				
				if (Abd_U - Abd_L) < thresh:
					bln_loop = False
			
			minAmp.append(A_test)

		# max_Vs = np.array(max_Vs) / 30.0
		minAmp = [ (x - 5.5) for x in minAmp ]

		fig, ax1 = plt.subplots(figsize=(12, 7))
		plt.grid()
		ax1.plot(delayVals, minAmp, 'r-')
			
		ax1.set_xlabel('second spike delay (ms)')
		ax1.set_ylabel('second spike additional min stimulation voltage', color='r')

		def func(t, a, b, c):
			return a * np.exp(b * t + c)

		popt, pcov = curve_fit(func, delayVals, minAmp)

		# xs = sym.Symbol('x')
		# tex = sym.latex(func(xs,*popt)).replace('$', '')
		# plt.title(r'$f(x)= %s$' %(tex),fontsize=16)
		plt.plot(delayVals, func(delayVals, *popt))
		plt.title("Fitted Curve a * exp(b * t + c) with a = %d, b = %d, c = %d" % (popt[0], popt[1], popt[2]))

		plt.show()
		

	


