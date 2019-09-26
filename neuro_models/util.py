import numpy as np

from scipy import constants as spConst
from scipy.integrate import odeint

import sympy as sym
import sympy.physics.units as u

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


def eval_model(expr_in, var_symb, var_arr, dict_consts, model_units = None):
	'''
	args:
		- `expr_in` 	: sympy expression of eqn to be evaluated
		- `var_symb`	: symbol in `expr_in` that we are evaluating for
		- `var_arr` 	: array of floats to use as values for `var`
		- `dict_consts` : dict of (symbol, value) to be substituted
							into `expr_in`, note that value should have unit
		- `model_units`	: expected units of whole expr (key 'self') and all symbols (symbol as key)
	'''
	
	# substitute
	lst_subs = []
	for symb, val in dict_consts.items():
		# TODO: fix unit checking
		val_checked = val
		# val_checked = check_unit(val, model_units[symb])
		lst_subs.append((symb, val_checked))

	expr_model = expr_in.subs(lst_subs)

	# sym.pprint(expr_model)

	# check output units
	# TODO: fix unit checking
	# check_unit( expr_model.subs(var_symb, model_units[var_symb]), model_units['self'] )

	# return (expr_model, function)
	return (
		expr_model,
		sym.lambdify( var_symb, expr_model, 'numpy' )
	)
	# sym.lambdify( var_symb, expr_model / model_units[var_symb], 'numpy' )




# base neural membrane model class
class NM_model(object):

	def __init__(
			self, 
			model_naming_in,
			model_expr_in,
			vars_in,
			dict_syms,
			stim_in,
			dict_units = None,
		):
		'''
		lst_naming  :  name of each equation (left hand side, as a string)
		model_expr  :  model as list of sympy expressions
		lst_vars  	:  of the vars being solved for (sympy symbols)
		syms  		:  set of variables in the expression, mapped to their values (default to np.nan)
		units  		:  map symbols to units
		stim 		:  tuple (symbol, func) for applied current over time
		sys_N		:  number of equations in the system
		'''
		self.lst_naming = model_naming_in
		self.model_exprs = model_expr_in
		self.lst_vars = vars_in
		self.syms = dict_syms
		self.units = dict_units
		self.stim = stim

		self.sys_N = len(self.model_exprs)


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
			temp_func = sym.lambdify(self.lst_vars + stim[0], temp_expr)
			self.model_funcs.append(temp_func)
		
	def solve(
			self,
			IC = None,
			# timepoint list is easily overriden in two different ways 
			T = np.arange( _t_min, _t_max, _dt ), 
		):
		# default initial conditions are all zeroes
		if IC is None:
			IC = np.zeros((self.sys_N,), dtype = float)


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

		self.sln = (T, _Vy)

		return _Vy

	


