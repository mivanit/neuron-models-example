'''
Wang-Buzsaki neuron model
'''
#%%

import numpy as np
import sympy as sym
from scipy.integrate import odeint
from copy import deepcopy

from neuro_models.neuroUtil import *


# Average potassium, sodium, leak channel conductance per unit area (mS/cm^2)
_g_K, _g_Na, _g_L = sym.symbols('g_K g_Na g_L')
# Average potassium, sodium, leak potentials (mV)
_E_K, _E_Na, _E_L = sym.symbols('E_K E_Na E_L')
# capacitance of membrane, applied current
_C_m, _I_A = sym.symbols('C_m I_A')
# membrane voltage, potassium gating var, leak gating var
_V_m, _n, _h = sym.symbols('V_m n h')


# rate funcs
# _alpha_n, _beta_n, _alpha_n, _beta_n, _alpha_n, _beta_n = sym.symbols('alpha_n beta_n alpha_n beta_n alpha_n beta_n')
# steady states
# _n_inf, _m_inf, _h_inf = sym.symbols('n_inf m_inf h_inf')

WB_consts = {
	_C_m : 1.0,
	_E_Na : 55.0,
	_E_K : -90.0,
	_E_L : -65.0,
	_g_Na : 35.0,
	_g_K : 9.0,
	_g_L : 0.1,
}

# Sodium ion-channel rate functions
_alpha_m = 0.1 * ( 35.0 + _V_m ) / ( 1 - sym.exp( -1 * ( _V_m + 35.0 ) / 10.0 ))
_beta_m = 4.0 * sym.exp( - ( _V_m + 60.0 ) / 18.0 )

# leak channel rate values
_alpha_h = 0.35 * sym.exp( - ( _V_m + 58.0 ) / 20 )
_beta_h = 5.0 / ( 1 + sym.exp( -0.1 * ( _V_m + 28.0 ) ))

# Potassium ion-channel rate functions
_alpha_n = 0.05 * ( 34.0 + _V_m ) / ( 1 - sym.exp( -0.1 * ( _V_m + 34.0 )) )
_beta_n = 0.625 * sym.exp( - ( _V_m + 44.0 ) / 80.0 )

def get_model():
	# Wang-Buzsaki model expressions

	# n, m, and h steady-state values
	_n_inf = _alpha_n / ( _alpha_n + _beta_n )
	_m_inf = _alpha_m / ( _alpha_m + _beta_m )
	_h_inf = _alpha_h / ( _alpha_h + _beta_h )

	# currents
	_I_K = _g_K * (_n ** 4.0) * ( _V_m - _E_K )
	_I_Na = _g_Na * ( _m_inf ** 3.0 ) * _h * (_V_m - _E_Na)
	_I_L = _g_L * (_V_m - _E_L)

	# diffeqs

	WB_dv_dt = ( _I_A - _I_K - _I_Na - _I_L ) / _C_m
	WB_dn_dt = ( _alpha_n * ( 1.0 - _n ) ) - ( _beta_n * _n )
	WB_dh_dt = ( _alpha_h * ( 1.0 - _h ) ) - ( _beta_h * _h )
	
	return NM_model(
		name_in = 'Wang-Buzsaki Model',
		model_naming_in = [
			'voltage / dt',
			'K current / dt',
			'leak current / dt',
		],
		model_expr_in = [
			WB_dv_dt,
			WB_dn_dt,
			WB_dh_dt,
		],
		lst_vars_in = [ _V_m, _n, _h ],
		dict_syms = deepcopy(WB_consts),
		stim_sym_in = _I_A,
		dict_units = None,
	)