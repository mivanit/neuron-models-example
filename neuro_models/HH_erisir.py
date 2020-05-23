'''
Erisir neuron model
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
_alpha_n, _beta_n, _alpha_n, _beta_n, _alpha_n, _beta_n = sym.symbols('alpha_n beta_n alpha_n beta_n alpha_n beta_n')
# steady states
_n_inf, _m_inf, _h_inf = sym.symbols('n_inf m_inf h_inf')


HHE_consts = {
	_C_m : 1.0,
	_E_Na : 60.0,
	_E_K : -90.0,
	_E_L : -70,
	_g_Na : 112.0,
	_g_K : 224.0,
	_g_L : 0.5,
}



# Sodium ion-channel rate functions
_alpha_m = 40 * ( 75.5 - _V_m ) / ( sym.exp( ( 75.5 - _V_m ) / 13.5 ) - 1)
_beta_m = 1.2262 * sym.exp( - _V_m  / 42.248 )

# leak channel rate values
_alpha_h = 0.0035 * sym.exp( - _V_m / 24.186 )
_beta_h = -0.017 * ( _V_m + 51.25 ) / ( sym.exp( - ( _V_m + 51.25 ) / 5.2 ) - 1 )

# Potassium ion-channel rate functions
_alpha_n = ( 95.0 - _V_m )/( sym.exp( ( 95.0 - _V_m ) / 11.8 ) - 1)
_beta_n = 0.025 * sym.exp( - _V_m / 22.222 )


# n, m, and h steady-state values
_n_inf = _alpha_n / ( _alpha_n + _beta_n )
_m_inf = _alpha_m / ( _alpha_m + _beta_m )
_h_inf = _alpha_h / ( _alpha_h + _beta_h )
  

# Erisir model expressions

# currents
_I_K = _g_K * (_n ** 2.0) * ( _V_m - _E_K )
_I_Na = _g_Na * ( _m_inf ** 3.0 ) * _h * (_V_m - _E_Na)
_I_L = _g_L * (_V_m - _E_L)

# diffeqs

HHE_dv_dt = ( _I_A - _I_K - _I_Na - _I_L ) / _C_m
HHE_dn_dt = ( _alpha_n * ( 1.0 - _n ) ) - ( _beta_n * _n )
HHE_dh_dt = ( _alpha_h * ( 1.0 - _h ) ) - ( _beta_h * _h )

# Rate Function Constants (RFC)
rfc = {
	'an_1' :  95.0,
	'an_2' :  11.8,
	
	'bn_1' :  0.025,
	'bn_2' :  22.222,
	
	'am_1' :  75.0,
	'am_2' :  40.0,
	'am_3' :  13.5,
	
	'bm_1' :  1.2262,
	'bm_2' :  42.248,
	
	'ah_1' :  0.0035,
	'ah_2' :  24.186,
	
	'bh_1' :  -0.017,
	'bh_2' :  51.25,
	'bh_3' :  5.2,
}


def get_model():
	return NM_model(
		name_in = 'Erisir Model',
		model_naming_in = [
			'voltage / dt',
			'K current / dt',
			'leak current / dt',
		],
		model_expr_in = [
			HHE_dv_dt,
			HHE_dn_dt,
			HHE_dh_dt,
		],
		lst_vars_in = [ _V_m, _n, _h ],
		dict_syms = deepcopy(HHE_consts),
		stim_sym_in = _I_A,
		dict_units = None,
	)