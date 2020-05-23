'''
Morris–Lecar neuron model
'''

#%%

import numpy as np
# import mpmath as mp
import sympy as sym
from scipy.integrate import odeint
from copy import deepcopy

from neuro_models.neuroUtil import *

# Average potassium, calcium, leak channel conductance per unit area (mS/cm^2)
_g_K, _g_Ca, _g_L = sym.symbols('g_K g_Ca g_L')
# Average potassium, calcium, leak potentials (mV)
_E_K, _E_Ca, _E_L = sym.symbols('E_K E_Ca E_L')
# capacitance of membrane, applied current, phi
_C_m, _I_A, _phi = sym.symbols('C_m I_A phi')
# membrane voltage, recovery variable (for K)
_V_m, _n = sym.symbols('V_m n')
# tuning parameters
_V_tune_1, _V_tune_2, _V_tune_3, _V_tune_4 = sym.symbols('V_tune_1 V_tune_2 V_tune_3 V_tune_4') 

#* various parameter regiemes

# Hopf bifurcation regieme
ML_consts_Hopf = {
	_g_Ca : 4.4,
	_g_K  : 8.0,
	_g_L  : 2.0,

	_E_Ca : 120.0,
	_E_K  : -84.0,
	_E_L  : 60.0,

	_C_m  : 20.0,
	_phi  : 0.04,

	_V_tune_1 : -1.2,
	_V_tune_2 : 18.0,
	_V_tune_3 : 2.0,
	_V_tune_4 : 30.0,
}

# Saddle Node on a Limit Cycle regieme
ML_consts_SNLC = {
	_g_Ca : 4.0,
	_g_K  : 8.0,
	_g_L  : 2.0,

	_E_Ca : 120.0,
	_E_K  : -84.0,
	_E_L  : 60.0,

	_C_m  : 20.0,
	_phi  : 0.067,

	_V_tune_1 : -1.2,
	_V_tune_2 : 18.0,
	_V_tune_3 : 12.0,
	_V_tune_4 : 17.4,
}

# homoclinic bifurcation regieme
ML_consts_Homoclinic = {
	_g_Ca : 4.0,
	_g_K  : 8.0,
	_g_L  : 2.0,

	_E_Ca : 120.0,
	_E_K  : -84.0,
	_E_L  : 60.0,

	_C_m  : 20.0,
	_phi  : 0.23,

	_V_tune_1 : -1.2,
	_V_tune_2 : 18.0,
	_V_tune_3 : 12.0,
	_V_tune_4 : 17.4,
}

ML_consts_regiemes = {
	'Hopf' : ML_consts_Hopf,
	'SNLC' : ML_consts_SNLC,
	'Homoclinic' : ML_consts_Homoclinic,
}


# steady states
_n_inf, _m_inf, _tau_n = sym.symbols('n_inf m_inf tau_n')




# Potassium ion-channel rate functions
_alpha_n = - 0.01 * ( _V_m + 55.0 )/( sym.exp(-( _V_m + 55.0 )/10)-1)
_beta_n = 0.125 * sym.exp(-( _V_m + 65.0 )/80)

# Sodium ion-channel rate functions
_alpha_m = -0.1 * ( _V_m + 40.0 ) / ( sym.exp(-(_V_m + 40.0)/10)-1 )
_beta_m = 4 * sym.exp(-( _V_m + 65 )/18)

# leak channel rate values
_alpha_h = 0.07 * sym.exp( -( _V_m + 65 )/20 )
_beta_h = 1.0 / (sym.exp(-( _V_m + 35 )/10)+1)


# n, m, and h steady-state values
_n_inf = _alpha_n / ( _alpha_n + _beta_n )
_m_inf = _alpha_m / ( _alpha_m + _beta_m )
_h_inf = _alpha_h / ( _alpha_h + _beta_h )


# expressions

# currents
_I_K = _g_K * (_n ** 4.0) * ( _V_m - _E_K )
_I_Na = _g_Na * ( _m ** 3.0 ) * _h * (_V_m - _E_Na)
_I_L = _g_L * (_V_m - _E_L)

# diffeqs

ML_dv_dt = ( _I_A - _I_K - _I_Na - _I_L ) / _C_m
ML_dn_dt = ( _alpha_n * ( 1.0 - _n ) ) - ( _beta_n * _n )
ML_dm_dt = ( _alpha_m * ( 1.0 - _m ) ) - ( _beta_m * _m)
ML_dh_dt = ( _alpha_h * ( 1.0 - _h ) ) - ( _beta_h * _h )


def get_model_HH():
	return NM_model(
		name_in = 'Hodgkin-Huxley model',
		model_naming_in = [
			'voltage / dt',
			'K gate rate / dt',
			'Na gate rate / dt',
			'leak gate rate / dt',
		],
		model_expr_in = [
			ML_dv_dt,
			ML_dn_dt,
			ML_dm_dt,
			ML_dh_dt,
		],
		lst_vars_in = [ _V_m, _n, _m, _h ],
		dict_syms = deepcopy(ML_consts),
		stim_sym_in = _I_A,
		dict_units = None,
		steady_in = np.array([
			-65.0, 
			0.052934217620864, 
			0.596111046346827,
			0.317681167579781,
		])
	)








