import numpy as np
from scipy import constants as spConst
import sympy as sym
import sympy.physics.units as u

from neuro_models.neuroUtil import *


#* models

# GHK
_T, _P, _z, _V, _c_in, _c_out = sym.symbols('T P z V c_in c_out')

_xi = _F * _z * _V / (_R * _T)
	
model_GHK = _P * _F * _z * _xi * ( _c_out * np.e ** (-1 * _xi) - _c_in ) / ( np.e ** (-1 * _xi) - 1 )

model_GHK_units = {
	'self' 	: u.ampere,
	_V		: u.volts,
	_T 		: u.kelvin,
	_P		: u.cm / u.second,
	_z		: u.coulomb,
	_c_in	: u.moles,
	_c_out	: u.moles,
}
