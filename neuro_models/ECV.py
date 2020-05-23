import numpy as np
from scipy import constants as spConst
import sympy as sym
import sympy.physics.units as u

from neuro_models.neuroUtil import *


# ECV (equivalent curcuit)

_t, _V_m, _r_m, _i_A, _c_m, _E_rest = sym.symbols('t V_m r_m i_A c_m E_rest')

# diffeq
# equals dV / dt
# model_ECV_diffeq = 

# time dependent
model_ECV_timeDep = _r_m * _i_A * (1 - np.e**( - _t / ( _r_m * _c_m ) )) + _E_rest















