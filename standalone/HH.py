# originally copied from
# https://gist.github.com/giuseppebonaccorso/60ce3eb3a829b94abf64ab2b7a56aaef

#%%

import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint

# * consts

# Set random seed (for reproducibility)
np.random.seed(1000)

# Average potassium channel conductance per unit area (mS/cm^2)
gK = 36.0

# Average sodium channel conductance per unit area (mS/cm^2)
gNa = 120.0

# Average leak channel conductance per unit area (mS/cm^2)
gL = 0.3

# Membrane capacitance per unit area (uF/cm^2)
Cm = 1.0

# Potassium potential (mV)
# E_K = -12.0
E_K = -77.0

# Sodium potential (mV)
# E_Na = 115.0
E_Na = 50.0

# Leak potential (mV)
# E_L = 10.613
E_L = -54.4

# Time values
tmin = 0
# tmax = 150
# tmax = 500
tmax = 1150
dt = 0.01

# * funcs

# Potassium ion-channel rate functions

def alpha_n(Vm):
	anv = 55.0
	return -0.01 * ( Vm + anv )/( np.exp(-( Vm + anv )/10)-1)
	# return - (0.01 * (55.0 + Vm)) / (np.exp( - (0.1 * Vm)) - 1.0)

def beta_n(Vm):
	bnv = 65.0
	return 0.125 * np.exp(-( Vm + bnv )/80)
	# return 0.125 * np.exp(-Vm / 80.0)

# Sodium ion-channel rate functions

def alpha_m(Vm):
	return -0.1 * ( Vm + 40.0 ) / ( np.exp(-(Vm + 40.0)/10)-1 )
	# return (0.1 * (25.0 - Vm)) / (np.exp(2.5 - (0.1 * Vm)) - 1.0)

def beta_m(Vm):
	return 4 * np.exp(-( Vm + 65 )/18)
	# return 4.0 * np.exp(-Vm / 18.0)

def alpha_h(Vm):
	return 0.07 * np.exp( -( Vm + 65 )/20 )
	# return 0.07 * np.exp(-Vm / 20.0)

def beta_h(Vm):
	return 1.0 / (np.exp(-( Vm + 35 )/10)+1)
	# return 1.0 / (np.exp(3.0 - (0.1 * Vm)) + 1.0)
  
# n, m, and h steady-state values

def n_inf(Vm=0.0):
	return alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))

def m_inf(Vm=0.0):
	return alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))

def h_inf(Vm=0.0):
	return alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))
  





def compute(
		stim,
		# State (Vm, n, m, h)
		IC = np.array([
			-65.0, 
			0.052934217620864, 
			0.596111046346827,
			0.317681167579781,
		]),
		T = np.linspace(tmin, tmax, (tmax-tmin)//dt), 
		bln_plot = True
	):

	# Compute derivatives
	def compute_derivatives(y, t0):
		dy = np.zeros((4,))
		
		Vm = y[0]
		n = y[1]
		m = y[2]
		h = y[3]
		
		# dVm/dt
		GK = (gK / Cm) * np.power(n, 4.0)
		GNa = (gNa / Cm) * np.power(m, 3.0) * h
		GL = gL / Cm
		
		dy[0] = (
			(stim(t0) / Cm) 		# stimulus current
			- (GK * (Vm - E_K)) 	# potassium
			- (GNa * (Vm - E_Na)) 	# sodium
			- (GL * (Vm - E_L))		# leak
		)
		
		# dn/dt
		dy[1] = (alpha_n(Vm) * (1.0 - n)) - (beta_n(Vm) * n)
		
		# dm/dt
		dy[2] = (alpha_m(Vm) * (1.0 - m)) - (beta_m(Vm) * m)
		
		# dh/dt
		dy[3] = (alpha_h(Vm) * (1.0 - h)) - (beta_h(Vm) * h)
		
		return dy

	# Solve ODE system
	Vy = odeint(compute_derivatives, IC, T)

	
	if bln_plot:
		Idv = [stim(t) for t in T]
		
		# fig, ax = plt.subplots(figsize=(12, 7))
		# ax.plot(T, Idv, 'r-')
		# ax.set_xlabel('Time (ms)')
		# ax.set_ylabel(r'Current density (uA/$cm^2$)')
		# ax.set_title('Stimulus (Current density)')

		# Neuron potential
		fig, ax = plt.subplots(figsize=(12, 7))
		ax.plot(T, Vy[:, 0], 'b-')
		ax.set_xlabel('Time (ms)')
		ax.set_ylabel('Vm (mV)')
		ax.set_title('Neuron potential with two spikes')
		plt.grid()

		ax.plot(T, [x - 57.5 for x in Idv], 'r-')

		# plt.show()
	
	return (T, Vy)

#%%
