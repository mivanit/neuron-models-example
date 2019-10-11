

# %%




import os
os.chdir(os.path.join(os.getcwd(), 'C:\\Users\\mivan\\Google Drive\\Winter_2019\\Math_404\\hw_04\\'))
# os.chdir(os.path.join(os.getcwd(), 'mivan\\Google Drive\\Winter_2019\\Math_404\\hw_04\\'))
# os.chdir(os.path.join(os.getcwd(), 'Math_404\\hw_04\\'))
# os.chdir(os.path.join(os.getcwd(), ''))
print(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import graph_fctns as gf

######## ######## ##     ## ########  ##          ###    ######## ########
   ##    ##       ###   ### ##     ## ##         ## ##      ##    ##
   ##    ##       #### #### ##     ## ##        ##   ##     ##    ##
   ##    ######   ## ### ## ########  ##       ##     ##    ##    ######
   ##    ##       ##     ## ##        ##       #########    ##    ##
   ##    ##       ##     ## ##        ##       ##     ##    ##    ##
   ##    ######## ##     ## ##        ######## ##     ##    ##    ########


#%%
# ========================================
# TEMPLATE BIFURCATION PLOT CELL
# update ranges
# set function
# label
# plot stuff
# show

def template_bifur():
	fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k')
	gf.update_ranges(fig, ax, (-10.0, 10.0), (-10.0, 10.0))
	plt.title('Bifurcation diagram for  ' + r'$\dot{x} = rx - \frac{x}{1+x}$')

	def f_dx(r,x):
		Dx = r * x - x / (1 + x)
		return Dx
	
	gf.plot_bifur_cntr(f_dx, gf.Lsp(300,300))
	gf.plot_bifur(gf.f_dxr(f_dx), gf.Lsp(100,10), 0.0001)
	gf.plot_vecfield(gf.f_dx0(f_dx), gf.Lsp_trim(8,8) )

	plt.savefig('img\\test', dpi=300, bbox_inches='tight', pad_inches=0.15)

template_bifur()


#%%
# ========================================
# TEMPLATE VECTOR PLOT CELL

def template_1D_vec():
	fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k')
	gf.update_ranges(fig, ax, x_label='x', y_label='Dx', x_bdd=(-1.5,1.5))

	def f_Dx(x):
		r = -1.0
		Dx = r * (x**2.0) + 0.4
		return Dx

	gf.plot_1D_all(f_Dx, x_range=(-1.01, 1.01), pts_arrows=16, threshold=0.001, pts_zeroes=21)

	plt.savefig('img\\test_2', dpi=300)

template_1D_vec()

#%%
def p_1a():
	fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k')
	gf.update_ranges(fig, ax, x_bdd=(-10,10), y_bdd=(-10,10), x_label='x', y_label='y')

	def func(x,y):
		dxt = - y
		dyt = - x
		return (dxt, dyt)

	gf.plot_vecfield(func, gf.Lsp_trim(10,10), norm_mag=False)
	plt.savefig('img\\1a')

p_1a()



















#%%
def p_2a():
	fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k')
	gf.update_ranges(fig, ax, x_bdd=(-1.5, 1.5), y_bdd=(-2.5, 2.5), x_label='x', y_label='y')

	def func(x,y):

		dxt = y
		dyt = -4 * x
		return (dxt, dyt)

	gf.draw_path(func, (0,  1), 0.001, 10000)
	gf.draw_path(func, (0, -1), 0.001, 10000)
	gf.draw_path(func, ( 1, 0), 0.001, 10000)
	gf.draw_path(func, (-1, 0), 0.001, 10000)

	gf.plot_vecfield(func, gf.Lsp_trim(25,25), norm_mag=False)
	
	plt.savefig('img\\2a', dpi=300)

p_2a()




#%%
def p_2b():
	fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k')
	gf.update_ranges(fig, ax, x_bdd=(-1.5, 1.5), y_bdd=(-2.5, 2.5), x_label='x', y_label='y')

	def func(x,y):

		dxt = 0
		dyt = - y
		return (dxt, dyt)

	gf.draw_path(func, (0,  1), 0.001, 10000)
	gf.draw_path(func, (0, -1), 0.001, 10000)
	gf.draw_path(func, ( 1, 1), 0.001, 10000)
	gf.draw_path(func, (-1,-1), 0.001, 10000)

	gf.plot_vecfield(func, gf.Lsp_trim(20,10), norm_mag=False)
	
	plt.savefig('img\\2b', dpi=300)

p_2b()



#%%
def p_2c():
	fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k')
	gf.update_ranges(fig, ax, x_bdd=(-1.5, 1.5), y_bdd=(-2.5, 2.5), x_label='x', y_label='y')

	def func(x,y):

		dxt = -x
		dyt = -5 * y
		return (dxt, dyt)

	gf.draw_path(func, (0,  1), 0.001, 10000)
	gf.draw_path(func, (0, -1), 0.001, 10000)
	gf.draw_path(func, ( 1, 1), 0.001, 10000)
	gf.draw_path(func, (-1,-1), 0.001, 10000)

	gf.plot_vecfield(func, gf.Lsp_trim(20,10), norm_mag=False)
	
	plt.savefig('img\\2c', dpi=300)

p_2c()














#%%
def p_3():
	fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k')
	gf.update_ranges(fig, ax, x_bdd=(-1.5, 1.5), y_bdd=(-2.5, 2.5), x_label='x', y_label='y')

	def func(x,y):

		dxt = 4 * x - y
		dyt = 2 * x + y
		return (dxt, dyt)

	gf.draw_path(func, (0,  1), 0.001, 10000)
	gf.draw_path(func, (0, -1), 0.001, 10000)
	gf.draw_path(func, ( 1, 1), 0.001, 10000)
	gf.draw_path(func, (-1,-1), 0.001, 10000)

	gf.plot_vecfield(func, gf.Lsp_trim(20,10), norm_mag=False)
	
	plt.savefig('img\\3', dpi=300)

p_3()
















#%%
def p_5a():
	fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k')
	gf.update_ranges(fig, ax, x_bdd=(-2,2), y_bdd=(-2,2), x_label='x', y_label='y')

	def func(x,y):
		dxt = 5.0 * x + 10.0 * y
		dyt = - x - y
		return (dxt, dyt)

	def nullc_1(x):
		y = - x / 2.0
		return y

	def nullc_2(x):
		y = - x
		return y

	gf.plot_graph_xy(nullc_1)
	gf.plot_graph_xy(nullc_2)

	gf.draw_path(func, (0,  0.001), 0.001, 10000)
	gf.draw_path(func, (0, -0.001), 0.001, 10000)
	# gf.draw_path(func, ( 0.001, 0), 0.001, 10000)
	# gf.draw_path(func, (-0.001, 0), 0.001, 10000)

	gf.plot_vecfield(func, gf.Lsp_trim(25,25), norm_mag=False)
	# plt.savefig('img\\5a')
	plt.savefig('img\\5a_2', dpi=300)

p_5a()


#%%
def p_5b():
	fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k')
	gf.update_ranges(fig, ax, x_bdd=(-1.1, 1.1), y_bdd=(-1.1, 1.1), x_label='x', y_label='y')

	def func(x,y):
		# $\td{x} = -3x + 2y,\ \td{y} = x-2y$
		dxt = -3.0 * x + 2.0 * y
		dyt = x - 2.0 * y
		return (dxt, dyt)

	gf.draw_path(func, (0,  1), 0.001, 10000)
	gf.draw_path(func, (0, -1), 0.001, 10000)
	gf.draw_path(func, ( 1, 0), 0.001, 10000)
	gf.draw_path(func, (-1, 0), 0.001, 10000)

	gf.plot_vecfield(func, gf.Lsp_trim(25,25), norm_mag=False)
	# plt.savefig('img\\5a')
	plt.savefig('img\\5b', dpi=300)

p_5b()


# %%


#%%
def p_6():
	fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k')
	gf.update_ranges(fig, ax, x_bdd=(-1.1, 1.1), y_bdd=(-1.1, 1.1), x_label='x', y_label='y')

	k = 1.0
	m = 1.0
	b = 1.0 * np.sqrt(4.0 * k * m)

	plt.title('Phase plot for  ' + r'$m\ddot{x} + b\dot{x} + kx = 0 \quad$' + f'm = {m}, b = {b}, k = {k}')

	def func(x,y):

		dxt = m * y
		dyt = (k * x + b * y) / (- m)
		return (dxt, dyt)

	gf.draw_path(func, (0,  1), 0.001, 10000)
	gf.draw_path(func, (0, -1), 0.001, 10000)
	gf.draw_path(func, ( 1, 0), 0.001, 10000)
	gf.draw_path(func, (-1, 0), 0.001, 10000)

	gf.plot_vecfield(func, gf.Lsp_trim(25,25), norm_mag=False)
	
	i_k, i_m, i_b = int(k), int(m), int(b)
	plt.savefig('img\\6_kmb_' + f'{i_k}_{i_m}_{i_b}', dpi=300)

p_6()






#%%
def p_7():
	fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k')
	gf.update_ranges(fig, ax, x_bdd=(-1.1, 1.1), y_bdd=(-1.1, 1.1), x_label='R', y_label='J')

	a = 2.0
	b = -1.0

	plt.title('Phase plot for  ' + r'$\dot{R} = aJ,\ \dot{J} = bR \quad$' + f'a = {a}, b = {b}')

	def func(R,J):

		dRt = a * J
		dJt = b * R
		return (dRt, dJt)

	gf.draw_path(func, ( .5,  .5), 0.001, 10000)
	gf.draw_path(func, ( .5, -.5), 0.001, 10000)
	gf.draw_path(func, (-.5,  .5), 0.001, 10000)
	gf.draw_path(func, (-.5, -.5), 0.001, 10000)

	gf.draw_path(func, (0,  .5), 0.001, 10000)
	gf.draw_path(func, (0, -.5), 0.001, 10000)
	gf.draw_path(func, ( .5, 0), 0.001, 10000)
	gf.draw_path(func, (-.5, 0), 0.001, 10000)

	gf.plot_vecfield(func, gf.Lsp_trim(25,25), norm_mag=False)
	
	i_a, i_b = int(a), int(b)
	# plt.savefig('img\\7_ab_' + f'{i_a}_{i_b}', dpi=300)

p_7()










#%%
def p_8():
	fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='w', edgecolor='k')
	gf.update_ranges(fig, ax, x_bdd=(-1.1, 1.1), y_bdd=(-1.1, 1.1), x_label='R', y_label='J')

	a = -1.0
	b = 1.0

	plt.title('Phase plot for  ' + r'$\dot{R} = 0,\ \dot{J} = aR + bJ \quad$' + f'a = {a}, b = {b}')

	def func(R,J):

		dRt = 0
		dJt = a * R + b * J
		return (dRt, dJt)

	gf.draw_path(func, ( .5,  .5), 0.001, 10000)
	gf.draw_path(func, ( .5, -.5), 0.001, 10000)
	gf.draw_path(func, (-.5,  .5), 0.001, 10000)
	gf.draw_path(func, (-.5, -.5), 0.001, 10000)

	gf.draw_path(func, (0,  .75), 0.001, 10000)
	gf.draw_path(func, (0, -.75), 0.001, 10000)
	gf.draw_path(func, ( .75, 0), 0.001, 10000)
	gf.draw_path(func, (-.75, 0), 0.001, 10000)

	gf.plot_vecfield(func, gf.Lsp_trim(25,25), norm_mag=False)
	
	i_a, i_b = int(a), int(b)
	# plt.savefig('img\\8_ab_' + f'{i_a}_{i_b}', dpi=300)

p_8()