import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

# from mpl_toolkits import mplot3d
# import matplotlib.cm as cm
# from matplotlib.path import Path
# import matplotlib.gridspec as gridspec








#######  ######## ######## ##     ## ########
##    ## ##          ##    ##     ## ##     ##
##       ##          ##    ##     ## ##     ##
 ######  ######      ##    ##     ## ########
      ## ##          ##    ##     ## ##
##    ## ##          ##    ##     ## ##
 ######  ########    ##     #######  ##

def f_dxr(f_dx):
	def func(r):
		def func_n(x):
			return f_dx(r,x)
		return func_n
	return func

def f_dx0(f_dx):
	def func(r,x):
		return (0, f_dx(r,x))
	return func

# DEFAULT RANGES:
x_min = -10.0
x_max = 10.0
y_min = -10.0
y_max = 10.0
x_range = x_max - x_min
y_range = y_max - y_min

# trim factor for vector field
TF = 0.05

def Lsp(numPts_x, numPts_y):
	return (np.linspace(x_min, x_max, numPts_x), np.linspace(y_min, y_max, numPts_y))

def Lsp_trim(numPts_x, numPts_y):
	return (np.linspace(x_min + TF * x_range, x_max - TF * x_range, numPts_x), np.linspace(y_min + TF * y_range, y_max - TF * y_range, numPts_y))

def update_ranges(in_fig, in_ax, x_bdd=False, y_bdd=False, TF_n = 0.05, x_label='r', y_label='x*', imgsize=(6, 3), polar=False):
	global fig
	global ax
	fig = in_fig
	ax = in_ax

	global TF
	TF = TF_n

	if polar == False:
		if x_bdd != False:
			global x_min
			global x_max
			
			x_min, x_max = x_bdd
			plt.xlim(x_min, x_max)
			
			global x_range
			x_range = x_max - x_min

		if y_bdd != False:
			global y_min
			global y_max
			
			y_min, y_max = y_bdd
			plt.ylim(y_min, y_max)

			global y_range
			y_range = y_max - y_min

		plt.xlabel(x_label)
		plt.ylabel(y_label)

		ax.xaxis.label.set_color('k')
		ax.yaxis.label.set_color('k')

		ax.spines['bottom'].set_color('k')
		ax.spines['left'].set_color('k')
		ax.spines['right'].set_color('k')
		ax.spines['top'].set_color('k')

	if polar:
		# if polar
		ax.set_rticks([])
		# ax.set_rticks([0.0, 1.0])
		ax.grid=True
		# ax.set_thetagrids((0,45,90,135,180,225,270,315), labels=(' ',' ',' ',' ',' ',' ',' ',' '), color='k')

	if fig != False:
		fig.set_size_inches(*imgsize)
		
		fig.patch.set_facecolor('w')
		fig.patch.set_alpha(0.01)

	ax.tick_params(colors='k')
	ax.title.set_color('k')
	ax.set_facecolor('w')



# standard linspace
std_sp = np.linspace(x_min, x_max, 100)
std_sp_x = np.linspace(x_min, x_max, 100)
std_sp_y = np.linspace(y_min, y_max, 100)



##     ## ####  ######   ######
###   ###  ##  ##    ## ##    ##
#### ####  ##  ##       ##
## ### ##  ##   ######  ##
##     ##  ##        ## ##
##     ##  ##  ##    ## ##    ##
##     ## ####  ######   ######

def v_mag(coord):
	return np.sqrt(coord[0]**2.0 + coord[1]**2.0)

# TODO:
# def in_sorted_list(list, val, threshold):
	# assumes list sorted, searches


# takes in f(r) : r -> g(x)
# spits out h : (r,x) -> y
def fctn_converter(func):
	def func_new(r,x):
		return func(r)(x)
	return func_new

# takes an array, threshold
# returns a list of indecies at which consecutive terms are further apart than "thresh"
def get_split_pos(arr, threshold = 0.0001):
	output = []
	for i in range(1,len(arr)-1):
		if arr[i] != None and arr[i-1] != None:
			if np.fabs(arr[i] - arr[i-1]) > threshold:
				output.append(i)
	return output


# inserts "None" into the array at the given indecies
def ins_None(arr, indxs):
	for i in reversed(indxs):
		arr.insert(i, None)
	return arr


# stability enum
class Stab(Enum):
	stable = 0
	unstable = 1
	hlf_R = 2
	hlf_L = -2


 ######   ######## ########
##    ##  ##          ##
##        ##          ##
##   #### ######      ##
##    ##  ##          ##
##    ##  ##          ##
 ######   ########    ##

def get_1Dstab(func, pos):
	# one-dimensional
	# tries points to the left and right
	Delta = 0.01

	if func(pos + Delta) > 0:
		if func(pos - Delta) > 0:
			# both sides greater, half stable
			return Stab.hlf_R
		else:
			# right greater, left smaller, unstable
			return Stab.unstable
	else:
		if func(pos - Delta) > 0:
			# right smaller, left greater, stable
			return Stab.stable
		else:
			# both smaller, half stable
			return Stab.hlf_L
		
def get_1Dzero( func, pos, testRange = (-1000.0,1000.0), threshold = 0.00001, maxN = 1e2 ):
	# uses simple gradient descent to find nearest zero
	# one-dimensional
	p_min = testRange[0]
	p_max = testRange[1]
	
	# threshold: once close enough, return
	
	# coeff of step size vs. point used to estimate slope
	k = 0.00001

	n=0

	while np.fabs(func(pos)) > threshold:
		# check inside testRange
		if pos < p_min:
			return False
		
		if pos > p_max:
			return False
		
		# adjust delta for slope estimation
		delta = k * func(pos)**2
		# get slope at pos
		m = (func(pos) - func(pos + delta)) / delta
		if np.fabs(m) < threshold:
			return False
		# find intersection, reset pos
		pos = pos + func(pos) / m

		n = n+1
		if n > maxN:
			return False

	# print("zero at " + str(pos))
	return pos


# TODO:
# def get_all_1Dzero(func, range = (-100,100), threshold = 0.001):
# 	# empty list of zeroes
# 	zeroes = []
# 	# list of ranges left to check
# 	ranges = [range]

# 	# while there are ranges left to check
# 	while len(ranges) > 0:
# 		# pop off a range
# 		r = ranges.pop()
# 		# get the zeroes nearest the edges of the range


# 		# sort list

	

########  ##        #######  ########
##     ## ##       ##     ##    ##
##     ## ##       ##     ##    ##
########  ##       ##     ##    ##
##        ##       ##     ##    ##
##        ##       ##     ##    ##
##        ########  #######     ##

# plots a 3-D surface
def plot_surf( func, sp = (std_sp_x, std_sp_y) ):
	# func is a function taking x,y and returning z

	# linspaces
	x = sp[0]
	y = sp[1]

	X, Y = np.meshgrid(x, y)
	Z = func(X, Y)
	ax = plt.axes(projection='3d')

	# SURFACE VS CONTOUR --- comment out one
	# ax.contour3D(X, Y, Z, 50, cmap='binary')
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')


def plot_graph_xy(func, x = std_sp_x, clr='k'):
	# func should take in x and give a y
	plt.plot( x, func(x), ls='-', color=clr)


#                           fff iii        lll      dd
#    vv   vv   eee    cccc ff         eee  lll      dd
#     vv vv  ee   e cc     ffff iii ee   e lll  dddddd
#      vvv   eeeee  cc     ff   iii eeeee  lll dd   dd
#       v     eeeee  ccccc ff   iii  eeeee lll  dddddd



def plot_vecfield(func, sp = (std_sp_x, std_sp_y), small_v = False, S = 1.0, norm_mag = True ):
	# func takes in x,y, returns a vector

	# linspaces
	x = sp[0]
	y = sp[1]

	# creating grid
	(X,Y) = np.meshgrid(x,y)
	# vector valued fctn
	u,v = func(X,Y)
	# u = 2*X*Y
	# v = x**2 + 2 * Y - 4
	# magnitude
	r = np.sqrt(u ** 2.0 + v ** 2.0)
	# normalization of u,v
	u = u / r 
	v = v / r
	# r = np.full(len(r), 1)
	if norm_mag:
		r = r / r

	if small_v:
		Q = plt.quiver(X,Y,u,v,r, pivot = "mid", units='width', scale_units = 'dots', scale = 0.07 / S, headlength = 3, headwidth = 2.5, width = 0.004 * S)
		plt.quiverkey(Q, 0.9, 0.9, 2, "", labelpos='E', coordinates='figure') 
	else:
		plt.quiver(X,Y,u,v,r, pivot = "mid")
	
	# streamplot(X,Y,u,v,r)
	

def plot_vecfield_1D(func, sp = std_sp_x ):
	# func takes in x, spits a 1d vector

	# linspaces
	x = sp
	y = np.linspace(0.0, 1.0, 1)

	# creating grid
	(X,Y) = np.meshgrid(x,y)
	# vector valued fctn
	u = func(x)
	v = [0]
	# magnitude
	r = np.fabs(u)
	# normalization of u,v
	u = u / r
	r = r / r

	plt.quiver(X,Y,u,v,r, pivot = "mid")


#                    tt    hh
#    pp pp     aa aa tt    hh
#    ppp  pp  aa aaa tttt  hhhhhh
#    pppppp  aa  aaa tt    hh   hh
#    pp       aaa aa  tttt hh   hh
#    pp

def plot_path_2D(func, coord, rangemax = 100, dt = .01):
	# initial point
	plt.plot( *coord, 'bo' )
	for n in range(0, 100):
		# vec valued fctn
		coord_vec = func(*coord)
		coord_vec = dt * coord_vec / v_mag(coord_vec)
		x = coord[0] + coord_vec[0]
		y = coord[1] + coord_vec[1]
		coord = (x,y)
		plt.plot( *coord, 'r.' )

def plot_path_1D(func, x, maxT = 5, dt = 0.001):
	# func should take in x, spit out 1D vector

	# labels
	xl = plt.xlabel("t")
	yl = plt.ylabel("x(t)")
	
	# initial point
	t = 0.0
	plt.plot( t, x, 'bo' )

	n_steps = int(maxT / dt)

	# array
	arr_t = np.linspace(0, maxT, n_steps )
	arr_x = np.linspace(0, maxT, n_steps )

	for n in range(0,n_steps):
		x = x + dt * func(x)
		arr_x[n] = x
		t = t + dt
	
	plt.plot( arr_t, arr_x, '-r' )


#     fff iii                    dd
#    ff       xx  xx   eee       dd
#    ffff iii   xx   ee   e  dddddd
#    ff   iii   xx   eeeee  dd   dd
#    ff   iii xx  xx  eeeee  dddddd


def plot_1Dfixed( pos, pt_type = Stab.stable, y_pos = 0):
	if pt_type == Stab.stable:
		plt.plot( pos, y_pos, 'o', markersize=7, markerfacecolor='black', markeredgecolor='black', markeredgewidth=2)
	
	if pt_type == Stab.unstable:
		plt.plot( pos, y_pos, 'o', markersize=7, markerfacecolor='white', markeredgecolor='black', markeredgewidth=2)
	
	if pt_type == Stab.hlf_R:
		plt.plot( pos, y_pos, 'o', markersize=10, markerfacecolor='white', markeredgecolor='black', markeredgewidth=2)
		plt.plot( pos, y_pos, '>', markersize=5, markerfacecolor='black', markeredgecolor='black', markeredgewidth=2)
	
	if pt_type == Stab.hlf_L:
		plt.plot( pos, y_pos, 'o', markersize=10, markerfacecolor='white', markeredgecolor='black', markeredgewidth=2)
		plt.plot( pos, y_pos, '<', markersize=5, markerfacecolor='black', markeredgecolor='black', markeredgewidth=2)


def plot_1Dfixed_near(func, pos, testRange = (-1000.0, 1000.0), threshold = 0.00001):
	# plots a zero by the given point
	x = get_1Dzero(func, pos, testRange, threshold)
	if x != False:
		plot_1Dfixed(x, get_1Dstab(func, x))



#    bb      iii  fff
#    bb          ff   uu   uu rr rr
#    bbbbbb  iii ffff uu   uu rrr  r
#    bb   bb iii ff   uu   uu rr
#    bbbbbb  iii ff    uuuu u rr


def plot_bifur_cntr( func, sp = (std_sp_x, std_sp_y)):
	# func_o: only (r) given, returns a function for that r
	# that function takes in x and gives 1D vector

	# linspaces
	r = sp[0]
	x = sp[1]

	R, X = np.meshgrid(r, x)
	Z = func(R, X)

	plt.contour(R, X, Z, [0])


def plot_bifur( func, sp = (std_sp_x, std_sp_y), threshold = 0.001, connect_S = True, connect_U = True, plot_hlf = True, draw_S = True, draw_U = True, draw_U_over = False ):
	# func: only (r) given, returns a function for that r
	# that function takes in x and gives 1D vector

	# linspaces
	R = sp[0]
	X = sp[1]

	x_step = (X[1] - X[0]) * 1.01

	# stable coords
	S_crds_R = []
	S_crds_X = []
	# unstable coords
	U_crds_R = []
	U_crds_X = []
	# half stable coords
	HL_crds_R = []
	HL_crds_X = []
	HR_crds_R = []
	HR_crds_X = []

	# plt.axes(xlabel = 'r', ylabel = 'x fixed pts')

	# for every r, iterate over every x and show any fixed points
	for r in np.nditer(R):
		for i in range(1, len(X)-1):
			# find the nearest zero to the point, plot
			x = get_1Dzero(func(r), X[i], (X[i] - x_step, X[i] + x_step), threshold)
			if x != False:
				# sort by stability, append to appropriate array
				stab = get_1Dstab(func(r), x)
				if stab == Stab.stable:
					S_crds_R.append(r)
					S_crds_X.append(x)
				if stab == Stab.unstable:
					U_crds_R.append(r)
					U_crds_X.append(x)
				if stab == Stab.hlf_L:
					HL_crds_R.append(r)
					HL_crds_X.append(x)
				if stab == Stab.hlf_R:
					HR_crds_R.append(r)
					HR_crds_X.append(x)

	# split where points are too far apart
	S_crds_split = get_split_pos(S_crds_R, 0.5)
	U_crds_split = get_split_pos(U_crds_R, 0.5)

	S_crds_R = ins_None(S_crds_R, S_crds_split)
	S_crds_X = ins_None(S_crds_X, S_crds_split)

	U_crds_R = ins_None(U_crds_R, U_crds_split)
	U_crds_X = ins_None(U_crds_X, U_crds_split)

	S_crds_split = get_split_pos(S_crds_X, .9)
	U_crds_split = get_split_pos(U_crds_X, .9)

	S_crds_R = ins_None(S_crds_R, S_crds_split)
	S_crds_X = ins_None(S_crds_X, S_crds_split)

	U_crds_R = ins_None(U_crds_R, U_crds_split)
	U_crds_X = ins_None(U_crds_X, U_crds_split)


	# graph
	# draw stable points
	if draw_S:
		if connect_S:
			plt.plot( S_crds_R, S_crds_X, 'k-', markersize = 2 )
		else:
			plt.plot( S_crds_R, S_crds_X, 'k.', markersize = 2 )

	# draw unstable points
	if draw_U:	
		if draw_U_over:
			plt.plot( U_crds_R, U_crds_X, 'w.', markersize = 2 )

		if connect_U:
			plt.plot( U_crds_R, U_crds_X, 'k--' )
		else:
			plt.plot( U_crds_R, U_crds_X, 'c.',  markersize=2 )
	
	# draw half-stable points
	if plot_hlf:
		plt.plot( HL_crds_R, HL_crds_X, 'b.' )
		plt.plot( HR_crds_R, HR_crds_X, 'r.' )



   ##   ########     ########  ##        #######  ########
 ####   ##     ##    ##     ## ##       ##     ##    ##
   ##   ##     ##    ##     ## ##       ##     ##    ##
   ##   ##     ##    ########  ##       ##     ##    ##
   ##   ##     ##    ##        ##       ##     ##    ##
   ##   ##     ##    ##        ##       ##     ##    ##
 ###### ########     ##        ########  #######     ##

def plot_1D_all(func, x_range = (-10.0, 10.0), pts_curve = 100, pts_arrows = 10, pts_zeroes = 10, threshold = 0.00001 ):
	x_min = x_range[0]
	x_max = x_range[1]
	sp_x = np.linspace(x_min, x_max, pts_curve)
	sp_z = np.linspace(x_min, x_max, pts_zeroes)
	
	plot_graph_xy(func, sp_x)
	plot_vecfield_1D(func, np.linspace(x_min, x_max, pts_arrows))

	x_step = (sp_x[1] - sp_x[0]) * 1.01
	for i in range(1, len(sp_z)-1):
		plot_1Dfixed_near(func, sp_z[i], (sp_z[i] - x_step, sp_z[i] + x_step), threshold)



########   #######  ##          ###    ########
##     ## ##     ## ##         ## ##   ##     ##
##     ## ##     ## ##        ##   ##  ##     ##
########  ##     ## ##       ##     ## ########
##        ##     ## ##       ######### ##   ##
##        ##     ## ##       ##     ## ##    ##
##         #######  ######## ##     ## ##     ##


# TODO: polar vector field?

# def plot_polar_vecfield(func):
# 	# func takes in theta, spits out a 1d vector

# 	# linspaces
# 	x = sp
# 	y = np.linspace(0.0, 1.0, 1)

# 	# creating grid
# 	(X,Y) = np.meshgrid(x,y)
# 	# vector valued fctn
# 	u = func(x)
# 	v = [0]
# 	# magnitude
# 	r = np.fabs(u)
# 	# normalization of u,v
# 	u = u / r
# 	r = r / r

# 	plt.quiver(X,Y,u,v,r, pivot = "mid")

def plot_polarfixed( pos, pt_type = Stab.stable, y_pos = 1.0):
	if pt_type == Stab.stable:
		plt.plot( pos, y_pos, 'o', markersize=7, markerfacecolor='black', markeredgecolor='black', markeredgewidth=2)
	
	if pt_type == Stab.unstable:
		plt.plot( pos, y_pos, 'o', markersize=7, markerfacecolor='white', markeredgecolor='black', markeredgewidth=2)
	
	if pt_type == Stab.hlf_R:
		plt.plot( pos, y_pos, 'o', markersize=10, markerfacecolor='white', markeredgecolor='black', markeredgewidth=2)
		plt.plot( pos, y_pos, '>', markersize=5, markerfacecolor='black', markeredgecolor='black', markeredgewidth=2)
	
	if pt_type == Stab.hlf_L:
		plt.plot( pos, y_pos, 'o', markersize=10, markerfacecolor='white', markeredgecolor='black', markeredgewidth=2)
		plt.plot( pos, y_pos, '<', markersize=5, markerfacecolor='black', markeredgecolor='black', markeredgewidth=2)


def plot_polarfixed_near(func, pos, testRange = (0.0, 2 * np.pi), threshold = 0.0001, yshift=1.0, yscale=0.1):
	# plots a zero by the given point
	x = get_1Dzero(func, pos, testRange, threshold)
	if x != False:
		plot_polarfixed(x, get_1Dstab(func, x), y_pos=yshift)


def plot_polar(func, pts_curve = 100, pts_arrows = 10, pts_zeroes = 30, threshold = 0.0001, yshift=1.0, yscale=0.1 ):
	sp_theta = np.linspace(0, 2 * np.pi, pts_curve)
	sp_z = np.linspace(0, 2 * np.pi, pts_curve)

	def func_radplot(theta):
		return (func(theta) * yscale) + yshift

	# draw ref circles
	plt.plot(sp_theta, np.full(len(sp_theta), yshift), color = '0.5', ls='--')
	plt.plot(sp_theta, np.full(len(sp_theta), yshift+yscale), color = '0.75', ls='--')
	plt.plot(sp_theta, np.full(len(sp_theta), yshift-yscale), color = '0.75', ls='--')
	plt.plot(sp_theta, np.full(len(sp_theta), yshift-2*yscale), color = 'w')
	plt.plot(sp_theta, np.full(len(sp_theta), yshift-3*yscale), color = 'w')
	# draw func
	plt.plot(sp_theta, func_radplot(sp_theta), 'k-')

	# plot_vecfield_1D(func, np.linspace(x_min, x_max, pts_arrows))

	theta_step = (sp_theta[1] - sp_theta[0]) * 1.01
	for i in range(1, len(sp_z)-1):
		plot_polarfixed_near(func, sp_z[i], (sp_z[i] - theta_step, sp_z[i] + theta_step), threshold, yshift=yshift, yscale=yscale)



 #######  ########     ########     ###    ######## ##     ##
##     ## ##     ##    ##     ##   ## ##      ##    ##     ##
       ## ##     ##    ##     ##  ##   ##     ##    ##     ##
 #######  ##     ##    ########  ##     ##    ##    #########
##        ##     ##    ##        #########    ##    ##     ##
##        ##     ##    ##        ##     ##    ##    ##     ##
######### ########     ##        ##     ##    ##    ##     ##

def iter_point( func, coord_in, dt=0.1 ):
	# get vector at pt
	coord_vec = func(*coord_in)
	# get magnitude, modify vector
	mag = np.sqrt((coord_vec[0] ** 2.0) + (coord_vec[1] ** 2.0))
	# coord_vec =  coord_vec / mag
	# iterate current pt, return
	x = coord_in[0] + (dt * coord_vec[0] / mag)
	y = coord_in[1] + (dt * coord_vec[1] / mag)
	return (x,y)

def draw_path(func, coord, dt=0.001, max_n=1000):
	# initial point
	plt.plot( *coord, 'bo' )
	x_coords = [coord[0]]
	y_coords = [coord[1]]
	for n in range(0, max_n):
		coord = iter_point(func, coord, dt)
		x_coords.append(coord[0])
		y_coords.append(coord[1])

	plt.plot( x_coords, y_coords, 'r-' )











 #######  ##       ########
##     ## ##       ##     ##
##     ## ##       ##     ##
##     ## ##       ##     ##
##     ## ##       ##     ##
##     ## ##       ##     ##
 #######  ######## ########


# def plot_stream(X,Y,U,V,speed):
# 	w=15

# 	plt.figure(figsize=(7, 9))
# 	gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

# 	# points
# 	seed_points = np.array([[-4, -4], [-1, -2]])

# 	# ax3 = fig.add_subplot(gs[1, 1])
# 	strm = ax3.streamplot(X, Y, U, V, color=U, linewidth=2,
# 						cmap='autumn', start_points=seed_points.T)
# 	fig.colorbar(strm.lines)
# 	ax3.set_title('Controlling Starting Points')

# 	# Displaying the starting points with blue symbols.
# 	ax3.plot(seed_points[0], seed_points[1], 'bo')
# 	ax3.axis((-w, w, -w, w))


# def plot_manyR(r_vals, r_fctn):
# 	# r_vals should be an array of r values to try
# 	# r_fctn should take r as first arg, x as second
	
# 	# assumes r_vals is sorted
# 	# normalization for color scheme
# 	r_total_range = r_vals[len(r_vals) - 1] - r_vals[0]
# 	def norm_r(in_val):
# 		return (in_val - r_vals[0]) / r_total_range
	
# 	# graph all functions
# 	# for r in r_vals:








def plot_bifur_all( func, sp = (std_sp_x, std_sp_y), threshold = 0.0001 ):
	# func takes in (r,x), returns a vector
	
	# linspaces
	R = sp[0]
	X = sp[1]

	# ax = plt.axes()
	# ax.set_xlabel('r')
	# ax.set_ylabel('x fixed pts')

	# for every r, iterate over every x and show any fixed points
	for r in np.nditer(R):
		for x in np.nditer(X):
			# if the vector value is less than the threshold, plot a point
			if np.fabs(func(r,x)) < threshold:
				plt.plot( r, x, 'r.' )