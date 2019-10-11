
import numpy as np
import matplotlib
matplotlib.rcParams["savefig.directory"] = os.path.dirname(__file__)
import matplotlib.pyplot as plt
from scipy import optimize as spOpt
from scipy import integrate as spInt
import sympy as sym
import sympy.physics.units as u


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


class neuron(object):

	def __init__(
			self, 
			model_in,
			time_bounds_in = (0.0, 1500.0)
		):
		self.model = model_in

		self.time_bounds = time_bounds_in
		# start_compute_time = time_bdds[]
		# start_compute_idx = int(start_compute_time / model.dt)


	def get_spike_times( time_arr, wf, theshold = 0.0, start_idx = start_compute_idx ):
		'''
		return a list of spike peak times
		'''

		# array of all indecies where greater than 0
		result = np.argwhere(wf[ start_idx: ] > theshold)

		# iterate through `result` and get the first timepoint of each section
		# 	of consecutive indecies
		spike_idxs = []

		# special case for first spike
		if len(result) > 1:
			spike_idxs.append(result[0])
			
		# all other spikes
		for i in range(1,len(result)):
			# discard wherever the last value was consecutive
			if (result[i-1] < result[i] - 1):
				spike_idxs.append(result[i])

		return [time_arr[j + start_idx] for j in spike_idxs]

	def ds3_p1(lst_amps = np.arange(-5.0, 15.0, 1.0)):
		lst_spikeLists = []
		lst_timeArr = []
		lst_waveforms = []

		for amp in lst_amps:

			T, wf = HH.compute(get_stim_func_constPulse(amp), bln_plot = False)

			spike_times = get_spike_times(T, wf[:,0])

			lst_spikeLists.append(spike_times)
			lst_timeArr.append(T)
			lst_waveforms.append(wf)

			print('for amp = \t %f, found num_spikes = \t %d' % (amp, len(spike_times)))

		return (lst_amps, lst_spikeLists, lst_timeArr, lst_waveforms)


	#* amp vs num_spikes
	def plot_amp_vs_numSpikes(lst_amps, lst_spikeLists):
		plt.plot(lst_amps, [len(x) for x in lst_spikeLists], 'r.')
		plt.xlabel('Induced current')
		plt.ylabel('Number of spikes')
		plt.grid()

	#* amp vs spike times
	def plot_amp_vs_spikeTimes(lst_amps, lst_spikeLists):
		for i in range(len(lst_amps)):
			plt.plot([lst_amps[i]] * len(lst_spikeLists[i]), lst_spikeLists[i], 'b.')
		plt.xlabel('Induced current')
		plt.ylabel('timing of spike')
		plt.grid()

	#* print a given set of spikes
	def print_spike(
			test_spikeAmp, 
			pulse_bds = (60,95),
			find_spikes = False,
		):

		stimFunc = get_stim_func_constPulse(test_spikeAmp, pulse_bds[0], pulse_bds[1])

		T,wf = HH.compute(stimFunc, bln_plot = False)

		if find_spikes:
			spikes = get_spike_times(T,wf[:,0])
			print('found spikes at:')
			print(spikes)
			plt.plot(spikes, [0.0] * len(spikes), 'ko')

		plot_time = T[ start_compute_idx : ]

		lbl, = plt.plot(plot_time, wf[ start_compute_idx:, 0])
		# plt.plot(plot_time, [stimFunc(t) for t in plot_time], 'r-')

		plt.xlabel('Time (ms)')
		plt.ylabel('Vm (mV)')
		# plt.title('Neuron potential with two spikes')
		plt.grid()

		return lbl



	#* plot lots

	# window_strt = 6.475
	# window_end = 6.49
	# window_step = 0.0001
	# lst_amps, lst_spikeLists, lst_timeArr, lst_waveforms = ds3_p1(np.arange(window_strt, window_end, window_step))

	#*
	# amps_detailed = np.array(
	# 	list(np.arange(6.25, 6.2625, 0.0005))
	# 	+ list(np.arange(6.2625, 6.2633, 0.0001)) 
	# 	+ list(np.arange(6.2633, 6.26425, 0.00005)) 
	# 	+ list(np.arange(6.26425, 6.27, 0.0005))
	# )

	# lst_amps, lst_spikeLists, lst_timeArr, lst_waveforms = ds3_p1(amps_detailed)

	#*
	# plot_amp_vs_numSpikes(lst_amps, lst_spikeLists)

	#*
	# plot_amp_vs_spikeTimes(lst_amps, lst_spikeLists)
	# plt.plot([window_strt, window_end], [100, 100], 'k-')


	#*
	# lbl_A = print_spike( test_spikeAmp = 6.2650, pulse_bds = (100,1100) )
	# lbl_B = print_spike( test_spikeAmp = 6.2600, pulse_bds = (100,1100) )

	# plt.legend([lbl_B, lbl_A], ['I_A = 6.2600', 'I_A = 6.2650'])

	#*
	# lbl_A = print_spike( test_spikeAmp = 6.485, pulse_bds = (100,1100) )
	# lbl_B = print_spike( test_spikeAmp = 6.480, pulse_bds = (100,1100) )
	# plt.legend([lbl_B, lbl_A], ['I_A = 6.480', 'I_A = 6.485'])
	# plt.title(r'Spiking behavior for $i_A$ near continuous spiking threshold')


	# plt.title(r'Spiking behavior for $i_A$ near rheobase threshold')
	# plt.legend([lbl_B, lbl_A], ['I_A = 6.2600', 'I_A = 2.2400'])
	# plt.title(r'Spiking behavior for $i_A$ near rheobase threshold')


	# plt.savefig('img/temp.png', dpi = 200, figsize = (2.0,6.0))
	plt.show()
