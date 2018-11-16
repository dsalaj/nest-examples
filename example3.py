import nest
from pylab import *       # for plotting (matplotlib)
from utils import *  # for generating poisson spike trains
from numpy import *       # for numerical operations
import nest.raster_plot
import nest.voltage_trace

nest.set_verbosity("M_WARNING") # surpress too much text output

DT = 0.1       # The time step of the simulation [msec]

def generate_stimulus(nchannels, Rs, jitter, Rbase, Tsim):
    # used by construct_input_population
    if Rs == 0.0:
        Soccur = array([])
    else:
        Soccur = poisson_generator(Rs, t_stop = Tsim*1000)/1000.0
    spikes = []
    for i in range(nchannels):
        s = append(poisson_generator(Rbase, t_stop = Tsim*1000)/1000.0,Soccur+jitter*random.randn(len(Soccur)))
        s.sort()
        s*=1000.0  # times in ms for NEST
        # round to simulation precision
        s *= 10
        s = s.round()+1.0
        s = s/10.0
        spikes.append(s)
    return spikes, Soccur

def generate_stimulus_sequence(nchannels, Rs, jitter, Rbase, Tsim):
    #  used by construct_input_population
    if Rs == 0.0:
        Soccur = array([])
    else:
        Soccur = poisson_generator(Rs, t_stop = Tsim*1000)/1000.0
    spikes = []
    #inp_neuron = SpikingInputNeuron()
    for i in range(nchannels):
        s = append(poisson_generator(Rbase, t_stop = Tsim*1000)/1000.0,i*0.001+Soccur+jitter*random.randn(len(Soccur)))
        s.sort()
        s*=1000.0  # times in ms for NEST
        # round to simulation precision
        s *= 10
        s = s.round()+1.0
        s = s/10.0
        spikes.append(s)
    return spikes, Soccur


def construct_input_population(Nin, jitter, Tsim, sequence, sequence_only=False):
    # This is a hack.
    # Because in Nest, one cannot connect spike generators with other
    # neurons with STDP synapses, we need to first connect them to a
    # pool of iaf_psc_exp neurons which are then serving as the input pool
    # The pool will produce approximately Poissonian spike trains with rate Rin
    # Nin...number of input neurons
    # jitter...jitter of population spikes
    # Tsim.....Total simulation time
    # sequence....if True, the stimulus will be shifted in neuron i by i msec
    # Returns:
    # spike_generators...the spike generators' GIDs
    # input_neurons...the input neurons' GIDs

    # create input population
        if sequence_only:
            inp_spikes, s_occur = generate_stimulus_sequence(int(Nin), 2.0, jitter, 0.000000001, Tsim)
        else:
            if sequence:
               inp_spikes, s_occur = generate_stimulus_sequence(int(Nin/2), 2.0, jitter, 8.0, Tsim)
            else:
               inp_spikes, s_occur = generate_stimulus(int(Nin/2), 2.0, jitter, 8.0, Tsim)
            inp_spikes_2, s_occur_2 = generate_stimulus(int(Nin/2), 0.0, 0e-3, 8.0, Tsim)

            inp_spikes += inp_spikes_2

        spike_generators = nest.Create("spike_generator", Nin)
        for (sg, sp) in zip(spike_generators, inp_spikes):
                sp = sp[sp>0]
                nest.SetStatus([sg],{'spike_times': sp})

        input_neurons = nest.Create("iaf_psc_delta",Nin)
        # Choose threshold very close to resting potential so that each spike in a Poisson generator
        # elicits one spike in the corresponding input neuron
        Vresting = -60.0
        nrn_params =     {"V_m": Vresting,     # Membrane potential in mV
                      "E_L": Vresting,     # Resting membrane potential in mV
                      "C_m": 1.0e4/40,           # Capacity of the membrane in pF
                      "tau_m": 0.5,      # Membrane time constant in ms
                      "V_th": -59.9999,     # Spike threshold in mV
                      "V_reset": Vresting, # Reset potential of the membrane in mV
                      "t_ref": .2   # refractory time in ms
                      }
        nest.SetStatus(input_neurons,nrn_params)
        # Connect Poisson generators to input neurons "one-to-one"
        nest.Connect(spike_generators,input_neurons,{'rule':'one_to_one'},syn_spec={'weight':0.1})
        return spike_generators, input_neurons


def perform_simulation(sequence, jitter=0.0, alpha=1.1, Wmax_fact=2, Tsim=200.0, W = 20.0e2, sequence_only=False,
                       N=200):
    """
    Performs the network simulation.
    sequence...If True, stimulus in input population will be sequential
    N........Number of input neurons
    jitter...Jitter on input population events
    alpha....Scaling factor of negative STDP window size A- = -alpha*A+
    W........Initial weight of synapses
    Wmax_fact.....Maximal synaptic weight is given by Wmax = W * Wmax_fact
    Tsim.....Simulation time
    """

    syn_param = {"weight": 2 * 1e3,
                 "lambda": 0.005,
                 "tau_plus": 30.,
                 "alpha": alpha,
                 "Wmax": Wmax_fact * W,
                 "mu_plus": 0.,
                 "mu_minus": 0.
                 }
    nest.CopyModel("stdp_synapse", "syn", syn_param)

    Rm = 1.
    Cm = 30.e3
    tau_m = Rm * Cm / 1000.0
    lif_params = {#"V_m": -60.,  # Membrane potential in mV
                  "E_L": -60.,  # Resting membrane potential in mV
                  "C_m": Cm,  # Capacity of the membrane in pF
                  "tau_m": tau_m,  # Membrane time constant in ms
                  "V_th": -45.,  # Spike threshold in mV
                  "V_reset": -60.,  # Reset potential of the membrane in mV
                  "t_ref": 2.,  # refractory time in ms
                  "tau_syn_ex": 10.0,
                  "tau_syn_in": 10.0,
                  "tau_minus": 30.0,
                  }
    lif_neuron = nest.Create("iaf_psc_exp", 1)
    nest.SetStatus(lif_neuron, lif_params)
    voltmeter = nest.Create('voltmeter', 1, {'withgid': True})
    spike_rec = nest.Create('spike_detector')
    in_spike_rec = nest.Create('spike_detector')


    # the following creates N input neurons and sets their spike trains during simulation
    spike_generators, input_neurons = construct_input_population(N, jitter, Tsim, sequence, sequence_only)

    nest.Connect(input_neurons, lif_neuron, syn_spec={"model": "syn"})

    # connect recorders
    # nest.Connect(voltmeter, input_neurons)
    nest.Connect(lif_neuron, spike_rec)
    nest.Connect(input_neurons, in_spike_rec)
    # nest.Connect(spike_generators, in_spike_rec)

    weight_evolution = []
    for i in range(int(Tsim)):
        nest.Simulate(1 * 1000.0)
        synapses = nest.GetConnections(input_neurons, lif_neuron)
        weight_evolution.append(nest.GetStatus(synapses, "weight"))
    weight_evolution = np.array(weight_evolution).transpose()

    # To extract spikes of input neuons as a list of numpy-arrays, use the
    # following function provided in nnb_utils:
    spikes_in = get_spike_times(in_spike_rec)

    # extract spike times and convert to [s]
    # events = nest.GetStatus(spike_rec, 'events')
    # spikes = events[0]['times']
    spikes = get_spike_times(spike_rec)

    # nest.raster_plot.from_device(in_spike_rec)
    # show()
    # nest.raster_plot.from_device(spike_rec)
    # # nest.voltage_trace.from_device(voltmeter)
    # show()
    # exit(1)

    return spikes, weight_evolution, spikes_in


def plot_raster(spikes,tmax):
    """
    Spike raster plot for spikes in 'spikes' up to time tmax [in sec]
    spikes[i]: spike times of neuron i in seconds
    """
    i = 0
    for spks in spikes:
        sp = spks[spks<tmax]
        ns = len(sp)
        plot(sp,i*ones(ns),'b.')
        i=i+1


color_cycle = ['#0000ff', '#ff0000', '#00ff00', '#ee7020', '#ff00ff', '#ddc040']


def plot_figures(fig1,fig2, spikes, weights, inp_spikes, Tsim, filename_fig1, filename_fig2, Tmax_spikes=25,
                 plot_index=None):
    """
    This function plots two figures for analysis of results
    fig1,fig2....figure identifiers
    spikes.......spikes of the output neuron
    weights......recorded weights over time (column t is the weight vector at recording time index t
                 The function assumes that weights are recorded every second.
    inp_spikes...spikes of input neurons as list of numpy arrays (use get_spike_times)
    Tsim.........simulation time
    filename_figX...filenames of figures to save figures to file
    Tmax_spikes.....Computation of spike cross-correlations may take some time
                    Use Tmax_spikes to compute the cc only over time (0,Tmax_spike)
    """
    # crop spike times in order to save time during convolution:
    Nin = len(weights)
    Nin2 = int(Nin/2)
    spikes = spikes[spikes<Tmax_spikes]
    for i in range(inp_spikes.__len__()):
        inp_spikes[i] = inp_spikes[i][inp_spikes[i]<Tmax_spikes]

    if plot_index is not None:
        f = figure(2, figsize=(8, 8))
        f.subplots_adjust(top=0.89, left=0.09, bottom=0.15, right=0.93, hspace=0.30, wspace=0.40)

        ax = subplot(1,1,1)
        mean_up = mean(weights[0:Nin2,:], axis = 0)
        mean_down = mean(weights[Nin2:Nin,:], axis = 0)
        std_up = std(weights[0:Nin2,:], axis = 0)
        std_down = std(weights[Nin2:Nin,:], axis = 0)
        ax.set_color_cycle(color_cycle[plot_index:plot_index + 2])
        plot_index += 2
        plot(linspace(0,Tsim,len(mean_up)), mean_up)
        plot(linspace(0,Tsim,len(mean_down)), mean_down)
        errorbar(linspace(0, Tsim, len(mean_up))[::20], mean_up[::20], std_up[::20], fmt = '.')
        errorbar(linspace(0, Tsim, len(mean_down))[::20], mean_down[::20], std_down[::20], fmt = '.')
        xlabel('time [sec]')
        ylabel('avg. syn. weight')
        text(-0.19, 1.07, 'B', fontsize = 'large', transform = ax.transAxes)

        savefig("comparison")

    f = figure(fig1, figsize = (8,3.6   ))
    f.subplots_adjust(top= 0.89, left = 0.09, bottom = 0.15, right = 0.93, hspace = 0.30, wspace = 0.40)

    ax = subplot(1,2,1)
    imshow(weights, aspect = 'auto')
    ylim(0,Nin+1)
    xlabel('time [sec]')
    colorbar()
    ylabel('synapse id.')
    text(-0.19, 1.07, 'A', fontsize = 'large', transform = ax.transAxes)

    ax = subplot(1,2,2)
    mean_up = mean(weights[0:Nin2,:], axis = 0)
    mean_down = mean(weights[Nin2:Nin,:], axis = 0)
    std_up = std(weights[0:Nin2,:], axis = 0)
    std_down = std(weights[Nin2:Nin,:], axis = 0)
    plot(linspace(0,Tsim,len(mean_up)), mean_up, color = 'b')
    plot(linspace(0,Tsim,len(mean_down)), mean_down, color = 'r')
    errorbar(linspace(0, Tsim, len(mean_up))[::20], mean_up[::20], std_up[::20], fmt = 'b.')
    errorbar(linspace(0, Tsim, len(mean_down))[::20], mean_down[::20], std_down[::20], fmt = 'r.')
    xlabel('time [sec]')
    ylabel('avg. syn. weight')
    text(-0.19, 1.07, 'B', fontsize = 'large', transform = ax.transAxes)

    savefig(filename_fig1)

    f = figure(fig2, figsize = (8,8))
    f.subplots_adjust(top= 0.93, left = 0.09, bottom = 0.12, right = 0.95, hspace = 0.40, wspace = 0.40)
    ax = subplot(2,2,1)
    corr = avg_cross_correlate_spikes(inp_spikes[0:Nin2], 200, binsize = 5e-3, corr_range = (-100e-3,100e-3))
    plot(arange(-100e-3,101e-3, 5e-3), corr, marker = 'o')
    xlim(-100e-3,100e-3)
    xticks(list(arange(-0.1,0.101,0.05)), [ '-0.1', '-0.05', '0', '0.05', '0.1' ] )
    xlabel('time lag [sec]')
    axvline(0.0)
    title('input correl. first group', fontsize = 15)
    ylabel('counts/bin')
    text(-0.19, 1.07, 'A', fontsize = 'large', transform = ax.transAxes)

    ax = subplot(2,2,2)
    corr = avg_cross_correlate_spikes(inp_spikes[Nin2:Nin], 200, binsize = 5e-3, corr_range = (-100e-3,100e-3))
    plot(arange(-100e-3,101e-3, 5e-3), corr, marker = 'o')
    xlim(-100e-3,100e-3)
    xticks(list(arange(-0.1,0.101,0.05)), [ '-0.1', '-0.05', '0', '0.05', '0.1' ] )
    xlabel('time lag [sec]')
    ylabel('counts/bin')
    title('input correl. second group', fontsize = 15)
    axvline(0.0)
    text(-0.19, 1.07, 'B', fontsize = 'large', transform = ax.transAxes)

    ax = subplot(2,2,3)
    corr = avg_cross_correlate_spikes_2sets(inp_spikes[0:Nin2], [spikes], binsize = 5e-3, corr_range = (-100e-3,100e-3))
    plot(arange(-100e-3,101e-3, 5e-3), corr, marker = 'o')
    xlim(-100e-3,100e-3)
    xticks(list(arange(-0.1,0.101,0.05)), [ '-0.1', '-0.05', '0', '0.05', '0.1' ] )
    xlabel('time lag [sec]')
    title('input-output correl. first group', fontsize = 15)
    ylabel('counts/bin')
    axvline(0.0)
    text(-0.19, 1.07, 'C', fontsize = 'large', transform = ax.transAxes)

    ax = subplot(2,2,4)
    corr = avg_cross_correlate_spikes_2sets(inp_spikes[Nin2:Nin], [spikes], binsize = 5e-3, corr_range = (-100e-3,100e-3))
    plot(arange(-100e-3,101e-3, 5e-3), corr, marker = 'o')
    xlim(-100e-3,95e-3)
    xticks(list(arange(-0.1,0.101,0.05)), [ '-0.1', '-0.05', '0', '0.05', '0.1' ] )
    title('input-output correl. second group', fontsize = 15)
    xlabel('time lag [sec]')
    ylabel('counts/bin')
    axvline(0.0)
    text(-0.19, 1.07, 'D', fontsize = 'large', transform = ax.transAxes)

    savefig(filename_fig2)
    close(fig1)
    close(fig2)


spikes, weight_evolution, spikes_in = perform_simulation(sequence=False, jitter=.002, alpha=1.1, Tsim=200.)
plot_raster(spikes, 25)
show()
plot_raster(spikes_in, 25)
show()
spikes = spikes[0]
plot_figures(0, 1, spikes, weight_evolution, spikes_in, 200., filename_fig1="ex3_b_fig1", filename_fig2="ex3_b_fig2")



nest.ResetKernel()
spikes, weight_evolution, spikes_in = perform_simulation(sequence=False, jitter=.002, alpha=1.0, Tsim=200.)
spikes = spikes[0]
plot_figures(0, 1, spikes, weight_evolution, spikes_in, 200.,
             filename_fig1="ex3_c_1_0_fig1", filename_fig2="ex3_c_1_0_fig2", plot_index=0)
nest.ResetKernel()
spikes, weight_evolution, spikes_in = perform_simulation(sequence=False, jitter=.002, alpha=1.3, Tsim=200.)
spikes = spikes[0]
plot_figures(0, 1, spikes, weight_evolution, spikes_in, 200.,
             filename_fig1="ex3_c_1_3_fig1", filename_fig2="ex3_c_1_3_fig2", plot_index=2)
nest.ResetKernel()
spikes, weight_evolution, spikes_in = perform_simulation(sequence=False, jitter=.002, alpha=2.5, Tsim=200.)
spikes = spikes[0]
plot_figures(0, 1, spikes, weight_evolution, spikes_in, 200.,
             filename_fig1="ex3_c_2_5_fig1", filename_fig2="ex3_c_2_5_fig2", plot_index=4)



spikes, weight_evolution, spikes_in = perform_simulation(sequence=True, jitter=.0, alpha=1.1, Tsim=200.)
spikes = spikes[0]
plot_figures(0, 1, spikes, weight_evolution, spikes_in, 200., filename_fig1="ex3_d_fig1", filename_fig2="ex3_d_fig2")

nest.ResetKernel()
spikes, weight_evolution, spikes_in = perform_simulation(sequence=True, jitter=.0, alpha=1.1, Tsim=200., Wmax_fact=1.5)
spikes = spikes[0]
plot_figures(0, 1, spikes, weight_evolution, spikes_in, 200.,
             filename_fig1="ex3_d_Wmax_fig1", filename_fig2="ex3_d_Wmax_fig2")



nest.ResetKernel()
spikes, weight_evolution, spikes_in = perform_simulation(sequence=True, jitter=.0, alpha=1.1, Tsim=400.,
                                                         sequence_only=True, N=52)
spikes = spikes[0]
plot_figures(0, 1, spikes, weight_evolution, spikes_in, 400.,
             filename_fig1="ex3_e_fig1", filename_fig2="ex3_e_fig2")
